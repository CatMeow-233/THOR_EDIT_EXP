"""Tests of the public surface of research_rl.capability and of its documentation.

* ``research_rl/capability/__init__.py``: ``__all__`` is exactly the union of the modules' public
  names (each module's ``__all__`` plus the shared types of ``types.py``), every name resolves to
  the object of its defining module, the ``trl_gate`` names are lazy (importing the package does
  not import ``trl_gate``, ``research_rl.trainers`` or TRL), the intra-package imports form a DAG
  and every module imports as the *first* capability module in a fresh interpreter.
* ``docs/research_rl/CAPABILITY.md`` section 4.2: every block marked ``<!-- quickstart:NAME -->``
  is executed verbatim (the Python blocks in one namespace, in order; the shell blocks as
  subprocesses with ``--port 0`` / ``--store`` / ``--output-dir`` pointed at temporary paths).

CPU only; tiny models are built in-process; nothing is downloaded. The numbers the quickstart
prints come from random tiny models and are smoke values, not results.
"""

from __future__ import annotations

import ast
import concurrent.futures
import importlib
import inspect
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import research_rl.capability as capability
from research_rl.capability import GateDecision
from research_rl.models.model_interface import fingerprint

REPO = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO / "research_rl" / "capability"
CAPABILITY_MD = REPO / "docs" / "research_rl" / "CAPABILITY.md"
README_MD = REPO / "docs" / "research_rl" / "README.md"
API_MD = REPO / "docs" / "research_rl" / "API.md"

EAGER_MODULES = (
    "types",
    "tasks",
    "stats",
    "gate",
    "alarm",
    "weights",
    "scores",
    "spaces",
    "es",
    "probe",
    "tracker",
    "loop",
    "operators",
    "search",
    "edits",
    "pilot",
    "calibration",
    "calibration_alarm",
)
LAZY_MODULE = "trl_gate"
ALL_MODULES = (*EAGER_MODULES, LAZY_MODULE)
QUICKSTART_BLOCKS = ("splits", "loop", "edits", "ui", "pilot")
SUBPROCESS_TIMEOUT = 300.0


# ============================================================================== helpers
def _module(name: str) -> Any:
    return importlib.import_module(f"research_rl.capability.{name}")


def _types_public_names() -> set[str]:
    """Classes defined in ``types.py`` (it has no ``__all__``; ``ChannelSelection`` is imported)."""
    mod = _module("types")
    return {
        n
        for n, obj in vars(mod).items()
        if not n.startswith("_") and inspect.isclass(obj) and obj.__module__ == mod.__name__
    }


def _module_public_names(name: str) -> set[str]:
    if name == "types":
        return _types_public_names()
    return set(_module(name).__all__)


def _run_python(code: str, *, timeout: float = SUBPROCESS_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter from the repository root (CPU, offline, 1 thread)."""
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _quickstart_blocks() -> dict[str, tuple[str, str]]:
    """``{name: (language, code)}`` of the ``<!-- quickstart:NAME -->`` blocks of CAPABILITY.md."""
    text = CAPABILITY_MD.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- quickstart:([a-z0-9_-]+) -->\n```(python|bash)\n(.*?)\n```\n", re.DOTALL)
    blocks: dict[str, tuple[str, str]] = {}
    for name, lang, code in pattern.findall(text):
        assert name not in blocks, f"duplicate quickstart block {name!r}"
        blocks[name] = (lang, code)
    return blocks


def _shell_argv(code: str) -> list[str]:
    """The single ``python ...`` command of a shell block, with the current interpreter."""
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert len(lines) == 1, f"a quickstart shell block must be one command, got {lines}"
    argv = shlex.split(lines[0])
    assert argv[0] == "python", argv
    return [sys.executable, *argv[1:]]


def _replace_option(argv: list[str], option: str, value: str) -> list[str]:
    assert option in argv, (option, argv)
    i = argv.index(option)
    return [*argv[: i + 1], value, *argv[i + 2 :]]


# ============================================================================== exports
def test_all_is_the_union_of_module_names() -> None:
    """Fails (on purpose) when a module's ``__all__`` changes and the package is not regenerated."""
    expected: set[str] = set()
    for name in ALL_MODULES:
        expected |= _module_public_names(name)
    exported = list(capability.__all__)
    assert len(exported) == len(set(exported)), "duplicate names in __all__"
    assert set(exported) == expected


def test_every_exported_name_resolves_to_its_module_object() -> None:
    owners: dict[str, list[str]] = {}
    for name in ALL_MODULES:
        for public in _module_public_names(name):
            owners.setdefault(public, []).append(name)
    for public in capability.__all__:
        value = getattr(capability, public)
        for owner in owners[public]:
            assert value is getattr(_module(owner), public), (public, owner)
    # a name exported by two modules is one object (a re-export, never a collision)
    for public, mods in owners.items():
        objs = {id(getattr(_module(m), public)) for m in mods}
        assert len(objs) == 1, (public, mods)


def test_star_import_and_dir_cover_all() -> None:
    namespace: dict[str, Any] = {}
    exec("from research_rl.capability import *", namespace)
    assert set(capability.__all__) <= set(namespace)
    assert set(capability.__all__) <= set(dir(capability))


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute"):
        capability.not_a_capability_name  # noqa: B018
    with pytest.raises(ImportError):
        exec("from research_rl.capability import not_a_capability_name", {})


def test_lazy_trl_gate_names_are_typed_objects() -> None:
    trl_gate = _module(LAZY_MODULE)
    from transformers import TrainerCallback

    from research_rl.trainers.rl_common import AdvantageShaper

    assert capability.CapabilityGateCallback is trl_gate.CapabilityGateCallback
    assert issubclass(capability.CapabilityGateCallback, TrainerCallback)
    assert issubclass(capability.DisagreementScreen, AdvantageShaper)
    assert capability.GATE_COMPONENT == trl_gate.GATE_COMPONENT


def test_import_package_is_light_and_trl_gate_loads_on_access() -> None:
    code = """
import json, sys
import research_rl.capability as c
before = {m: m in sys.modules for m in ("research_rl.capability.trl_gate", "research_rl.trainers", "trl")}
in_dict = "CapabilityGateCallback" in vars(c)
cls = c.CapabilityGateCallback
after = "research_rl.capability.trl_gate" in sys.modules
print(json.dumps({"before": before, "in_dict": in_dict, "after": after, "cls": cls.__module__}))
"""
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["before"] == {
        "research_rl.capability.trl_gate": False,
        "research_rl.trainers": False,
        "trl": False,
    }
    assert out["in_dict"] is False
    assert out["after"] is True
    assert out["cls"] == "research_rl.capability.trl_gate"


# ============================================================================== import graph
def _is_type_checking_block(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(path: Path) -> set[str]:
    """Absolute module names imported at runtime by ``path`` (``if TYPE_CHECKING:`` excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    def visit(node: ast.AST) -> None:
        if _is_type_checking_block(node):
            assert isinstance(node, ast.If)
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
        for sub in ast.iter_child_nodes(node):
            visit(sub)

    visit(tree)
    return found


def test_intra_package_imports_form_a_dag() -> None:
    prefix = "research_rl.capability."
    graph: dict[str, set[str]] = {}
    for name in ALL_MODULES:
        deps = {
            imp[len(prefix) :].split(".")[0]
            for imp in _runtime_imports(PACKAGE_DIR / f"{name}.py")
            if imp.startswith(prefix)
        }
        graph[name] = {d for d in deps if d in ALL_MODULES and d != name}
        # a submodule never imports the package root (that would run __init__ inside a submodule)
        assert "research_rl.capability" not in _runtime_imports(PACKAGE_DIR / f"{name}.py"), name
    assert graph["types"] == set(), "types.py must not depend on other capability modules"
    state: dict[str, int] = {}  # 1 = on the stack, 2 = done

    def visit(node: str, stack: list[str]) -> None:
        if state.get(node) == 2:
            return
        assert state.get(node) != 1, f"import cycle: {' -> '.join([*stack, node])}"
        state[node] = 1
        for dep in sorted(graph[node]):
            visit(dep, [*stack, node])
        state[node] = 2

    for name in ALL_MODULES:
        visit(name, [])
    # the eager part of the package never depends on trl_gate
    for name in EAGER_MODULES:
        assert LAZY_MODULE not in graph[name], name


def test_dependencies_do_not_import_capability_back() -> None:
    """Every other research_rl package (except the UI, which is built on capability) is a
    dependency of capability or unrelated to it; none of them may import capability."""
    offenders = []
    for path in sorted((REPO / "research_rl").rglob("*.py")):
        rel = path.relative_to(REPO / "research_rl").parts
        if rel[0] in ("capability", "ui"):
            continue
        if any(
            imp == "research_rl.capability" or imp.startswith("research_rl.capability.")
            for imp in _runtime_imports(path)
        ):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == []


_FIRST_IMPORT = """
import importlib, sys, types
pkg = types.ModuleType("research_rl.capability")
pkg.__path__ = [{path!r}]
sys.modules["research_rl.capability"] = pkg  # bare package: the module below is the entry point
mod = importlib.import_module("research_rl.capability.{name}")
public = getattr(mod, "__all__", [n for n in vars(mod) if not n.startswith("_")])
missing = [n for n in public if not hasattr(mod, n)]
assert not missing, missing
print("ok", "research_rl.capability.trl_gate" in sys.modules)
"""


@pytest.mark.integration
def test_every_module_imports_first_in_a_fresh_interpreter() -> None:
    """Each module is the first capability module imported (the package ``__init__`` is replaced
    by a bare package), so every module's own import order is exercised once (cycle check)."""

    def run(name: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        return name, _run_python(_FIRST_IMPORT.format(path=str(PACKAGE_DIR), name=name))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(run, ALL_MODULES))
    for name in ALL_MODULES:
        proc = results[name]
        assert proc.returncode == 0, f"{name}: {proc.stderr[-2000:]}"
        loaded_trl_gate = proc.stdout.strip().splitlines()[-1] == "ok True"
        assert loaded_trl_gate == (name == LAZY_MODULE), name


# ============================================================================== documentation
def test_quickstart_blocks_are_present() -> None:
    blocks = _quickstart_blocks()
    assert set(blocks) == set(QUICKSTART_BLOCKS)
    assert {n for n, (lang, _) in blocks.items() if lang == "python"} == {"splits", "loop", "edits"}


def test_spec_section_3_and_api_reference_present() -> None:
    text = CAPABILITY_MD.read_text(encoding="utf-8")
    assert text.index("## 3. Pilot kill criteria") < text.index("## 4. As built")
    api = API_MD.read_text(encoding="utf-8")
    for name in ("research_rl.capability", *(f"research_rl.capability.{m}" for m in ALL_MODULES)):
        assert f"## `{name}`" in api, name
    readme = README_MD.read_text(encoding="utf-8")
    assert "| `capability/` |" in readme and "CAPABILITY.md" in readme


UI_MODULES = ("api", "jobs", "loading", "runs", "server", "workspace")


class _NamedDefault:
    """Renders a long default that is a module constant by its name (as API.md does)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


def _api_signature(mod: Any, fn: Any) -> str:
    sig = inspect.signature(fn)
    consts = {id(v): k for k, v in vars(mod).items() if k.isupper() and len(repr(v)) > 80}
    params = [
        p.replace(default=_NamedDefault(consts[id(p.default)])) if id(p.default) in consts else p
        for p in sig.parameters.values()
    ]
    return str(sig.replace(parameters=params))


def _api_sections() -> dict[str, str]:
    text = API_MD.read_text(encoding="utf-8")
    pattern = r"^## `(research_rl[\w.]*)`\n(.*?)(?=^## `|^<!-- capability-api:end|\Z)"
    return {m.group(1): m.group(2) for m in re.finditer(pattern, text, re.S | re.M)}


@pytest.mark.parametrize(
    "modname",
    [
        "research_rl.ui",
        *(f"research_rl.ui.{m}" for m in UI_MODULES),
        *(f"research_rl.capability.{m}" for m in ALL_MODULES),
    ],
)
def test_api_reference_matches_public_names_and_signatures(modname: str) -> None:
    """API.md has a section per public module listing its ``__all__`` with the live signatures."""
    section = _api_sections().get(modname)
    assert section is not None, f"API.md has no section for {modname}"
    mod = importlib.import_module(modname)
    public = getattr(mod, "__all__", None) or [  # ``types`` has no ``__all__``: its own public names
        n for n, v in vars(mod).items() if not n.startswith("_") and getattr(v, "__module__", None) == modname
    ]
    for name in public:
        assert f"`{name}" in section, f"{modname}.{name} missing from API.md"
        obj = getattr(mod, name)
        if inspect.isfunction(obj):
            assert f"`{name}{_api_signature(mod, obj)}`" in section, f"{modname}.{name}: stale signature"
        elif inspect.isclass(obj) and obj.__module__ == mod.__name__:
            for attr, raw in vars(obj).items():
                fn = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
                if attr.startswith("_") or not inspect.isfunction(fn):
                    continue
                expected = f"`{attr}{_api_signature(mod, fn)}`"
                assert expected in section, f"{modname}.{name}.{attr}: stale signature"


@pytest.fixture
def quiet_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``tempfile.mkdtemp`` (used by the quickstart) creates its directories under ``tmp_path``."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    yield tmp_path


def test_quickstart_python_blocks_execute(quiet_tempdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _quickstart_blocks()
    namespace: dict[str, Any] = {"__name__": "capability_quickstart"}
    torch.manual_seed(0)

    # --- splits, sampler and score function
    exec(compile(blocks["splits"][1], "CAPABILITY.md:quickstart:splits", "exec"), namespace)
    items = namespace["items"]
    assert len(items) == 4 and all(it.slice == "t_rule/unseen_pairs" for it in items)
    assert namespace["score_fn"](namespace["model"], items) == [0.0] * 4  # random weights

    # --- one GatedLoop run with ESOperator
    start = {n: p.detach().clone() for n, p in namespace["model"].named_parameters()}
    exec(compile(blocks["loop"][1], "CAPABILITY.md:quickstart:loop", "exec"), namespace)
    records = namespace["records"]
    loop, es = namespace["loop"], namespace["es"]
    assert [r.decision for r in records] == [GateDecision.HOLD, GateDecision.HOLD]
    assert [r.reason for r in records] == ["no_transfer_gain", "no_transfer_gain"]
    assert [r.update_index for r in records] == [1, 3]
    assert es.forward_passes_total == 4 * 4 * (1 + 32)  # 4 updates x 4 members x (generate + 32)
    alarm = namespace["alarm"]
    assert alarm.min_rollouts == 32 and alarm.updates == 4
    assert alarm.skipped_updates == 0  # n_eff = 32 per update: every report is inside the scope
    params = dict(namespace["model"].named_parameters())
    moved = [n for n in es.parameter_names() if not torch.equal(params[n].detach(), start[n])]
    assert moved, "the ES steps must change the selected gains (non-constant fitness)"
    assert all(torch.equal(params[n].detach(), start[n]) for n in start if n not in es.parameter_names())
    log = namespace["workdir"] / "decisions.jsonl"
    assert [json.loads(line) for line in log.read_text().splitlines()] == [r.to_dict() for r in records]
    assert loop.decision_index == 2

    # --- VersionStore edit cycle
    exec(compile(blocks["edits"][1], "CAPABILITY.md:quickstart:edits", "exec"), namespace)
    store, root, version = namespace["store"], namespace["root"], namespace["version"]
    assert version.parent == root.id
    assert store.head == root.id and [v.id for v in store.history()] == [root.id]
    assert len(store.versions()) == 2
    assert fingerprint(namespace["edit_model"]) == root.fingerprint
    assert store.evaluation_count == 1
    assert namespace["workdir"].is_relative_to(quiet_tempdir)
    out = capsys.readouterr().out
    assert "forward passes, 0 backward passes" in out


def test_readme_capability_import_executes() -> None:
    text = README_MD.read_text(encoding="utf-8")
    section = text[text.index("## Capability-gated training") :]
    section = section[: section.index("\n## ", 1)]
    code = re.findall(r"```python\n(.*?)\n```", section, re.DOTALL)
    assert code
    for block in code:
        exec(block, {})


def _read_lines(stream: Any, out: queue.Queue[str]) -> None:
    for line in iter(stream.readline, ""):
        out.put(line)
    out.put("")


def test_quickstart_ui_tiny_demo(tmp_path: Path) -> None:
    """The documented UI command, on a free port and a temporary store: it binds 127.0.0.1, prints
    the token URL once and answers an authenticated status request (401 without the token)."""
    argv = [*_shell_argv(_quickstart_blocks()["ui"][1]), "--port", "0", "--store", str(tmp_path / "store")]
    assert "--tiny-demo" in argv
    env = dict(os.environ, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    env.pop("RESEARCH_RL_UI_TOKEN", None)  # the server generates its own token
    proc = subprocess.Popen(
        argv, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=_read_lines, args=(proc.stdout, lines), daemon=True)
    reader.start()
    try:
        url: str | None = None
        token = ""
        seen: list[str] = []
        while url is None:
            try:
                line = lines.get(timeout=120)
            except queue.Empty:
                pytest.fail(f"UI did not print its URL within 120 s: {seen}")
            if line == "":
                pytest.fail(f"UI exited early: {seen}")
            seen.append(line)
            m = re.search(r"open (http://\S+/)#token=([A-Za-z0-9_-]+)", line)
            if m:
                url = m.group(1)
                token = m.group(2)
        assert url.startswith("http://127.0.0.1:"), url
        assert any("tiny demo" in s for s in seen)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(url + "api/status", timeout=30)
        assert err.value.code == 401
        request = urllib.request.Request(url + "api/status", headers={"X-Session-Token": token})
        with urllib.request.urlopen(request, timeout=30) as resp:
            assert resp.status == 200
            status = json.loads(resp.read())
        assert status["tiny_demo"] is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        if proc.stdout is not None:
            proc.stdout.close()


@pytest.mark.integration
def test_quickstart_pilot_smoke(tmp_path: Path) -> None:
    """The documented pilot smoke command with a temporary ``--output-dir`` (a plumbing check)."""
    out = tmp_path / "pilot_smoke"
    argv = _replace_option(_shell_argv(_quickstart_blocks()["pilot"][1]), "--output-dir", str(out))
    assert "--smoke" in argv
    env = dict(os.environ, OMP_NUM_THREADS="2", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    for name in ("results.json", "summary.md", "start_model.safetensors", "pilot_config.json"):
        assert (out / name).is_file(), name
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["banner"] == capability.BANNER
    # one seed: every section-3 criterion is INCONCLUSIVE (the smoke run is not a result)
    verdicts = {
        k: v["verdict"] for k, v in results["criteria"].items() if isinstance(v, dict) and "verdict" in v
    }
    assert verdicts and set(verdicts.values()) == {capability.VERDICT_INCONCLUSIVE}


# ============================================================================== 4.3 limits
SECTION_3_SHA256 = "49a3d56e48de2b7025db21134f7d8381a075fd0cb7c632caa5a58251caa69714"


def _section(text: str, heading: str) -> str:
    """The body of the markdown section that starts at ``heading`` (up to the next heading of the
    same or a higher level)."""
    start = text.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    m = re.search(rf"\n#{{1,{level}}} ", text[start + len(heading) :])
    return text[start : start + len(heading) + m.start()] if m else text[start:]


def _normalized(text: str) -> str:
    """Whitespace-collapsed text (the docs are hard-wrapped)."""
    return " ".join(text.split())


def test_spec_section_3_is_unchanged() -> None:
    """Section 3 (the kill criteria, fixed before any run) is byte-identical to the original."""
    import hashlib

    text = CAPABILITY_MD.read_text(encoding="utf-8")
    section = text[text.index("## 3. Pilot kill criteria") : text.index("## 4. As built")].strip()
    assert hashlib.sha256(section.encode("utf-8")).hexdigest() == SECTION_3_SHA256


def _horizon_table() -> dict[str, list[str]]:
    """``{first cell: cells}`` of the table after ``<!-- horizon-table -->`` in section 4.3."""
    limits = _section(CAPABILITY_MD.read_text(encoding="utf-8"), "### 4.3 ")
    assert "<!-- horizon-table -->" in limits, "section 4.3 needs the horizon table"
    table = limits[limits.index("<!-- horizon-table -->") :].split("\n\n")[0]
    rows = [ln for ln in table.splitlines() if ln.startswith("|")][2:]  # skip header + rule
    parsed = {}
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        parsed[cells[0]] = cells
    return parsed


def _cell_int(cell: str) -> int:
    """``"24 495"`` -> 24495 (thin-space / space thousands separators allowed)."""
    digits = cell.replace(" ", "").replace(" ", "").replace(",", "")
    assert digits.isdigit(), cell
    return int(digits)


def _default_loop(config: Any, *, num_groups: int) -> Any:
    """A real ``GatedLoop`` with ``config``, the CHANNELS rollback unit with ``num_groups`` channel
    groups (0 = no group stage) and an ES operator (no update is run: only the start-of-run horizon
    check is exercised)."""
    from research_rl.capability import (
        CapabilityGate,
        CapabilityProbe,
        ChannelGainSpace,
        ChannelGroup,
        CompositionSplits,
        DisagreementAlarm,
        ESConfig,
        ESOperator,
        ExactAnswerVerifier,
        FitnessResult,
        GateController,
        GatedLoop,
        ItemSampler,
        OperatorSet,
        RollbackUnit,
        make_score_fn,
    )
    from research_rl.models.ffn_channels import ChannelSelection
    from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama

    tokenizer = build_char_tokenizer()
    splits = CompositionSplits.build(OperatorSet(seed=0, num_operators=10), seed=0)
    model = build_tiny_llama(len(tokenizer), seed=0)
    probe = CapabilityProbe(
        ItemSampler(splits, root_seed=1),
        make_score_fn(tokenizer, ExactAnswerVerifier(), max_new_tokens=8, batch_size=32),
        transfer_slices=("t_rule/unseen_pairs",),
        retention_slices=("retention/atomic",),
        items_per_transfer_slice=config.items_per_transfer_slice,
        items_per_retention_slice=config.items_per_retention_slice,
    )

    def fitness(m: Any, update_index: int, member_index: int) -> Any:
        return FitnessResult(reward_v1=0.0, reward_v2=0.0, v1_pass=(False,), v2_pass=(False,))

    space = ChannelGainSpace(model, ChannelSelection({0: [0, 1]}))
    es = ESOperator(space, fitness, ESConfig(population=4, seed=0))
    groups = [ChannelGroup(f"g{i}", ChannelSelection({1: [i]})) for i in range(num_groups)]
    controller = GateController(
        CapabilityGate(config), probe, rollback_unit=RollbackUnit.CHANNELS, channel_groups=groups or None
    )
    return GatedLoop(model, es, controller, DisagreementAlarm(rollouts_per_unit=4), config=config)


def _max_accepted_updates(loop: Any, limit: int = 100_000) -> int:
    """The largest ``num_updates`` for which ``loop.check_horizon`` does not raise."""
    best = -1
    for u in range(limit):
        try:
            loop.check_horizon(u)
        except ValueError:
            return best
        best = u
    raise AssertionError(f"no horizon limit below {limit} updates")


def test_horizon_table_matches_the_code() -> None:
    """Every figure of the 4.3 horizon table is what the code computes: the zero-discordance
    ceiling (``GateConfig.retention_horizon``), the verdict the ``GatedLoop`` start check allows
    (``GatedLoop.horizon``) and the largest ``run(num_updates)`` that check accepts."""
    from research_rl.capability import GateConfig
    from research_rl.ui.workspace import WorkspaceConfig

    table = _horizon_table()
    configs = {
        "`GateConfig()` defaults": GateConfig(),
        "UI defaults (`WorkspaceConfig`)": WorkspaceConfig().gate_config(),
    }
    assert set(table) == set(configs)
    for label, config in configs.items():
        cells = table[label]
        assert len(cells) == 7, cells
        assert _cell_int(cells[1]) == config.items_per_retention_slice, label
        assert float(cells[2]) == config.retention_margin, label
        assert config.alpha_schedule == "spending" and config.alpha == 0.05
        assert _cell_int(cells[3]) == config.retention_horizon(), label
        no_groups = _default_loop(config, num_groups=0)
        full = _default_loop(config, num_groups=config.max_group_rollbacks)
        assert no_groups.controller.max_verdicts_per_decision == 1 + len(config.shrink_lambdas)
        assert full.controller.max_verdicts_per_decision == (
            1 + len(config.shrink_lambdas) + config.max_group_rollbacks
        )
        assert no_groups.horizon() == full.horizon()
        assert _cell_int(cells[4]) == full.horizon(), label
        assert _cell_int(cells[5]) == _max_accepted_updates(no_groups), label
        assert _cell_int(cells[6]) == _max_accepted_updates(full), label
    limits = _normalized(_section(CAPABILITY_MD.read_text(encoding="utf-8"), "### 4.3 "))
    assert "`VersionStore.evaluate` runs no horizon check" in limits


def test_version_store_evaluate_has_no_horizon_check(tmp_path: Path) -> None:
    """Behaviour documented in 4.3: past the zero-discordance ceiling a ``VersionStore`` edit with
    no observed drop fails retention (``retention_ok`` False), with no warning and no label."""
    import warnings

    from research_rl.capability import (
        CapabilityProbe,
        CompositionSplits,
        EditOp,
        EditRecord,
        ExactAnswerVerifier,
        GateConfig,
        ItemSampler,
        OperatorSet,
        VersionStore,
        make_score_fn,
    )
    from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama

    config = GateConfig(retention_margin=0.3, items_per_transfer_slice=8, items_per_retention_slice=16)
    ceiling = config.retention_horizon()
    assert ceiling == 2
    tokenizer = build_char_tokenizer()
    splits = CompositionSplits.build(OperatorSet(seed=0, num_operators=10), seed=0)
    probe = CapabilityProbe(
        ItemSampler(splits, root_seed=2),
        make_score_fn(tokenizer, ExactAnswerVerifier(), max_new_tokens=8, batch_size=32),
        transfer_slices=("t_rule/unseen_pairs",),
        retention_slices=("retention/atomic",),
        items_per_transfer_slice=8,
        items_per_retention_slice=16,
    )
    store = VersionStore(tmp_path / "store", build_tiny_llama(len(tokenizer), seed=1))
    store.init_from_model(note="start")
    outcomes = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(ceiling + 1):
            staged = store.stage(EditRecord(op=EditOp.SCALE, selection={0: [0]}, params={"factor": 1.0}))
            verdict = store.evaluate(staged, config, probe)
            (retention,) = verdict.retention
            assert retention.point == 0.0  # the identity edit: no observed drop
            outcomes.append(verdict.retention_ok)
            store.discard(staged)  # a discarded evaluation still spends the store's ledger
    assert outcomes == [True] * ceiling + [False]
    assert not [w for w in caught if "horizon" in str(w.message).lower()]
    assert store.evaluation_count == ceiling + 1
    assert "horizon" not in json.dumps(verdict.to_dict()).lower()  # no horizon_exhausted label

    # remedy documented in 4.3: the same (exhausted) store passes again with a larger retention
    # slice; only alpha and schedule are pinned by the ledger
    larger = GateConfig(retention_margin=0.3, items_per_transfer_slice=8, items_per_retention_slice=64)
    assert larger.retention_horizon() is None or larger.retention_horizon() > ceiling + 1
    larger_probe = CapabilityProbe(
        ItemSampler(splits, root_seed=2),
        make_score_fn(tokenizer, ExactAnswerVerifier(), max_new_tokens=8, batch_size=32),
        transfer_slices=("t_rule/unseen_pairs",),
        retention_slices=("retention/atomic",),
        items_per_transfer_slice=8,
        items_per_retention_slice=64,
    )
    staged = store.stage(EditRecord(op=EditOp.SCALE, selection={0: [0]}, params={"factor": 1.0}))
    verdict = store.evaluate(staged, larger, larger_probe)
    assert verdict.retention[0].point == 0.0 and verdict.retention_ok
    store.discard(staged)

    # the UI quickstart points to it, with the evaluation index computed from the UI defaults
    from research_rl.ui.workspace import WorkspaceConfig

    first_unpassable = WorkspaceConfig().gate_config().retention_horizon()
    assert first_unpassable is not None
    ordinal = f"{first_unpassable + 1}th evaluation"
    text = CAPABILITY_MD.read_text(encoding="utf-8")
    ui_part = text[text.index("<!-- quickstart:ui -->") :]
    after_ui = _normalized(ui_part[: ui_part.index("**The pilot smoke run**")])
    assert "4.3" in after_ui and ordinal in after_ui, "the UI quickstart must point to the store horizon"
    limits = _normalized(_section(text, "### 4.3 "))
    assert ordinal in limits


def _clean_alarm_rate(items: int, members: int, min_rollouts: int, p_v1_only: float) -> float:
    """Share of 400 clean, stationary 200-update streams on which ``DisagreementAlarm`` fires at
    least once. Each update: ``items`` shared items, each V1-only with probability ``p_v1_only``
    (V2 never passes), replicated exactly across ``members`` ES members
    (``rollouts_per_unit=members``, so ``n_eff = items``). Seeded with ``random.Random(0)``."""
    import random

    from research_rl.capability import DisagreementAlarm, OperatorReport

    rng = random.Random(0)
    hits = 0
    streams = 400
    for _ in range(streams):
        alarm = DisagreementAlarm(min_rollouts=min_rollouts, rollouts_per_unit=members)
        fired = False
        for t in range(200):
            per_item = [rng.random() < p_v1_only for _ in range(items)]
            v1 = tuple(per_item * members)
            state = alarm.update(
                OperatorReport(
                    update_index=t,
                    reward_v1=0.0,
                    reward_v2=0.0,
                    v1_pass=v1,
                    v2_pass=tuple(False for _ in v1),
                    completion_lengths=tuple(8 for _ in v1),
                )
            )
            fired |= state.alarm
        hits += fired
    return hits / streams


def test_alarm_scope_example_figures() -> None:
    """The 4.3 "Alarm scope" example: an ES feed of 4 members x 8 shared items with
    ``min_rollouts=8`` (outside the calibrated scope) versus 32 independent items (inside it)."""
    figures = {
        (8, 4, 8, 0.05): "36 %",
        (8, 4, 8, 0.1): "64.5 %",
        (32, 1, 32, 0.05): "0 %",
        (32, 1, 32, 0.1): "0.25 %",
    }
    measured = {key: _clean_alarm_rate(*key) for key in figures}
    assert measured == {
        (8, 4, 8, 0.05): 0.36,
        (8, 4, 8, 0.1): 0.645,
        (32, 1, 32, 0.05): 0.0,
        (32, 1, 32, 0.1): 0.0025,
    }
    limits = _normalized(_section(CAPABILITY_MD.read_text(encoding="utf-8"), "### 4.3 "))
    scope = limits[limits.index("**Alarm scope.**") :]
    scope = scope[: scope.index("**", len("**Alarm scope.**"))]
    for text in figures.values():
        assert text in scope, text


def test_quickstart_alarm_is_inside_its_calibrated_scope() -> None:
    """The quickstart alarm keeps ``DisagreementAlarm``'s documented calibration scope
    (``min_rollouts`` >= the default 32 independent units per update) and 4.3 states that scope."""
    from research_rl.capability import DisagreementAlarm

    default_min = inspect.signature(DisagreementAlarm).parameters["min_rollouts"].default
    assert default_min == 32
    code = _quickstart_blocks()["loop"][1]
    tree = ast.parse(code)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DisagreementAlarm"
    ]
    assert len(calls) == 1
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    if "min_rollouts" in kwargs:
        value = kwargs["min_rollouts"]
        assert isinstance(value, ast.Constant) and value.value >= default_min
    for name in ("threshold", "drift"):
        assert name not in kwargs, f"the quickstart keeps the calibrated default {name}"
    limits = _normalized(_section(CAPABILITY_MD.read_text(encoding="utf-8"), "### 4.3 "))
    assert "**Alarm scope.**" in limits and "`n_eff >= 32`" in limits
