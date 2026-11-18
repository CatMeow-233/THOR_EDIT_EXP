"""Tests of scripts/gpu/capability_p1.py (the P1 capability pilot script).

CPU only, no network: the end-to-end test runs ``--dry-run`` (a tiny in-process Llama in bf16 on
CPU, outputs labelled ``DRY_RUN_CPU``) with ``socket`` connections patched to raise. The real
``--model-dir`` path is exercised on CPU up to the CUDA check, and its loader on a tiny checkpoint
written to a temporary directory. A dry run is a plumbing check: nothing here says anything about
7-8B models or GPUs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from safetensors import safe_open

from research_rl.capability.pilot import BANNER, NOT_TESTED, arm_specs, build_splits
from research_rl.capability.tasks import ItemSampler
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama, save_tiny_model

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gpu" / "capability_p1.py"


def _load_script() -> ModuleType:
    """Import the script as a module; the offline variables it sets are restored afterwards so the
    rest of the test session is unaffected (the subprocess test checks them)."""
    saved = {
        k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM")
    }
    spec = importlib.util.spec_from_file_location("capability_p1_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


p1 = _load_script()


class _NoNetwork:
    """Socket patch: every connection attempt is recorded and refused."""

    def __init__(self) -> None:
        self.attempts: list[tuple[Any, ...]] = []

    def refuse(self, *args: Any, **kwargs: Any) -> Any:
        self.attempts.append(args)
        raise OSError("network access is disabled in this test")

    def install(self, mp: pytest.MonkeyPatch) -> None:
        mp.setattr(socket.socket, "connect", self.refuse)
        mp.setattr(socket.socket, "connect_ex", self.refuse)
        mp.setattr(socket, "create_connection", self.refuse)


def _main(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = p1.main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out[out.index("{\n") :]) if "{\n" in out else {}


# ============================================================================== dry run
@pytest.fixture(scope="module")
def dry_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, dict[str, Any], _NoNetwork, int]]:
    out = tmp_path_factory.mktemp("p1_dry")
    guard = _NoNetwork()
    with pytest.MonkeyPatch.context() as mp:
        guard.install(mp)
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("TRANSFORMERS_OFFLINE", "1")
        torch.set_num_threads(2)
        code = p1.main(["--dry-run", "--output-dir", str(out)])
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    yield out, results, guard, code


def test_dry_run_end_to_end_outputs(dry_run: tuple[Path, dict[str, Any], _NoNetwork, int]) -> None:
    out, results, guard, code = dry_run
    assert code == 0
    assert guard.attempts == []  # no network access was attempted
    for name in (
        "results.json",
        "summary.md",
        "memory_plan.json",
        "p1_config.json",
        "start_delta.safetensors",
    ):
        assert (out / name).is_file(), name
    config = p1.dry_run_config()
    logs = sorted(p.name for p in (out / "decisions").iterdir())
    assert logs == sorted(f"{s.file_stem}.DRY_RUN_CPU.jsonl" for s in arm_specs(config))
    for log in logs:
        lines = (out / "decisions" / log).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1  # one block = one decision
    assert results["status"] == "complete"
    assert results["pilot"] == "P1"
    assert set(results["criteria"]) >= {"gate", "subset", "operator", "low_start"}


def test_dry_run_labels_every_output(dry_run: tuple[Path, dict[str, Any], _NoNetwork, int]) -> None:
    out, results, _, _ = dry_run
    assert results["label"] == "DRY_RUN_CPU"
    assert results["measured_on"] == "DRY_RUN_CPU"  # never a CUDA device name in a dry run
    assert results["script_status"] == NOT_TESTED
    assert "NOT_TESTED" in results["script_note"]
    assert results["environment"]["gpu"] == NOT_TESTED
    assert results["environment"]["device"] == "cpu"
    for arm in results["arms"]:
        assert arm["gpu_peak_allocated_bytes"] == NOT_TESTED
        assert arm["decision_log"].endswith(".DRY_RUN_CPU.jsonl")
    for name in ("p1_config.json", "memory_plan.json"):
        data = json.loads((out / name).read_text(encoding="utf-8"))
        assert data["label"] == "DRY_RUN_CPU" and data["script_status"] == NOT_TESTED
    with safe_open(str(out / "start_delta.safetensors"), framework="pt") as f:
        meta = f.metadata()
    assert meta["label"] == "DRY_RUN_CPU" and meta["script_status"] == NOT_TESTED
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# Capability pilot P1 summary (DRY_RUN_CPU)")
    assert "NOT_TESTED" in summary and "DRY_RUN_CPU" in summary
    assert BANNER not in summary  # the P0 banner never labels a P1 output
    assert "P0 summary" not in summary


def test_dry_run_pipeline_semantics(dry_run: tuple[Path, dict[str, Any], _NoNetwork, int]) -> None:
    _, results, _, _ = dry_run
    config = p1.dry_run_config()
    block = config.base_gate_config().block_updates
    assert results["config"]["updates"] == block
    # token budget from the tokenizer check, never below the character contract
    tok = results["tokenizer_check"]
    splits = build_splits(config)
    assert results["config"]["max_new_tokens"] == tok["required_max_new_tokens"]
    assert tok["required_max_new_tokens"] >= tok["max_answer_tokens"] + 1
    assert tok["required_max_new_tokens"] >= splits.operators.max_len + 1
    # bf16 weights with fp32 masters in SFT and GRPO; ES over ChannelGainSpace
    assert results["model"]["dtype"] == str(torch.bfloat16)
    assert results["sft"]["optimizer"] == "HookedOptimizer(AdamW, master_weights=True)"
    assert results["sft"]["steps"] == config.sft_steps > 0
    by_op = {a["operator"]: a for a in results["arms"]}
    assert "MaskRule" in by_op["grpo"]["space"]["optimizer"]
    assert by_op["grpo"]["space"]["channels"] == results["selection"]["count"]
    assert by_op["es_informed"]["space"]["type"] == "ChannelGainSpace"
    for arm in results["arms"]:
        assert len(arm["blocks"]) == 1
        assert arm["counts"]["rollouts"] > 0
        # the arms really train: some update had signal and the operator's weights moved (bf16
        # served weights; GRPO through fp32 masters + MaskRule, ES through ChannelGainSpace)
        assert arm["updates_with_signal"] > 0, arm["key"]
        assert arm["weights_changed"] is True, arm["key"]
    assert by_op["grpo"]["counts"]["backward_calls"] == block
    assert by_op["es_informed"]["counts"]["backward_calls"] == 0
    # memory plan: computed, labelled, with the phases of the arms present
    plan = results["memory_plan"]
    assert plan["computed_not_measured"] is True
    assert plan["budget_bytes"] is None and plan["over_budget"] is None
    assert {"sft", "grpo", "es_informed"} <= set(plan["phases"])
    assert plan["device_peak_estimate_bytes"] > plan["parameter_bytes"] > 0
    assert plan["stage"] == "exact after channel selection"
    host = plan["host_bytes"]
    assert plan["host_peak_estimate_bytes"] == host["start_reset_snapshot"] + 2 * max(
        host["grpo_gate_snapshot"], host["es_informed_gate_snapshot"]
    )
    grpo = plan["phases"]["grpo"]
    assert grpo["parts"]["fp32_master"] == 4 * grpo["elements"]
    assert grpo["parts"]["adamw_moments_fp32"] == 8 * grpo["elements"]


# ============================================================================== refusals
def test_refuses_without_model_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    code, report = _main(["--output-dir", str(out)], capsys)
    assert code == p1.EXIT_REFUSED
    assert report["status"] == "REFUSED"
    assert "--model-dir is required" in report["reason"]
    assert report["script_status"] == NOT_TESTED
    assert not out.exists()


def test_refuses_pickle_only_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "pytorch_model.bin").write_bytes(b"not loaded")
    out = tmp_path / "out"
    for extra in ([], ["--dry-run"]):
        code, report = _main(["--model-dir", str(model_dir), "--output-dir", str(out), *extra], capsys)
        assert code == p1.EXIT_REFUSED
        assert report["status"] == "REFUSED"
        assert "no *.safetensors" in report["reason"] and "pytorch_model.bin" in report["reason"]
    assert not out.exists()


def test_refuses_hub_id_and_missing_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, report = _main(
        ["--model-dir", "meta-llama/Llama-3.1-8B", "--output-dir", str(tmp_path / "o")], capsys
    )
    assert code == p1.EXIT_REFUSED and "not an existing local directory" in report["reason"]
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "model.safetensors").write_bytes(b"")
    code, report = _main(["--model-dir", str(empty), "--output-dir", str(tmp_path / "o")], capsys)
    assert code == p1.EXIT_REFUSED and "no config.json" in report["reason"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA refusal")
def test_refuses_without_cuda_unless_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_dir = save_tiny_model(tmp_path / "tiny")
    out = tmp_path / "out"
    code, report = _main(["--model-dir", str(model_dir), "--output-dir", str(out)], capsys)
    assert code == p1.EXIT_NOT_TESTED
    assert report["status"] == NOT_TESTED
    assert "torch.cuda.is_available() is False" in report["reason"]
    assert not out.exists()


def test_help_states_not_tested() -> None:
    text = " ".join(p1.build_parser().format_help().split())
    assert "NOT_TESTED" in text and "DRY_RUN_CPU" in text
    assert "Nothing is ever downloaded" in text
    assert "NOT_TESTED" in (p1.__doc__ or "")


# ============================================================================== offline loading
def test_offline_env_is_set_before_transformers_import(tmp_path: Path) -> None:
    code = (
        "import importlib.util, json, sys\n"
        "assert 'transformers' not in sys.modules and 'huggingface_hub' not in sys.modules\n"
        f"spec = importlib.util.spec_from_file_location('p1', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['p1'] = m; spec.loader.exec_module(m)\n"
        "import os, huggingface_hub.constants as c\n"
        "print(json.dumps({'hub': os.environ['HF_HUB_OFFLINE'], 'tf': os.environ['TRANSFORMERS_OFFLINE'],"
        " 'const': bool(c.HF_HUB_OFFLINE), 'loaded': 'transformers' in sys.modules}))\n"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "OMP_NUM_THREADS": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report == {"hub": "1", "tf": "1", "const": True, "loaded": True}


class _LoadSpy:
    """Records every ``torch.load`` call and the kwargs of both ``from_pretrained`` calls (each
    recorded call is delegated to the real implementation)."""

    def __init__(self) -> None:
        self.torch_loads: list[str] = []
        self.pretrained: dict[str, dict[str, Any]] = {}

    def install(self, mp: pytest.MonkeyPatch) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        real_torch_load = torch.load

        def torch_load(*args: Any, **kwargs: Any) -> Any:
            self.torch_loads.append(str(args[0]) if args else str(kwargs.get("f")))
            return real_torch_load(*args, **kwargs)

        mp.setattr(torch, "load", torch_load)
        for cls in (AutoModelForCausalLM, AutoTokenizer):
            real = cls.from_pretrained

            def wrapper(*args: Any, _real: Any = real, _name: str = cls.__name__, **kwargs: Any) -> Any:
                self.pretrained[_name] = dict(kwargs)
                return _real(*args, **kwargs)

            mp.setattr(cls, "from_pretrained", wrapper)


def test_loader_is_offline_and_never_reads_pickle(tmp_path: Path) -> None:
    model_dir = save_tiny_model(tmp_path / "tiny")
    (model_dir / "pytorch_model.bin").write_bytes(b"garbage: would fail if it were unpickled")
    guard = _NoNetwork()
    spy = _LoadSpy()
    with pytest.MonkeyPatch.context() as mp:
        guard.install(mp)
        spy.install(mp)
        model, tokenizer, info = p1.load_local_checkpoint(str(model_dir), device="cpu")
        config = p1.dry_run_config()
        check = p1.check_tokenizer(tokenizer, build_splits(config), items_per_slice=8, root_seed=0)
    assert guard.attempts == []
    assert spy.torch_loads == []
    assert set(spy.pretrained) == {"AutoModelForCausalLM", "AutoTokenizer"}
    for name, kwargs in spy.pretrained.items():
        assert kwargs.get("local_files_only") is True, name
        assert kwargs.get("trust_remote_code") is False, name
    assert spy.pretrained["AutoModelForCausalLM"].get("use_safetensors") is True
    assert info["ignored_pickle_files"] == ["pytorch_model.bin"]
    assert info["safetensors"] == ["model.safetensors"]
    assert next(model.parameters()).dtype == torch.bfloat16
    assert info["num_layers"] == 2
    assert check["required_max_new_tokens"] == build_splits(config).operators.max_len + 1


def _sharded_checkpoint(tmp_path: Path, shard_name: str, *, pickle_shard: bool) -> Path:
    """Tiny checkpoint whose weights live in one shard named ``shard_name`` behind a
    ``model.safetensors.index.json``; ``pickle_shard`` writes the shard with ``torch.save``."""
    from safetensors.torch import load_file, save_file

    model_dir = save_tiny_model(tmp_path / "tiny")
    state = load_file(str(model_dir / "model.safetensors"))
    (model_dir / "model.safetensors").unlink()
    if pickle_shard:
        torch.save(state, model_dir / shard_name)
    else:
        save_file(state, str(model_dir / shard_name))
    index = {"metadata": {}, "weight_map": {k: shard_name for k in state}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return model_dir


def test_sharded_safetensors_index_loads_without_pickle(tmp_path: Path) -> None:
    model_dir = _sharded_checkpoint(tmp_path, "model-00001-of-00001.safetensors", pickle_shard=False)
    spy = _LoadSpy()
    with pytest.MonkeyPatch.context() as mp:
        spy.install(mp)
        _, _, info = p1.load_local_checkpoint(str(model_dir), device="cpu")
    assert spy.torch_loads == []
    assert info["safetensors"] == ["model-00001-of-00001.safetensors"]


def test_refuses_index_pointing_at_pickle_shard(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from safetensors.torch import save_file

    model_dir = _sharded_checkpoint(tmp_path, "model-00001-of-00001.bin", pickle_shard=True)
    save_file({"dummy": torch.zeros(1)}, str(model_dir / "unused.safetensors"))
    spy = _LoadSpy()
    with pytest.MonkeyPatch.context() as mp:
        spy.install(mp)
        with pytest.raises(p1.P1Refusal, match="non-safetensors shards"):
            p1.load_local_checkpoint(str(model_dir), device="cpu")
    assert spy.torch_loads == [] and spy.pretrained == {}
    out = tmp_path / "out"
    code, report = _main(["--model-dir", str(model_dir), "--output-dir", str(out)], capsys)
    assert code == p1.EXIT_REFUSED and report["status"] == "REFUSED"
    assert "model-00001-of-00001.bin" in report["reason"]
    assert not out.exists()


@pytest.mark.parametrize(
    "weight_map",
    [
        {"w": "../outside.safetensors"},
        {"w": "/abs/model.safetensors"},
        {"w": "sub/x.safetensors"},
        {"w": 3},
        {},
    ],
)
def test_refuses_index_with_bad_shard_names(tmp_path: Path, weight_map: dict[str, Any]) -> None:
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    with pytest.raises(p1.P1Refusal, match=r"weight_map|bare file names"):
        p1.validate_model_dir(str(model_dir))
    (model_dir / "model.safetensors.index.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(p1.P1Refusal, match="not readable JSON"):
        p1.validate_model_dir(str(model_dir))


def test_refuses_non_standard_safetensors_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_dir = save_tiny_model(tmp_path / "tiny")
    (model_dir / "model.safetensors").rename(model_dir / "consolidated.safetensors")
    (model_dir / "pytorch_model.bin").write_bytes(b"garbage: would fail if it were unpickled")
    out = tmp_path / "out"
    code, report = _main(["--model-dir", str(model_dir), "--output-dir", str(out)], capsys)
    assert code == p1.EXIT_REFUSED and report["status"] == "REFUSED"
    assert "consolidated.safetensors" in report["reason"] and "pytorch_model.bin" in report["reason"]
    assert not out.exists()


def test_refuses_weight_override_and_adapter_dirs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_dir = save_tiny_model(tmp_path / "tiny")
    config_text = (model_dir / "config.json").read_text(encoding="utf-8")
    config = json.loads(config_text)
    config["transformers_weights"] = "adapter_model.bin"
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "adapter_model.bin").write_bytes(b"garbage: would fail if it were unpickled")
    out = tmp_path / "out"
    code, report = _main(["--model-dir", str(model_dir), "--output-dir", str(out)], capsys)
    assert code == p1.EXIT_REFUSED and "transformers_weights" in report["reason"]
    (model_dir / "config.json").write_text(config_text, encoding="utf-8")
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "x"}), encoding="utf-8"
    )
    code, report = _main(["--model-dir", str(model_dir), "--output-dir", str(out)], capsys)
    assert code == p1.EXIT_REFUSED and "PEFT adapter" in report["reason"]
    (model_dir / "config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(p1.P1Refusal, match="not a JSON object"):
        p1.validate_model_dir(str(model_dir))
    assert not out.exists()


def test_tokenizer_check_refuses_unknown_characters() -> None:
    tokenizer = build_char_tokenizer(chars="0123456789()=,")  # no letters: operator names become <unk>
    config = p1.dry_run_config()
    with pytest.raises(p1.P1Refusal, match="does not round-trip"):
        p1.check_tokenizer(tokenizer, build_splits(config), items_per_slice=4, root_seed=0)


# ============================================================================== units
def test_memory_budget_enforced() -> None:
    tokenizer = build_char_tokenizer()
    model = build_tiny_llama(len(tokenizer), seed=0).to(torch.bfloat16)
    config = p1.dry_run_config()
    options = p1.P1Options(dry_run=True, sft_scope="all")
    plan = p1.memory_plan(
        model,
        config,
        sft_names=p1.sft_parameter_names(model, options),
        selections={},
        reset_names=[],
        device_total_bytes=1000,
        fraction=0.9,
    )
    assert plan["budget_bytes"] == 900 and plan["over_budget"] is True
    params = sum(p.numel() for p in model.parameters())
    assert plan["phases"]["sft"]["elements"] == params
    assert plan["phases"]["sft"]["total_bytes"] == params * (4 + 8 + 2 + 4)
    with pytest.raises(p1.P1Refusal, match="exceeds the budget"):
        p1.enforce_budget(plan, options, [])
    seen: list[str] = []
    p1.enforce_budget(plan, p1.P1Options(dry_run=True, allow_over_budget=True), seen)
    assert len(seen) == 1 and "exceeds the budget" in seen[0]


def test_config_and_parsers() -> None:
    assert p1.parse_layers("3-5,1,3") == (1, 3, 4, 5)
    with pytest.raises(ValueError):
        p1.parse_layers("5-3")
    assert p1.parse_arms("grpo,es_informed+gate_on") == (
        "grpo+gate_off",
        "grpo+gate_on",
        "es_informed+gate_on",
    )
    with pytest.raises(ValueError):
        p1.parse_arms("adam")
    with pytest.raises(p1.P1Refusal, match="ChannelGainSpace"):
        p1.check_p1_config(p1.dry_run_config(es_space="weight"))
    with pytest.raises(p1.P1Refusal, match="all_channels"):
        p1.check_p1_config(p1.dry_run_config(es_full_space="ffn_weights"))
    gpu = p1.gpu_config()
    p1.check_p1_config(gpu)
    assert gpu.es_space == "gain" and gpu.jobs == 1
    tokenizer = build_char_tokenizer()
    model = build_tiny_llama(len(tokenizer), seed=0)
    names = p1.sft_parameter_names(model, p1.P1Options(sft_layers=(1,)))
    assert names == [f"model.layers.1.mlp.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj")]


def test_channel_layer_restriction() -> None:
    config = p1.dry_run_config()
    model, tokenizer = p1.build_dry_run_model(config)
    splits = build_splits(config)
    sampler = ItemSampler(splits, root_seed=0)
    budget = splits.operators.max_len + 1
    info = p1.select_channels(model, tokenizer, sampler, config, budget, (1,))
    assert set(info["selection"]) == {"1"}
    restriction = info["restriction"]
    assert restriction["global_k"] == config.informed_k * 2  # 2 layers, 1 allowed
    assert restriction["count"] == info["count"] == len(info["selection"]["1"]) > 0
    with pytest.raises(p1.P1Refusal, match="out of range"):
        p1.select_channels(model, tokenizer, sampler, config, budget, (7,))
    selections = p1.arm_selections(model, config, info)
    assert set(selections) == {"grpo", "es_informed"}
    names = p1.reset_parameter_names(model, selections)
    assert names == [f"model.layers.1.mlp.{proj}.weight" for proj in ("down_proj", "gate_proj", "up_proj")]


def test_dry_run_ffn_scope_with_layer_restriction(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The path the GPU instructions use: FFN-scope SFT on chosen layers, channels restricted."""
    out = tmp_path / "out"
    guard = _NoNetwork()
    with pytest.MonkeyPatch.context() as mp:
        guard.install(mp)
        code, report = _main(
            [
                "--dry-run",
                "--output-dir",
                str(out),
                "--sft-scope",
                "ffn",
                "--sft-layers",
                "1",
                "--channel-layers",
                "1",
                "--sft-steps",
                "200",
            ],
            capsys,
        )
    assert code == 0 and report["label"] == "DRY_RUN_CPU"
    assert guard.attempts == []
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["sft"]["trained_tensors"] == 3
    with safe_open(str(out / "start_delta.safetensors"), framework="pt") as f:
        assert sorted(f.keys()) == [
            f"model.layers.1.mlp.{p}.weight" for p in ("down_proj", "gate_proj", "up_proj")
        ]
    assert results["selection"]["restriction"]["channel_layers"] == [1]
    assert set(results["selection"]["selection"]) == {"1"}
    by_op = {a["operator"]: a for a in results["arms"]}
    assert by_op["grpo"]["space"]["layers"] == [1]
    assert results["memory_plan"]["phases"]["grpo"]["layers"] == [1]
    for arm in results["arms"]:
        assert len(arm["blocks"]) == 1
    assert by_op["es_informed"]["weights_changed"] is True


def test_refuses_bad_layers_and_config_before_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    code, report = _main(["--dry-run", "--output-dir", str(out), "--sft-layers", "5"], capsys)
    assert code == p1.EXIT_REFUSED and "--sft-layers [5] out of range" in report["reason"]
    code, report = _main(["--dry-run", "--output-dir", str(out), "--channel-layers", "9"], capsys)
    assert code == p1.EXIT_REFUSED and "--channel-layers [9] out of range" in report["reason"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"no_such_field": 1}), encoding="utf-8")
    code, report = _main(["--dry-run", "--output-dir", str(out), "--config", str(bad)], capsys)
    assert code == p1.EXIT_REFUSED and "invalid configuration" in report["reason"]
    weight = tmp_path / "weight.json"
    weight.write_text(json.dumps({"es_space": "weight"}), encoding="utf-8")
    code, report = _main(["--dry-run", "--output-dir", str(out), "--config", str(weight)], capsys)
    assert code == p1.EXIT_REFUSED and "ChannelGainSpace" in report["reason"]
    assert not out.exists()


def test_gpu_summary_labels(dry_run: tuple[Path, dict[str, Any], _NoNetwork, int]) -> None:
    """The P1_GPU branch of the summary (rendered from a relabelled copy of the dry-run results;
    no GPU numbers are produced here)."""
    _, results, _, _ = dry_run
    doctored = json.loads(json.dumps(results))
    doctored.update(
        {"label": "P1_GPU", "banner": p1.P1_BANNERS["P1_GPU"], "measured_on": "<cuda device name>"}
    )
    doctored["model"]["model_dir"] = "/models/example"
    text = p1.render_p1_summary(doctored)
    assert text.startswith("# Capability pilot P1 summary (P1_GPU)")
    assert "P1_GPU on <cuda device name>" in text and "Script status NOT_TESTED" in text
    assert "throughput NOT_TESTED" in text
    assert BANNER not in text and "tiny Llama" not in text
