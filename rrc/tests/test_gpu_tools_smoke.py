"""Smoke tests of the GPU first-run tools on CPU (DESIGN.md §15): the `rrc_selftest --report` JSON
(through the host build `rrc_selftest_host`), `rrc_bench` argument parsing and JSON shape (through
the host build `rrc_bench_host`, CPU backend) and `rrc/bench/torch_baseline.py` on a tiny local
model with `--device cpu`.

None of this is a benchmark or a GPU validation: the host builds run on the CPU, their JSON says
"smoke" and carries no timing claims (`measured_on` null, `tokens_per_s` null), which is checked.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import pytest

RRC_DIR = pathlib.Path(__file__).resolve().parents[1]
BASELINE = RRC_DIR / "bench" / "torch_baseline.py"


@pytest.fixture(scope="module")
def tools(rrc_build: pathlib.Path) -> dict[str, pathlib.Path]:
    out = {"selftest": rrc_build / "rrc_selftest_host", "bench": rrc_build / "rrc_bench_host"}
    for k, p in out.items():
        assert p.is_file(), f"{p} was not built by `make -C rrc cpu`"
    return out


@pytest.fixture(scope="module")
def tiny_qwen3(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    d = tmp_path_factory.mktemp("tiny") / "qwen3"
    subprocess.run([sys.executable, str(RRC_DIR / "tools" / "make_tiny_model.py"), "--variant", "qwen3", "--out", str(d)],
                   check=True, capture_output=True)
    return d


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


# ---- rrc_selftest --report -------------------------------------------------------------------------------


def test_selftest_report_e2e_pass(tools, tmp_path):
    rep = tmp_path / "report.json"
    r = run([tools["selftest"], "--backend", "emu", "--only", "e2e", "--e2e-filter", "qwen3/f32", "--report", rep,
             "--e2e-dir", tmp_path])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert r.stdout.rstrip().splitlines()[-1] == "rrc_selftest: PASS"
    d = json.loads(rep.read_text())
    assert d["schema"] == "rrc-selftest-report-v1" and d["tool"] == "rrc_selftest_host"
    assert d["complete"] is True and d["result"] == "PASS" and d["fatal_error"] is None
    assert "no GPU" in d["status_note"] and "no GPU" in d["device"]["name"]
    assert d["op_passes"] == []  # --only e2e
    e = d["e2e"]
    assert e["result"] == "PASS" and e["failures"] == 0 and e["checks"] == len(e["items"])
    names = {it["name"] for it in e["items"]}
    for want in ("logits", "loss", "grad", "grad_recompute_none", "det_forward", "det_grad", "train_loss",
                 "train_grad_norm", "train_params", "train_loss_decreases", "det_train"):
        assert f"qwen3/f32/{want}" in names, want
    for it in e["items"]:
        assert set(it) == {"name", "pass", "dev_err", "ref_err", "tol", "detail"}
        assert it["pass"] is True
        if it["name"].split("/")[-1] in ("logits", "loss", "grad", "train_loss", "train_params"):
            assert it["ref_err"] is not None and it["dev_err"] <= it["tol"]
    assert not list(tmp_path.glob("rrc_e2e_*")), "the e2e work directory was not removed"


def test_selftest_report_ops_and_paths(tools, tmp_path):
    rep = tmp_path / "report.json"
    r = run([tools["selftest"], "--backend", "emu", "--quick", "--only", "gemm", "--report", rep])
    assert r.returncode == 0, r.stdout[-3000:]
    d = json.loads(rep.read_text())
    assert d["complete"] and d["result"] == "PASS" and d["e2e"] is None  # an op filter disables the e2e stage
    names = [p["name"] for p in d["op_passes"]]
    assert names == ["main", "capped"]
    main = d["op_passes"][0]
    assert [o["op"] for o in main["ops"]] == ["gemm"] and main["ops"][0]["cases"] > 100
    g = main["gemm"]
    assert len(g["path_per_dtype_combination"]) == 8 and all(v for v in g["path_per_dtype_combination"].values())
    assert set(g["cases_per_layout_and_level"]) == {"plain", "batched", "bcastA", "bcastB", "interleavedC", "misaligned",
                                                     "fusedAC", "aligned"}
    assert main["summary"]["gemm_path_failures"] == 0 and main["summary"]["gemm_path_checks"] >= 9
    assert d["op_passes"][1]["gemm"]["chunked_cases"] > 0


def test_selftest_report_records_failure(tools, tmp_path):
    """The fault device (a 1e-3 error in the attention backward) must fail, and the report must say so."""
    rep = tmp_path / "report.json"
    r = run([tools["selftest"], "--backend", "fault", "--only", "e2e", "--e2e-filter", "qwen2/f32", "--report", rep])
    assert r.returncode == 1
    assert r.stdout.rstrip().splitlines()[-1] == "rrc_selftest: FAIL"
    d = json.loads(rep.read_text())
    assert d["complete"] and d["result"] == "FAIL" and d["e2e"]["result"] == "FAIL"
    failed = {it["name"] for it in d["e2e"]["items"] if not it["pass"]}
    assert "qwen2/f32/grad" in failed
    assert "qwen2/f32/logits" not in failed  # forward is untouched by the fault


def test_selftest_bad_args(tools):
    for args in (["--e2e-size", "huge"], ["--nonsense"], ["--e2e-steps", "-1"], ["--backend", "gpu"]):
        r = run([tools["selftest"], *args])
        assert r.returncode == 2, args


# ---- rrc_bench (host smoke build) ---------------------------------------------------------------------------


def test_bench_args(tools):
    b = tools["bench"]
    r = run([b, "--help"])
    assert r.returncode == 0 and "usage: rrc_bench" in r.stdout
    for args in ([], ["--steps", "2"], ["--synthetic-model", "gpt2"], ["--synthetic-model", "qwen3", "--model", "/x"],
                 ["--synthetic-model", "qwen3", "--seq-len", "abc"], ["--synthetic-model", "qwen3", "--recompute", "half"],
                 ["--synthetic-model", "qwen3", "--frobnicate", "1"], ["--synthetic-model", "qwen3", "--steps", "0"],
                 ["--synthetic-model", "qwen3", "--dtype", "fp8"], ["--synthetic-model"]):
        r = run([b, *args])
        assert r.returncode == 2, (args, r.stderr)
        assert "usage: rrc_bench" in r.stderr


def bench_json(tools, tmp_path, *args) -> dict:
    out = tmp_path / "bench.json"
    r = run([tools["bench"], *args, "--out", out])
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(out.read_text())


def test_bench_host_json_shape(tools, tiny_qwen3, tmp_path):
    d = bench_json(tools, tmp_path, "--model", tiny_qwen3, "--dtype", "f32", "--seq-len", "16", "--micro-batch", "2",
                   "--grad-accum", "2", "--warmup", "1", "--steps", "2", "--lora-r", "4", "--lora-alpha", "8",
                   "--targets", "q_proj,v_proj,down_proj", "--recompute", "none", "--attn-chunk", "8", "--ce-chunk", "8")
    assert d["schema"] == "rrc-bench-v1" and d["tool"] == "rrc_bench" and d["error"] is None
    assert d["status"].startswith("smoke") and d["measured_on"] is None and "not a GPU" in d["timing_source"]
    res = d["results"]
    assert res["tokens_per_s"] is None and res["step_time_s"] is None and res["wall_step_time_s"] is None
    assert res["smoke_wall_step_time_s"] > 0
    assert len(res["losses"]) == 3 and all(math.isfinite(x) for x in res["losses"])
    assert len(res["grad_norms"]) == 3 and res["grad_norms"][0] > 0
    c = d["config"]
    assert c["tokens_per_step"] == 16 * 2 * 2 and c["model_type"] == "qwen3" and c["base_dtype"] == "f32"
    assert c["recompute"] == "none" and c["gradient_checkpointing"] is False
    assert c["lora"]["target_modules"] == ["q_proj", "v_proj", "down_proj"] and c["lora"]["adapter_dtype"] == "f32"
    assert d["gemm"] is None and d["device"] is None


def test_bench_host_error_is_reported(tools, tmp_path):
    out = tmp_path / "bench.json"
    r = run([tools["bench"], "--model", tmp_path / "missing", "--steps", "1", "--out", out])
    assert r.returncode == 1
    d = json.loads(out.read_text())
    assert d["status"] == "error" and "missing" in d["error"] and d["measured_on"] is None


# ---- torch_baseline.py (CPU smoke test, not a benchmark) --------------------------------------------------


@pytest.mark.parametrize("gc", ["on", "off"])
def test_torch_baseline_cpu_smoke_same_keys(tools, tiny_qwen3, tmp_path, gc):
    out = tmp_path / f"torch_{gc}.json"
    common = ["--seq-len", "16", "--micro-batch", "2", "--grad-accum", "2", "--lora-r", "4", "--lora-alpha", "8",
              "--targets", "q_proj,v_proj,down_proj"]
    r = run([sys.executable, BASELINE, "--model", tiny_qwen3, "--device", "cpu", "--dtype", "f32", "--warmup", "0",
             "--steps", "1", "--gradient-checkpointing", gc, "--out", out, *common])
    assert r.returncode == 0, r.stderr[-3000:]
    t = json.loads(out.read_text())
    assert t["tool"] == "torch_baseline" and t["status"].startswith("smoke") and t["measured_on"] is None
    assert t["results"]["tokens_per_s"] is None and t["results"]["step_time_s"] is None
    assert len(t["results"]["losses"]) == 1 and math.isfinite(t["results"]["losses"][0])
    assert t["results"]["grad_norms"][0] > 0, "no LoRA gradient (frozen base + checkpointing)"
    assert t["config"]["gradient_checkpointing"] is (gc == "on")
    assert t["config"]["recompute"] == ("full" if gc == "on" else "none")
    assert "sdpa" in t["config"]["attention"] and t["config"]["lora"]["adapter_dtype"] == "f32"
    assert t["config"]["optimizer"]["impl"].startswith("torch.optim.AdamW (single)")
    rr = bench_json(tools, tmp_path, "--model", tiny_qwen3, "--dtype", "f32", "--warmup", "0", "--steps", "1",
                    "--recompute", "full" if gc == "on" else "none", *common)
    assert set(t) == set(rr)
    assert set(t["results"]) == set(rr["results"])
    assert set(t["config"]) == set(rr["config"]) and set(t["config"]["lora"]) == set(rr["config"]["lora"])
    assert set(t["config"]["optimizer"]) == set(rr["config"]["optimizer"])
    # the same workload: tokens per step, and the LoRA layout (trainable parameter count)
    assert t["config"]["tokens_per_step"] == rr["config"]["tokens_per_step"]
    assert t["config"]["lora"]["trainable_params"] == rr["config"]["lora"]["trainable_params"]


def test_torch_baseline_bad_args(tiny_qwen3):
    base = [sys.executable, BASELINE, "--model", tiny_qwen3]
    for args in (["--device", "cpu", "--targets", "q_proj,lm_head", "--gradient-checkpointing", "on"],
                 ["--device", "cpu"],  # --gradient-checkpointing is required: the output must state it
                 ["--device", "cpu", "--gradient-checkpointing", "maybe"],
                 ["--device", "cpu", "--gradient-checkpointing", "on", "--seq-len", "1"],
                 ["--device", "tpu", "--gradient-checkpointing", "on"]):
        r = run(base + args)
        assert r.returncode == 2, (args, r.stderr[-500:])
