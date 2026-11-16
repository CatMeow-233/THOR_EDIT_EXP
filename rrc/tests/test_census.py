"""rrc_census (DESIGN.md §13): the JSON report and an independent check of its numbers.

Everything rrc_census prints is COMPUTED by a dry run (no device, nothing measured). These tests
check the report's fields, recompute the GEMM FLOPs of every role from the config with closed-form
formulas written here (independently of the C call sequence), and check the shape-only model's
parameter count against transformers' own count for the tiny checkpoints.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess

import pytest

import make_tiny_model as mtm  # noqa: E402  (rrc/tools, put on sys.path by conftest)

PRESETS = ["qwen3-8b", "qwen2.5-7b", "llama-3.1-8b"]


@pytest.fixture(scope="session")
def census(rrc_build: pathlib.Path) -> pathlib.Path:
    exe = rrc_build / "rrc_census"
    assert exe.is_file(), f"{exe} was not built"
    return exe


def run(census: pathlib.Path, *args: str) -> dict:
    p = subprocess.run([str(census), *args], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def expected_flops(m: dict, T: int, B: int, r: int, attn_chunk: int, ce_chunk: int) -> dict[str, float]:
    """Per-step GEMM FLOPs by role for recompute "full", all 7 LoRA targets on every layer, one
    document per row, every token labelled (the census batch), grad_accum 1."""
    H, I, L, V = m["hidden_size"], m["intermediate_size"], m["num_hidden_layers"], m["vocab_size"]
    hd, Hq, Hkv = m["head_dim"], m["num_attention_heads"], m["num_key_value_heads"]
    Q, KV, N = Hq * hd, Hkv * hd, B * T
    lin = H * Q + 2 * H * KV + Q * H + 2 * H * I + I * H  # multiply-adds per token, all six base GEMMs
    lin_nodown = lin - I * H
    base_fwd = 2 * N * (L * lin + (L - 1) * lin_nodown)  # forward + recompute without down_proj
    base_dx = 2 * N * L * lin
    n_sel = B * (T - 1)
    lm = 2 * n_sel * V * H
    # attention: one "set" = sum over query chunks of 2 R Kc hd Hq (scores, P V, dP, dV, dQ, dK alike)
    one_set = 0
    for q0 in range(0, T, attn_chunk):
        R = min(attn_chunk, T - q0)
        one_set += 2 * R * (q0 + R) * hd * Hq
    one_set *= B
    attn = L * (2 + 5) * one_set + (L - 1) * 2 * one_set  # forward, backward (recompute P + 4), recompute
    # LoRA (in, out) per module; forward t and y, recompute t (+ y except down), backward dB, dt, dA, dx
    mods = [(H, Q), (H, KV), (H, KV), (Q, H), (H, I), (H, I), (I, H)]
    lora = 0
    for i, (fi, fo) in enumerate(mods):
        fwd = 2 * N * r * (fi + fo)
        rec = 2 * N * r * fi + (0 if i == 6 else 2 * N * r * fo)
        bwd = 2 * N * r * (2 * fo + 2 * fi)
        lora += L * (fwd + bwd) + (L - 1) * rec
    return {
        "base_fwd": base_fwd,
        "base_dx": base_dx,
        "lm_head_fwd": lm,
        "lm_head_dx": lm,
        "attention": attn,
        "lora": lora,
    }


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_report_fields_and_flops(preset, census):
    T, r = 4096, 16
    d = run(census, "--preset", preset, "--seq-len", str(T), "--r", str(r), "--alpha", "32")
    assert d["status"] == "computed"
    assert "NOT_TESTED" in d["model"]["preset_note"] and "ASSUMED" in d["model"]["preset_note"]
    assert d["model"]["source"].startswith(f"preset:{preset}")
    g = d["gemm"]
    roles = {x["role"]: x for x in g["by_role"]}
    assert set(roles) == {"base_fwd", "base_dx", "lm_head_fwd", "lm_head_dx", "lora", "attention"}
    want = expected_flops(d["model"], T, 1, r, 256, 1024)
    for role, fl in want.items():
        assert roles[role]["flops"] == fl, (role, roles[role]["flops"], fl)
    assert g["flops_total"] == sum(want.values())
    assert sum(x["flops"] for x in g["by_combo"]) == g["flops_total"]
    assert 0.0 < g["native_bf16_fraction"] <= g["native_fraction"] <= 1.0
    # every GEMM against a frozen base weight is on a native bf16 x bf16 plan path (after §13's fix)
    for role in ("base_fwd", "base_dx", "lm_head_fwd", "lm_head_dx"):
        assert roles[role]["non_native_calls"] == 0, role
        assert roles[role]["flops_native_bf16"] == roles[role]["flops"], role
    assert g["base_weight"]["non_native_calls"] == 0
    mem = d["memory"]
    parts = ["weights_bytes", "rope_tables_bytes", "lora_params_grads_bytes", "adam_bytes", "workspace_bytes"]
    assert sum(mem[k] for k in parts) == mem["op_table_peak_bytes"]
    assert mem["weights_bytes"] == 2 * d["model"]["base_params"]  # bf16
    assert mem["lora_params_grads_bytes"] == 8 * d["run"]["lora"]["params"]
    assert mem["adam_bytes"] == 8 * d["run"]["lora"]["params"] + 4  # m, v and the norm scalar
    assert mem["cublaslt_workspace_bytes"] == 32 << 20
    assert mem["gemm_scratch_bytes"] <= 64 << 20  # the default cap (16 Mi f32)
    assert mem["peak_total_bytes"] == mem["op_table_peak_bytes"] + mem["gemm_scratch_bytes"] + mem[
        "sum_squares_partials_bytes"
    ] + mem["cublaslt_workspace_bytes"]
    t = d["transfers"]
    assert (t["sync"], t["memcpy_d2h"], t["memcpy_h2d"], t["alloc"], t["free"]) == (2, 2, 6, 0, 0)


def test_lora_param_count_matches_formula(census):
    d = run(census, "--preset", "llama-3.1-8b", "--seq-len", "64", "--r", "16", "--target", "cpu")
    m = d["model"]
    H, I, L = m["hidden_size"], m["intermediate_size"], m["num_hidden_layers"]
    Q, KV = m["num_attention_heads"] * m["head_dim"], m["num_key_value_heads"] * m["head_dim"]
    per_layer = 16 * ((H + Q) + 2 * (H + KV) + (Q + H) + 2 * (H + I) + (I + H))
    assert d["run"]["lora"]["params"] == L * per_layer
    # CPU target: no CUDA-internal buffers in the total
    assert d["memory"]["peak_total_bytes"] == d["memory"]["op_table_peak_bytes"]


@pytest.mark.parametrize("variant", ["llama", "qwen2_tied", "qwen3_hd32_bias", "llama_mqa"])
def test_shape_only_param_count_matches_transformers(variant, census, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    out = tmp_path / variant
    mtm.make_variant(variant, str(out), seed=0)
    d = run(census, "--config", str(out / "config.json"), "--seq-len", "8", "--target", "cpu", "--dtype", "f32")
    model = transformers.AutoModelForCausalLM.from_pretrained(str(out), torch_dtype=torch.float32)
    n_hf = sum(p.numel() for p in model.parameters())  # tied weights counted once, like rrc's storage
    assert d["model"]["base_params"] == n_hf
    assert d["memory"]["weights_bytes"] == 4 * n_hf


def test_refusals(census, tmp_path):
    p = subprocess.run([str(census), "--preset", "nope", "--seq-len", "8"], capture_output=True, text=True)
    assert p.returncode == 1 and "unknown preset" in p.stderr
    p = subprocess.run([str(census), "--preset", "qwen3-8b"], capture_output=True, text=True)
    assert p.returncode == 1  # --seq-len is required
    p = subprocess.run([str(census), "--preset", "qwen3-8b", "--seq-len", "50000"], capture_output=True, text=True)
    assert p.returncode == 2 and "exceeds the model's positions" in p.stderr
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "llama", "use_mrope": False}))
    p = subprocess.run([str(census), "--config", str(cfg), "--seq-len", "8"], capture_output=True, text=True)
    assert p.returncode == 2 and "use_mrope" in p.stderr
    p = subprocess.run([str(census), "--list-presets"], capture_output=True, text=True)
    assert p.returncode == 0 and all(x in p.stdout for x in PRESETS) and "NOT_TESTED" in p.stdout


def test_fraction_of_attention_flops_is_reported(census):
    """Sanity of the scale: at T = 4096 attention is a minority of the FLOPs, the LM head small."""
    d = run(census, "--preset", "qwen3-8b", "--seq-len", "4096", "--r", "16")
    roles = {x["role"]: x["flop_fraction"] for x in d["gemm"]["by_role"]}
    assert math.isclose(sum(roles.values()), 1.0, rel_tol=1e-12)
    assert 0.05 < roles["attention"] < 0.3 and roles["lm_head_fwd"] < 0.05
