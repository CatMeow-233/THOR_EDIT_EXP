"""The C loader (config.json + safetensors) compared with transformers on tiny random models.

rrc_inspect (rrc/src/rrc_inspect.c) prints what the C side derives; transformers is the reference:
derived config fields, biases / q-k norm presence, the exact checkpoint tensor set and shapes,
per-tensor sums, RoPE inverse frequencies and cos/sin tables. A table of config.json variants
checks the precedence and default rules (rope_scaling vs rope_parameters, rope_theta location,
llama3 parameters, kv-head and head_dim defaults) and the values the engine refuses.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

import make_tiny_model as mtm  # noqa: E402  (rrc/tools, put on sys.path by conftest)

POSITIONS = [0, 1, 7, 63, 64, 200, 255]


def run_inspect(exe: pathlib.Path, model_dir: pathlib.Path, *extra: str) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        [str(exe), *extra, "--positions", ",".join(map(str, POSITIONS)), str(model_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.stdout, f"rrc_inspect printed nothing (rc={proc.returncode}, stderr={proc.stderr})"
    return proc.returncode, json.loads(proc.stdout)


def hf_reference(model_dir: pathlib.Path) -> dict[str, Any]:
    """Everything the engine must derive, taken from transformers objects (not from config.json)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    attn = model.model.layers[0].self_attn
    rotary = model.model.rotary_emb
    rp = dict(cfg.rope_parameters)
    hd = attn.head_dim
    pos = torch.tensor([POSITIONS], dtype=torch.long)
    cos, sin = rotary(torch.zeros(1, dtype=torch.float32), pos)
    ref = {
        "model_type": cfg.model_type,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": hd,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rms_norm_eps": model.model.norm.variance_epsilon,
        "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        "qkv_bias": attn.q_proj.bias is not None,
        "o_bias": attn.o_proj.bias is not None,
        "qk_norm": hasattr(attn, "q_norm"),
        "rope_type": rp["rope_type"],
        "rope_theta": float(rp["rope_theta"]),
        "rope_attention_scaling": float(rotary.attention_scaling),
        "inv_freq": rotary.inv_freq.double().tolist(),
        "cos": cos[0, :, : hd // 2].double().tolist(),
        "sin": sin[0, :, : hd // 2].double().tolist(),
        "k_bias": attn.k_proj.bias is not None,
        "v_bias": attn.v_proj.bias is not None,
        "k_norm": hasattr(attn, "k_norm"),
        "eps_q_norm": attn.q_norm.variance_epsilon if hasattr(attn, "q_norm") else None,
        "head_dim_attn_scaling": attn.scaling,
    }
    if rp["rope_type"] == "llama3":
        ref.update(
            rope_factor=float(rp["factor"]),
            rope_low_freq_factor=float(rp["low_freq_factor"]),
            rope_high_freq_factor=float(rp["high_freq_factor"]),
            rope_original_max_position_embeddings=int(rp["original_max_position_embeddings"]),
        )
    # every layer has the same structure
    for layer in model.model.layers:
        assert (layer.self_attn.q_proj.bias is not None) == ref["qkv_bias"]
    return ref


CONFIG_KEYS = [
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "tie_word_embeddings",
    "qkv_bias",
    "o_bias",
    "qk_norm",
    "rope_type",
    "rope_theta",
    "rope_attention_scaling",
]


def compare_config(rrc: dict[str, Any], ref: dict[str, Any]) -> None:
    c = rrc["config"]
    for k in CONFIG_KEYS:
        assert c[k] == ref[k], f"{k}: rrc {c[k]!r} != transformers {ref[k]!r}"
    assert c["rms_norm_eps"] == pytest.approx(ref["rms_norm_eps"], rel=0, abs=0)
    # q/k/v biases always come together; q_norm and k_norm too; q/k norm eps = rms_norm_eps
    assert ref["k_bias"] == ref["v_bias"] == ref["qkv_bias"]
    assert ref["k_norm"] == ref["qk_norm"]
    if ref["eps_q_norm"] is not None:
        assert ref["eps_q_norm"] == c["rms_norm_eps"]
    assert ref["head_dim_attn_scaling"] == pytest.approx(c["head_dim"] ** -0.5, rel=1e-12)
    if ref["rope_type"] == "llama3":
        for k in ("rope_factor", "rope_low_freq_factor", "rope_high_freq_factor", "rope_original_max_position_embeddings"):
            assert c[k] == ref[k], f"{k}: rrc {c[k]!r} != transformers {ref[k]!r}"
    # transformers evaluates inv_freq in float32; rrc in double (then rounds to f32 for the tables)
    assert len(rrc["inv_freq"]) == len(ref["inv_freq"]) == c["head_dim"] // 2
    for i, (a, b) in enumerate(zip(rrc["inv_freq"], ref["inv_freq"])):
        assert a == pytest.approx(b, rel=1e-6), f"inv_freq[{i}]: rrc {a} transformers {b}"
    assert rrc["inv_freq_sum"] == pytest.approx(sum(ref["inv_freq"]), rel=1e-6)
    # cos/sin tables; angle = position * inv_freq differs by the inv_freq rounding (|pos| <= 255)
    for name in ("cos", "sin"):
        got, want = rrc["rope"][name], ref[name]
        assert len(got) == len(POSITIONS)
        for p, (gr, wr) in enumerate(zip(got, want)):
            for i, (a, b) in enumerate(zip(gr, wr)):
                assert a == pytest.approx(b, abs=3e-5), f"{name}[pos {POSITIONS[p]}][{i}]: rrc {a} transformers {b}"


# ---------------------------------------------------------------------------------------------
# Tiny checkpoints written by save_pretrained
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(mtm.VARIANTS))
def test_tiny_checkpoint_matches_transformers(variant: str, tmp_path: pathlib.Path, rrc_inspect: pathlib.Path) -> None:
    d = tmp_path / variant
    mtm.make_variant(variant, str(d))
    rc, out = run_inspect(rrc_inspect, d)
    assert rc == 0 and out["ok"], out
    ref = hf_reference(d)
    compare_config(out, ref)

    tensors = safetensors_torch.load_file(str(d / "model.safetensors"))
    tied = ref["tie_word_embeddings"]
    # save_pretrained omits lm_head.weight exactly when the embeddings are tied
    assert ("lm_head.weight" in tensors) == (not tied)
    # the expected set equals the file's tensor set, with shapes
    expected = {t["name"]: t["shape"] for t in out["expected_tensors"]}
    assert expected == {k: list(v.shape) for k, v in tensors.items()}
    # and equals the model's state dict (minus the tied alias)
    model = transformers.AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
    sd_names = set(model.state_dict()) - ({"lm_head.weight"} if tied else set())
    assert set(expected) == sd_names
    assert out["check"] == "ok" and out["check_status"] == "RRC_OK"
    got = {t["name"]: t for t in out["tensors"]}
    assert set(got) == set(tensors)
    for name, t in tensors.items():
        assert got[name]["dtype"] == "F32"
        assert got[name]["shape"] == list(t.shape)
        assert got[name]["sum"] == pytest.approx(t.double().sum().item(), rel=1e-6, abs=1e-6), name


def test_bf16_and_sharded_checkpoint(tmp_path: pathlib.Path, rrc_inspect: pathlib.Path) -> None:
    src = tmp_path / "src"
    mtm.make_variant("qwen2", str(src))
    model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    out_dir = tmp_path / "bf16_sharded"
    model.save_pretrained(out_dir, max_shard_size="100KB")
    shards = sorted(p.name for p in out_dir.glob("*.safetensors"))
    assert len(shards) > 1 and (out_dir / "model.safetensors.index.json").is_file(), shards
    rc, out = run_inspect(rrc_inspect, out_dir)
    assert rc == 0 and out["check"] == "ok", out
    sd = {k: v for k, v in model.state_dict().items()}
    got = {t["name"]: t for t in out["tensors"]}
    assert set(got) == set(sd)
    for name, t in sd.items():
        assert got[name]["dtype"] == "BF16"
        assert got[name]["sum"] == pytest.approx(t.double().sum().item(), rel=1e-9, abs=1e-9), name


def _rewrite(d: pathlib.Path, edit) -> None:
    tensors = safetensors_torch.load_file(str(d / "model.safetensors"))
    edit(tensors)
    safetensors_torch.save_file(tensors, str(d / "model.safetensors"), metadata={"format": "pt"})


@pytest.mark.parametrize(
    "variant, edit, status, needle",
    [
        ("llama_tied", lambda t: t.__setitem__("lm_head.weight", t["model.embed_tokens.weight"].clone()),
         "RRC_ERR_UNSUPPORTED", "lm_head.weight"),
        ("llama", lambda t: t.pop("lm_head.weight"), "RRC_ERR_NOT_FOUND", "lm_head.weight"),
        ("qwen3", lambda t: t.pop("model.layers.1.self_attn.k_norm.weight"), "RRC_ERR_NOT_FOUND", "k_norm"),
        ("qwen2", lambda t: t.__setitem__("model.layers.0.self_attn.rotary_emb.inv_freq", torch.ones(8)),
         "RRC_ERR_UNSUPPORTED", "rotary_emb.inv_freq"),
        ("qwen2", lambda t: t.__setitem__("model.layers.0.self_attn.o_proj.bias", torch.zeros(64)),
         "RRC_ERR_UNSUPPORTED", "o_proj.bias"),
        ("llama", lambda t: t.__setitem__("model.norm.weight", torch.ones(65)), "RRC_ERR_SHAPE", "model.norm.weight"),
        ("llama", lambda t: t.__setitem__("model.norm.weight", torch.ones(64, dtype=torch.float16)),
         "RRC_ERR_DTYPE", "model.norm.weight"),
    ],
)
def test_checkpoint_tensor_set_mismatch(variant, edit, status, needle, tmp_path, rrc_inspect) -> None:
    d = tmp_path / variant
    mtm.make_variant(variant, str(d))
    _rewrite(d, edit)
    rc, out = run_inspect(rrc_inspect, d)
    assert rc == 0, out
    assert out["check_status"] == status, out["check"]
    assert needle in out["check"]


# ---------------------------------------------------------------------------------------------
# config.json variants (written directly, then loaded by both sides)
# ---------------------------------------------------------------------------------------------

L3 = dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0)
BASE = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=256)

ACCEPTED: dict[str, dict[str, Any]] = {
    "llama_minimal": dict(model_type="llama"),
    "llama_legacy_rope_theta": dict(model_type="llama", rope_theta=500000.0),
    "llama_rope_scaling_null": dict(model_type="llama", rope_scaling=None, rope_theta=250000.0),
    "llama_legacy_llama3_type": dict(model_type="llama", rope_theta=500000.0,
                                     rope_scaling=dict(type="llama3", original_max_position_embeddings=64, **L3)),
    "llama_llama3_no_omp": dict(model_type="llama", rope_scaling=dict(rope_type="llama3", **L3)),
    "llama_llama3_toplevel_omp": dict(model_type="llama", original_max_position_embeddings=32,
                                      rope_scaling=dict(rope_type="llama3", original_max_position_embeddings=128, **L3)),
    "llama_rope_type_beats_type": dict(model_type="llama", rope_scaling=dict(type="llama3", rope_type="default")),
    "llama_rope_parameters_toplevel_theta": dict(model_type="llama", rope_theta=3000.0,
                                                 rope_parameters=dict(rope_type="default")),
    "llama_rope_parameters_theta_wins": dict(model_type="llama", rope_theta=3000.0,
                                             rope_parameters=dict(rope_type="default", rope_theta=7000.0)),
    "llama_rope_scaling_empty": dict(model_type="llama", rope_scaling={}, rope_theta=123.0),
    "llama_head_dim_null": dict(model_type="llama", head_dim=None),
    "llama_head_dim_explicit": dict(model_type="llama", head_dim=8),
    "llama_kv_null": dict(model_type="llama", num_key_value_heads=None),
    "llama_tied": dict(model_type="llama", tie_word_embeddings=True),
    "qwen2_minimal": dict(model_type="qwen2"),
    "qwen2_kv_default_32": dict(model_type="qwen2", num_attention_heads=32, num_key_value_heads="__absent__"),
    "qwen2_head_dim_explicit": dict(model_type="qwen2", head_dim=8),
    "qwen2_sliding_but_no_sliding_layers": dict(model_type="qwen2", use_sliding_window=True, sliding_window=16,
                                                max_window_layers=2),
    "qwen2_attention_bias_key_ignored": dict(model_type="qwen2", attention_bias=False),
    "qwen3_head_dim_default_128": dict(model_type="qwen3"),
    "qwen3_attention_bias": dict(model_type="qwen3", head_dim=16, attention_bias=True),
    "qwen3_llama3_rope": dict(model_type="qwen3", head_dim=16,
                              rope_parameters=dict(rope_type="llama3", rope_theta=1e6,
                                                   original_max_position_embeddings=64, **L3)),
    "qwen3_eps": dict(model_type="qwen3", head_dim=16, rms_norm_eps=1e-6),
}

# Accepted by transformers, refused by rrc (engine does not implement the semantics).
REFUSED: dict[str, tuple[dict[str, Any], str]] = {
    "qwen2_sliding_window_layers": (dict(model_type="qwen2", use_sliding_window=True, sliding_window=16,
                                         max_window_layers=1), "sliding"),
    "qwen3_layer_types_sliding": (dict(model_type="qwen3", head_dim=16,
                                       layer_types=["full_attention", "sliding_attention"]), "layer_types"),
    "llama_rope_linear": (dict(model_type="llama", rope_scaling=dict(rope_type="linear", factor=2.0)), "linear"),
    "llama_rope_dynamic": (dict(model_type="llama", rope_scaling=dict(rope_type="dynamic", factor=2.0)), "dynamic"),
    "llama_rope_yarn": (dict(model_type="llama", rope_scaling=dict(rope_type="yarn", factor=2.0,
                                                                  original_max_position_embeddings=128)), "yarn"),
    "llama_partial_rotary": (dict(model_type="llama", partial_rotary_factor=0.5), "partial_rotary_factor"),
    "llama_both_rope_spellings": (dict(model_type="llama", rope_scaling=dict(rope_type="default", rope_theta=1.0e3),
                                       rope_parameters=dict(rope_type="default", rope_theta=2.0e3)), "both"),
    "llama_attention_bias": (dict(model_type="llama", attention_bias=True), "attention_bias"),
    "llama_mlp_bias": (dict(model_type="llama", mlp_bias=True), "mlp_bias"),
    "llama_gelu": (dict(model_type="llama", hidden_act="gelu"), "hidden_act"),
    "qwen2_attention_dropout": (dict(model_type="qwen2", attention_dropout=0.1), "attention_dropout"),
    "llama_unknown_key": (dict(model_type="llama", my_new_feature=1), "my_new_feature"),
}


def write_config(d: pathlib.Path, overrides: dict[str, Any]) -> pathlib.Path:
    cfg = dict(BASE)
    for k, v in overrides.items():
        if v == "__absent__":
            cfg.pop(k, None)
        else:
            cfg[k] = v
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_config_variant_matches_transformers(name: str, tmp_path: pathlib.Path, rrc_inspect: pathlib.Path) -> None:
    d = write_config(tmp_path / name, ACCEPTED[name])
    rc, out = run_inspect(rrc_inspect, d, "--config-only")
    assert rc == 0 and out["ok"], out
    compare_config(out, hf_reference(d))


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_config_variant_refused(name: str, tmp_path: pathlib.Path, rrc_inspect: pathlib.Path) -> None:
    overrides, needle = REFUSED[name]
    d = write_config(tmp_path / name, overrides)
    # transformers accepts these configs (the refusal is the engine's choice, not a broken file)
    transformers.AutoConfig.from_pretrained(d)
    rc, out = run_inspect(rrc_inspect, d, "--config-only")
    assert rc == 2 and not out["ok"], out
    assert out["status"] == "RRC_ERR_UNSUPPORTED", out
    assert needle in out["error"], out["error"]
    if name == "llama_unknown_key":
        rc, out = run_inspect(rrc_inspect, d, "--config-only", "--allow-unknown-keys")
        assert rc == 0 and out["ok"], out
