"""Write tiny random fp32 Llama / Qwen2 / Qwen3 checkpoints for the rrc tests.

Each checkpoint is a directory with config.json + model.safetensors written by
``save_pretrained`` (a few hundred KB). Weights are random (seeded); RMSNorm weights and biases are
perturbed away from their default init so that the tests exercise them.

    python rrc/tools/make_tiny_model.py --family llama --out /tmp/tiny_llama [--tied] [--llama3-rope]
    python rrc/tools/make_tiny_model.py --variant qwen3_hd32 --out /tmp/tiny

Nothing is downloaded: the models are built from a config with ``from_config``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Shared tiny dimensions: hidden 64, 2 layers, 4 heads, 2 kv heads (GQA), vocab 128.
TINY: dict[str, Any] = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=256,
    rms_norm_eps=1e-5,
)

# Llama-3.1-style rope scaling, sized so that head_dim 16 has frequencies in all three bands
# (unchanged, smoothed, divided by factor).
LLAMA3_ROPE: dict[str, Any] = dict(
    rope_type="llama3",
    rope_theta=500000.0,
    factor=8.0,
    low_freq_factor=1.0,
    high_freq_factor=4.0,
    original_max_position_embeddings=64,
)

# name -> (family, extra config kwargs)
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "llama": ("llama", {}),
    "llama_tied": ("llama", {"tie_word_embeddings": True}),
    "llama3_rope": ("llama", {"rope_parameters": LLAMA3_ROPE}),
    "qwen2": ("qwen2", {"rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0}}),
    "qwen2_tied": ("qwen2", {"tie_word_embeddings": True}),
    "qwen3": ("qwen3", {"head_dim": 16}),
    "qwen3_tied": ("qwen3", {"head_dim": 16, "tie_word_embeddings": True}),
    # head_dim != hidden_size / num_attention_heads and attention biases on
    "qwen3_hd32_bias": ("qwen3", {"head_dim": 32, "attention_bias": True}),
}


def build_config(family: str, **overrides: Any):
    from transformers import LlamaConfig, Qwen2Config, Qwen3Config

    cls = {"llama": LlamaConfig, "qwen2": Qwen2Config, "qwen3": Qwen3Config}[family]
    kw = dict(TINY)
    kw.update(overrides)
    return cls(**kw)


def make_tiny(family: str, out_dir: str, seed: int = 0, **overrides: Any) -> str:
    """Create a random fp32 model of `family` and save it to `out_dir`. Returns out_dir."""
    import torch
    from transformers import AutoModelForCausalLM

    cfg = build_config(family, **overrides)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed + 1)
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith(".bias"):
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    model.eval()
    model.save_pretrained(out_dir)
    return out_dir


def make_variant(name: str, out_dir: str, seed: int = 0) -> str:
    family, extra = VARIANTS[name]
    return make_tiny(family, out_dir, seed=seed, **extra)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=["llama", "qwen2", "qwen3"])
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--tied", action="store_true")
    ap.add_argument("--llama3-rope", action="store_true")
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.variant:
        make_variant(args.variant, args.out, seed=args.seed)
    else:
        if not args.family:
            ap.error("--family or --variant is required")
        extra: dict[str, Any] = {}
        if args.tied:
            extra["tie_word_embeddings"] = True
        if args.llama3_rope:
            extra["rope_parameters"] = LLAMA3_ROPE
        if args.head_dim:
            extra["head_dim"] = args.head_dim
        elif args.family == "qwen3":
            extra["head_dim"] = 16
        make_tiny(args.family, args.out, seed=args.seed, **extra)
    print(args.out)


if __name__ == "__main__":
    main()
