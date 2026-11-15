"""Low-precision weight updates: dtype checks, stochastic rounding and reproducible rounding RNG.

Why this exists
---------------
A bf16 number has 8 significant bits (7 stored + the implicit one), so two neighbouring bf16
values around ``w`` are ``|w| * 2**-7 .. 2**-8`` apart. When a trainable parameter is *stored* in
bf16 (model loaded with ``dtype=bfloat16`` and no fp32 master copy), ``torch.optim.AdamW`` writes
``w - lr * update`` back into the bf16 tensor with round-to-nearest-even: every update smaller than
about half a unit in the last place (≈ ``|w| * 2**-8`` relative) is rounded away **silently**, step
after step, so small learning rates or small corrections (e.g. :class:`P2PriorRule`'s
``Σ Δα τ``) never reach the weights. HF ``TrainingArguments(bf16=True)`` is *not* affected: it
keeps fp32 parameters and only autocasts the forward pass.

Two standard remedies, both available through
:class:`~research_rl.optim.optimizer_wrapper.HookedOptimizer`:

* **fp32 master weights** (``master_weights=True``): the optimizer updates an fp32 copy and the
  served bf16 weight is re-derived from it after every step; updates accumulate in the master.
* **stochastic rounding** (``write_rounding="stochastic"``): the served value is rounded up or
  down at random with probabilities that make it an *unbiased* estimate of the fp32 value, so
  sub-ulp updates still move the served weights in expectation.

Functions
---------
* :func:`is_low_precision` / :func:`low_precision_trainable` – detect bf16/fp16 trainable tensors.
* :func:`stochastic_round` – exact, unbiased fp32 → bf16 stochastic rounding (bit trick on the
  int32 view: add a uniform integer in ``[0, 2**16)`` and clear the low 16 bits). Values exactly
  representable in bf16 are returned unchanged; NaN/±inf pass through; finite values beyond the
  bf16 range round to ±inf with the probability the bit pattern dictates (round-to-nearest also
  overflows there). An fp32/fp64 target is a plain cast; an fp16 target raises
  ``NotImplementedError`` (fp16 has a different exponent width, the bit trick does not apply).
* :func:`round_to` – ``"nearest"`` (``x.to(dtype)``) or ``"stochastic"`` dispatch.
* :func:`rounding_generator` / :func:`rounding_seed_for` – a ``torch.Generator`` on the tensor's
  device seeded from ``(seed, step, parameter index)``, so rounding is reproducible across runs
  and checkpoint resumes without storing RNG state. Streams of different device types (CPU vs.
  CUDA Philox) differ, so the same seed gives different (equally valid) draws on another device.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

LOW_PRECISION_DTYPES = (torch.bfloat16, torch.float16)
ROUNDING_MODES = ("nearest", "stochastic")

_MASK64 = (1 << 64) - 1


def is_low_precision(dtype: torch.dtype) -> bool:
    """True for bf16 and fp16 (the dtypes whose round-to-nearest writes drop small updates)."""
    return dtype in LOW_PRECISION_DTYPES


def low_precision_trainable(named_params: Iterable[tuple[str, torch.Tensor]]) -> list[str]:
    """Names of parameters with ``requires_grad=True`` stored in bf16/fp16 (in iteration order)."""
    return [n for n, p in named_params if p.requires_grad and is_low_precision(p.dtype)]


def stochastic_round(
    x: torch.Tensor, dtype: torch.dtype, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Round ``x`` to ``dtype`` stochastically (unbiased); returns a new tensor of ``dtype``.

    For ``dtype=torch.bfloat16`` the input must be fp32 (or already bf16, returned as a copy).
    Let ``b`` be the fp32 bit pattern of a finite ``x`` and ``lo = b & 0xFFFF``. The result is the
    bf16 value with bits ``b >> 16`` (truncation toward zero) with probability ``1 - lo/2**16`` and
    the next one away from zero with probability ``lo/2**16``. Because fp32 values sharing the top
    16 bits are uniformly spaced between the two bf16 neighbours (also across a binade boundary and
    for subnormals), ``E[result] == x`` exactly. ``generator`` must live on ``x.device``.
    """
    if dtype in (torch.float32, torch.float64):
        return x.to(dtype)
    if dtype == torch.float16:
        raise NotImplementedError(
            "stochastic rounding to float16 is not implemented (the int32 bit trick is only exact for "
            "bfloat16, which shares fp32's exponent); use write_rounding='nearest' or bf16/fp32 parameters"
        )
    if dtype != torch.bfloat16:
        raise ValueError(f"stochastic_round: unsupported target dtype {dtype}")
    if x.dtype == torch.bfloat16:
        return x.clone()
    if x.dtype != torch.float32:
        raise ValueError(
            f"stochastic_round to bfloat16 needs a float32 input (got {x.dtype}); a double rounding via "
            "float32 would not be exact"
        )
    src = x.contiguous()
    bits = src.view(torch.int32)
    finite = torch.isfinite(src)
    noise = torch.randint(0, 1 << 16, src.shape, dtype=torch.int32, device=src.device, generator=generator)
    # Zero the NaN/inf lanes before the addition so the int32 sum cannot wrap; they are put back below.
    rounded = (torch.where(finite, bits, torch.zeros_like(bits)) + noise) & -65536  # clear the low 16 bits
    out = torch.where(finite, rounded.view(torch.float32), src)
    return out.to(torch.bfloat16)  # exact: the low 16 bits are zero (NaN/inf convert to NaN/inf)


def round_to(
    x: torch.Tensor, dtype: torch.dtype, mode: str = "nearest", generator: torch.Generator | None = None
) -> torch.Tensor:
    """``x.to(dtype)`` (``mode="nearest"``, round-to-nearest-even) or :func:`stochastic_round`."""
    if mode == "nearest":
        return x.to(dtype)
    if mode == "stochastic":
        return stochastic_round(x, dtype, generator)
    raise ValueError(f"rounding mode must be one of {ROUNDING_MODES}, got {mode!r}")


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def rounding_seed_for(seed: int, step: int, index: int) -> int:
    """A well-mixed 63-bit seed for (seed, optimizer step, parameter index); pure function."""
    z = _splitmix64(int(seed) & _MASK64)
    z = _splitmix64(z ^ (int(step) & _MASK64))
    z = _splitmix64(z ^ (int(index) & _MASK64))
    return z & ((1 << 63) - 1)


def rounding_generator(
    seed: int,
    step: int,
    index: int,
    device: torch.device | str = "cpu",
    *,
    generator: torch.Generator | None = None,
) -> torch.Generator:
    """Generator on ``device`` seeded with :func:`rounding_seed_for` ``(seed, step, index)``.

    Pass an existing ``generator`` on the same device to re-seed it instead of allocating one.
    """
    dev = torch.device(device)
    gen = generator if generator is not None else torch.Generator(device=dev)
    gen.manual_seed(rounding_seed_for(seed, step, index))
    return gen
