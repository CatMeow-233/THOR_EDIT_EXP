"""Model introspection shared by all parameter-level tools.

Supports Hugging Face decoder-only models whose FFN is a SwiGLU block with ``gate_proj``,
``up_proj`` and ``down_proj`` linear layers (Llama, Qwen2/2.5/3 dense, Mistral, ...):

    FFN(x) = down_proj( act(gate_proj(x)) * up_proj(x) )

``gate_proj.weight``/``up_proj.weight`` have shape ``[intermediate, hidden]`` and
``down_proj.weight`` has shape ``[hidden, intermediate]``, so intermediate channel ``c`` owns
row ``c`` of gate/up and column ``c`` of down.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

SWIGLU_NAMES = ("gate_proj", "up_proj", "down_proj")


class ModelInterfaceError(ValueError):
    pass


@dataclass(frozen=True)
class SwiGLURef:
    """Location of one SwiGLU FFN inside a model."""

    layer_index: int
    module_path: str  # e.g. "model.layers.3.mlp"
    module: nn.Module

    @property
    def gate(self) -> nn.Linear:
        return self.module.gate_proj  # type: ignore[return-value]

    @property
    def up(self) -> nn.Linear:
        return self.module.up_proj  # type: ignore[return-value]

    @property
    def down(self) -> nn.Linear:
        return self.module.down_proj  # type: ignore[return-value]

    @property
    def intermediate_size(self) -> int:
        return int(self.gate.weight.shape[0])

    def param_names(self) -> dict[str, str]:
        """Fully-qualified weight names, keyed by projection name."""
        return {k: f"{self.module_path}.{k}.weight" for k in SWIGLU_NAMES}


def is_swiglu(module: nn.Module) -> bool:
    return all(isinstance(getattr(module, n, None), nn.Linear) for n in SWIGLU_NAMES)


def get_decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Return ``(path, layers)`` for the decoder block list of an HF causal LM."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "layers"):
        obj: object = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.ModuleList):
            return path, obj
    raise ModelInterfaceError(f"cannot locate decoder layers in {type(model).__name__}")


def find_swiglu_ffns(
    model: nn.Module, layers: Iterable[int] | None = None, *, expert: str | None = None
) -> list[SwiGLURef]:
    """Locate the SwiGLU FFN (``layer.mlp``) of each decoder layer.

    If ``layer.mlp`` is an expert container (duck-typed: an ``experts`` ``nn.ModuleDict`` and an
    ``active`` name, e.g. :class:`research_rl.models.expert_manager.ExpertFFN`), the SwiGLU of
    ``expert`` (default: the container's active expert) is returned with its real parameter path
    ``<layers>.<i>.mlp.experts.<name>``, so channel-level tools can operate inside one expert.
    """
    path, block_list = get_decoder_layers(model)
    wanted = None if layers is None else set(layers)
    refs: list[SwiGLURef] = []
    for i, layer in enumerate(block_list):
        if wanted is not None and i not in wanted:
            continue
        mlp = getattr(layer, "mlp", None)
        module_path = f"{path}.{i}.mlp"
        experts = getattr(mlp, "experts", None)
        if isinstance(experts, nn.ModuleDict):
            name = expert if expert is not None else getattr(mlp, "active", None)
            if not isinstance(name, str) or name not in experts:
                raise ModelInterfaceError(f"layer {i}: unknown expert {name!r}; have {list(experts)}")
            mlp = experts[name]
            module_path = f"{module_path}.experts.{name}"
        elif expert is not None:
            raise ModelInterfaceError(f"layer {i} has no expert container but expert={expert!r} was given")
        if mlp is None or not is_swiglu(mlp):
            raise ModelInterfaceError(f"layer {i} has no SwiGLU `mlp` (found {type(mlp).__name__})")
        refs.append(SwiGLURef(layer_index=i, module_path=module_path, module=mlp))
    if wanted is not None and len(refs) != len(wanted):
        missing = sorted(wanted - {r.layer_index for r in refs})
        raise ModelInterfaceError(f"layer indices out of range: {missing}")
    return refs


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def snapshot(model: nn.Module, names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
    """Detached CPU copy of parameters (and only parameters) for exact comparisons."""
    wanted = set(names) if names is not None else None
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if wanted is None or n in wanted}


@torch.no_grad()
def fingerprint(tensors: dict[str, torch.Tensor] | nn.Module, names: Iterable[str] | None = None) -> str:
    """SHA-256 over names, shapes, dtypes and raw bytes. Bit-exact identity check.

    Streams one tensor at a time (only one CPU copy alive), so it is usable on large models.
    """
    keep = set(names) if names is not None else None
    if isinstance(tensors, nn.Module):
        source: dict[str, torch.Tensor] = {
            n: p for n, p in tensors.named_parameters() if keep is None or n in keep
        }
    else:
        source = {k: v for k, v in tensors.items() if keep is None or k in keep}
    h = hashlib.sha256()
    for name in sorted(source):
        t = source[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()
