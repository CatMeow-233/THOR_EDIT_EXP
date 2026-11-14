"""Expert Manager: a shared backbone with independently trained, response-level FFN experts.

Research setting (TLC): one frozen *shared backbone* plus several *FFN experts* that start as
exact copies of the base model's SwiGLU FFN (``layer.mlp``) in chosen decoder layers. Each
expert is trained separately (only its parameters require grad), saved / loaded on its own, and
at inference one expert is chosen **per response**: every token of a sequence, prompt and
generated continuation alike, goes through the same expert.

**This is not token-level MoE.** A token-level MoE layer (Mixtral, Qwen-MoE, ...) has a learned
router that sends *each token* to its top-k experts and mixes their outputs with router weights,
and the experts are trained jointly. Here there is no learned router and no mixing: the expert
of a row is an explicit, external decision (``activate``/``using`` for the whole batch, or a list
of names with one entry per batch row), and experts never see each other's gradients. Fusing the
trained experts into a token-level MoE is a later research step and is **not** implemented here.

What is implemented (and tested in ``tests/research_rl/test_expert_manager.py``):

* :class:`ExpertFFN` replaces ``layer.mlp`` and holds an ``nn.ModuleDict`` of SwiGLU copies; the
  original module is kept as expert ``"base"`` (same parameter objects, so it is bit-exact).
* :class:`ExpertManager` attaches/detaches, creates experts, freezes everything except the one
  trainable expert, fingerprints the backbone, routes per sample, saves/loads a single expert as
  safetensors with compatibility metadata, exports a *vanilla* HF checkpoint for one expert, and
  is a :class:`~research_rl.core.checkpoint.StatefulComponent` so HF/TRL checkpoint resume can
  recreate the expert structure before the Trainer loads the weights.
* :func:`load_model_with_experts` re-attaches experts when loading a checkpoint saved *with*
  experts attached.

Pitfall: attaching renames state-dict keys (``model.layers.i.mlp.gate_proj.weight`` becomes
``model.layers.i.mlp.experts.<name>.gate_proj.weight``). ``LlamaForCausalLM.from_pretrained`` on
such a checkpoint only logs a warning and **re-initialises the FFN randomly**. Use
:meth:`ExpertManager.export_merged` (vanilla keys) or :func:`load_model_with_experts`.

Backbone integrity: the first :meth:`ExpertManager.freeze_backbone` (or ``set_trainable``) records
the backbone fingerprint once; later calls *verify* the backbone against it (raising
:class:`BackboneChangedError`) instead of silently re-recording it, so the check covers a whole
multi-expert session. Use ``freeze_backbone(rebaseline=True)`` for an intentional reset.

Channel / neuron tools: the core ``find_swiglu_ffns`` resolves an attached layer to its **active**
expert by default (or the expert named by ``expert=``), with the real parameter path
``...mlp.experts.<name>.*``. Calling such a tool (e.g. ``channel_mask``) *before*
``set_trainable``/``activate`` therefore targets ``"base"``; pass ``expert=`` to be explicit.

Limitations: the routing assumes the FFN input has the sample on dim 0 (true for dense HF decoder
models; beam search / ``num_return_sequences`` change the batch and are rejected).
Parameters added by ``create_expert`` / ``set_trainable`` after an optimizer exists are unknown to
that optimizer; configure experts before building the trainer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from research_rl.models.model_interface import (
    ModelInterfaceError,
    find_swiglu_ffns,
    fingerprint,
    get_decoder_layers,
    is_swiglu,
)

BASE_EXPERT = "base"
EXPERT_FILE_FORMAT = "research_rl.expert/1"
MANAGER_STATE_FORMAT = "research_rl.expert_manager/1"
SIDECAR_FILE = "research_rl_experts.json"

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
# Config entries that do not change parameter shapes / the forward function of trained weights, or
# drift across save/load round trips: special-token ids (commonly set after loading, e.g. a pad id),
# init / dropout hyper-parameters, output flags and task metadata. Every other public entry is hashed
# (an exclude-list, so unknown architecture fields of other model families still fail loudly).
_CONFIG_HASH_EXCLUDE = frozenset(
    {
        "transformers_version",
        "architectures",
        "dtype",
        "torch_dtype",
        "use_cache",
        "output_hidden_states",
        "output_attentions",
        "return_dict",
        "id2label",
        "label2id",
        "problem_type",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "sep_token_id",
        "decoder_start_token_id",
        "initializer_range",
        "chunk_size_feed_forward",
        "tokenizer_class",
        "finetuning_task",
        "task_specific_params",
        "prefix",
    }
)


def _hashed_config_key(key: str) -> bool:
    if key.startswith("_") or key in _CONFIG_HASH_EXCLUDE:
        return False
    return not ("dropout" in key or key.endswith("_dropout_prob"))


class ExpertError(ValueError):
    """Unknown expert, invalid name or invalid manager operation."""


class ExpertCompatibilityError(ExpertError):
    """An expert file / checkpoint does not match this model."""


class BackboneChangedError(RuntimeError):
    """The shared backbone differs from its recorded fingerprint."""


def config_hash(config: Any) -> str:
    """SHA-256 of the architecture-relevant part of an HF config (stable across save/load).

    Special-token ids, ``initializer_range`` and dropout rates are not hashed (see
    ``_CONFIG_HASH_EXCLUDE``): they do not change which weights fit the model.
    """
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    kept = {k: v for k, v in raw.items() if _hashed_config_key(k)}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=str).encode()).hexdigest()


def model_class_name(model: nn.Module) -> str:
    cls = type(model)
    return f"{cls.__module__}.{cls.__qualname__}"


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ExpertError(f"invalid expert name {name!r}: use letters, digits, '_' or '-'")


class ExpertFFN(nn.Module):
    """Drop-in replacement for ``layer.mlp`` holding several SwiGLU experts.

    ``routing`` (one expert name per batch row) takes precedence over ``active``. Each routed row
    is computed by its own expert; rows sharing an expert are batched together.
    """

    def __init__(self, base: nn.Module, layer_index: int) -> None:
        super().__init__()
        if not is_swiglu(base):
            raise ModelInterfaceError(
                f"layer {layer_index}: expected a SwiGLU FFN, got {type(base).__name__}"
            )
        self.layer_index = layer_index
        self.experts = nn.ModuleDict({BASE_EXPERT: base})
        self.active: str = BASE_EXPERT
        self.routing: tuple[str, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.routing is None:
            return self.experts[self.active](x)
        if x.shape[0] != len(self.routing):
            raise ExpertError(
                f"routing has {len(self.routing)} entries but the FFN input has batch size {x.shape[0]} "
                "(beam search / num_return_sequences expand the batch; route the expanded batch instead)"
            )
        groups: dict[str, list[int]] = {}
        for row, name in enumerate(self.routing):
            groups.setdefault(name, []).append(row)
        if len(groups) == 1:
            return self.experts[next(iter(groups))](x)
        outputs: list[torch.Tensor] = []
        order: list[int] = []
        for name, rows in groups.items():
            idx = torch.tensor(rows, device=x.device, dtype=torch.long)
            outputs.append(self.experts[name](x.index_select(0, idx)))
            order.extend(rows)
        inverse = torch.argsort(torch.tensor(order, device=x.device, dtype=torch.long))
        return torch.cat(outputs, dim=0).index_select(0, inverse)

    def extra_repr(self) -> str:
        return f"layer={self.layer_index}, experts={list(self.experts)}, active={self.active!r}"


class ExpertManager:
    """Manage response-level FFN experts on an HF decoder-only model with SwiGLU FFNs."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._layers_path, self._blocks = get_decoder_layers(model)
        self._ffns: dict[int, ExpertFFN] = {}
        self._expert_names: list[str] = []
        self._trainable: str | None = None
        self._frozen = False
        self._backbone_ref: str | None = None

    # ------------------------------------------------------------------ structure
    @property
    def attached(self) -> bool:
        return bool(self._ffns)

    @property
    def layers(self) -> list[int]:
        return sorted(self._ffns)

    @property
    def expert_names(self) -> list[str]:
        return list(self._expert_names)

    @property
    def trainable_expert(self) -> str | None:
        return self._trainable

    def _require_attached(self) -> None:
        if not self._ffns:
            raise ExpertError("no experts attached: call attach(layers) first")

    def _require_expert(self, name: str) -> None:
        self._require_attached()
        if name not in self._expert_names:
            raise ExpertError(f"unknown expert {name!r}; known: {self._expert_names}")

    def attach(self, layers: Iterable[int] | None = None) -> list[int]:
        """Replace ``layer.mlp`` of the given decoder layers (default: all) by an :class:`ExpertFFN`.

        The original FFN module becomes expert ``"base"`` (no copy, same parameters).
        """
        if self._ffns:
            raise ExpertError(f"experts already attached to layers {self.layers}")
        wanted = sorted(set(range(len(self._blocks)) if layers is None else layers))
        if not wanted:
            raise ExpertError("attach() needs at least one layer")
        for i in wanted:
            if 0 <= i < len(self._blocks) and isinstance(getattr(self._blocks[i], "mlp", None), ExpertFFN):
                raise ExpertError(f"layer {i} already holds an ExpertFFN (another manager?)")
        refs = find_swiglu_ffns(self.model, wanted)  # validates range and SwiGLU structure
        for ref in refs:
            ffn = ExpertFFN(ref.module, ref.layer_index)
            self._blocks[ref.layer_index].mlp = ffn
            self._ffns[ref.layer_index] = ffn
        self._expert_names = [BASE_EXPERT]
        return wanted

    def create_expert(self, name: str, source: str = BASE_EXPERT) -> None:
        """Add expert ``name`` as an exact copy of expert ``source`` in every attached layer."""
        _check_name(name)
        self._require_expert(source)
        if name in self._expert_names:
            raise ExpertError(f"expert {name!r} already exists")
        for ffn in self._ffns.values():
            ffn.experts[name] = copy.deepcopy(ffn.experts[source])
        self._expert_names.append(name)
        if self._frozen:
            self._apply_trainability()

    def delete_expert(self, name: str) -> None:
        self._require_expert(name)
        if name == BASE_EXPERT:
            raise ExpertError("the base expert cannot be deleted")
        if self._trainable == name:
            raise ExpertError(f"expert {name!r} is trainable; set_trainable(None) first")
        for ffn in self._ffns.values():
            if ffn.active == name or (ffn.routing is not None and name in ffn.routing):
                raise ExpertError(f"expert {name!r} is active or routed")
        for ffn in self._ffns.values():
            del ffn.experts[name]
        self._expert_names.remove(name)

    # ------------------------------------------------------------------ selection
    @property
    def active(self) -> str:
        self._require_attached()
        return next(iter(self._ffns.values())).active

    @property
    def routing(self) -> tuple[str, ...] | None:
        self._require_attached()
        return next(iter(self._ffns.values())).routing

    def activate(self, name: str) -> None:
        """Use expert ``name`` for every row (clears per-sample routing)."""
        self._require_expert(name)
        for ffn in self._ffns.values():
            ffn.active = name
            ffn.routing = None

    def set_routing(self, names: Sequence[str] | None) -> None:
        """Per-sample (response-level) routing: ``names[b]`` is the expert of batch row ``b``."""
        self._require_attached()
        if names is None:
            routing = None
        else:
            if isinstance(names, str) or len(names) == 0:
                raise ExpertError("routing must be a non-empty sequence of expert names")
            for n in names:
                self._require_expert(n)
            routing = tuple(names)
        for ffn in self._ffns.values():
            ffn.routing = routing

    def clear_routing(self) -> None:
        self.set_routing(None)

    def _selection(self) -> tuple[str, tuple[str, ...] | None]:
        return self.active, self.routing

    def _restore_selection(self, state: tuple[str, tuple[str, ...] | None]) -> None:
        for ffn in self._ffns.values():
            ffn.active, ffn.routing = state

    @contextmanager
    def using(self, name: str) -> Iterator[None]:
        """Temporarily use expert ``name`` for all rows."""
        self._require_expert(name)
        saved = self._selection()
        try:
            self.activate(name)
            yield
        finally:
            self._restore_selection(saved)

    @contextmanager
    def routed(self, names: Sequence[str]) -> Iterator[None]:
        """Temporarily route row ``b`` through expert ``names[b]``."""
        saved = self._selection()
        try:
            self.set_routing(names)
            yield
        finally:
            self._restore_selection(saved)

    # ------------------------------------------------------------------ parameters / freezing
    def _expert_prefixes(self, name: str | None = None) -> list[str]:
        out = []
        for i in self.layers:
            base = f"{self._layers_path}.{i}.mlp.experts."
            out.append(base if name is None else f"{base}{name}.")
        return out

    def expert_parameter_names(self, name: str) -> list[str]:
        self._require_expert(name)
        prefixes = tuple(self._expert_prefixes(name))
        return [n for n, _ in self.model.named_parameters() if n.startswith(prefixes)]

    def expert_parameters(self, name: str) -> list[nn.Parameter]:
        wanted = set(self.expert_parameter_names(name))
        return [p for n, p in self.model.named_parameters() if n in wanted]

    def backbone_parameter_names(self) -> list[str]:
        """Every parameter that does not belong to any expert (the shared backbone)."""
        prefixes = tuple(self._expert_prefixes())
        return [n for n, _ in self.model.named_parameters() if not prefixes or not n.startswith(prefixes)]

    def trainable_parameter_names(self) -> list[str]:
        return [n for n, p in self.model.named_parameters() if p.requires_grad]

    def _apply_trainability(self) -> None:
        keep = set(self.expert_parameter_names(self._trainable)) if self._trainable else set()
        for n, p in self.model.named_parameters():
            p.requires_grad_(n in keep)

    def _check_or_record_backbone(self, rebaseline: bool) -> str:
        """Record the backbone reference if there is none (or ``rebaseline``); otherwise verify it."""
        if rebaseline or self._backbone_ref is None:
            self._backbone_ref = self.backbone_fingerprint()
        else:
            self.assert_backbone_unchanged()
        return self._backbone_ref

    def freeze_backbone(self, *, rebaseline: bool = False) -> str:
        """Freeze everything except the trainable expert (if any) and return the backbone reference
        fingerprint used by :meth:`assert_backbone_unchanged`.

        The reference is recorded only the first time (or when restored by :meth:`load_state_dict`);
        later calls first verify the current backbone against it and raise
        :class:`BackboneChangedError` on a difference, so switching experts cannot hide a corrupted
        backbone. ``rebaseline=True`` deliberately re-records it (e.g. after an intended change).
        """
        self._require_attached()
        ref = self._check_or_record_backbone(rebaseline)
        self._frozen = True
        self._apply_trainability()
        return ref

    def set_trainable(self, name: str | None, *, rebaseline: bool = False) -> None:
        """Make expert ``name`` the only trainable parameters and activate it (``None``: freeze all).

        Verifies the backbone against the recorded reference first (see :meth:`freeze_backbone`);
        on :class:`BackboneChangedError` the manager is left unchanged. The base expert is the
        reference copy and cannot be trained; create a new expert instead.
        """
        self._require_attached()
        if name is not None:
            self._require_expert(name)
            if name == BASE_EXPERT:
                raise ExpertError("the base expert is the frozen reference; train a copy (create_expert)")
        self._check_or_record_backbone(rebaseline)
        self._trainable = name
        self._frozen = True
        self._apply_trainability()
        if name is not None:
            self.activate(name)

    # ------------------------------------------------------------------ integrity
    def backbone_fingerprint(self) -> str:
        return fingerprint(self.model, self.backbone_parameter_names())

    def expert_fingerprint(self, name: str) -> str:
        return fingerprint(self.model, self.expert_parameter_names(name))

    def assert_backbone_unchanged(self, reference: str | None = None) -> None:
        ref = reference if reference is not None else self._backbone_ref
        if ref is None:
            raise ExpertError("no backbone reference: call freeze_backbone() or pass reference=")
        current = self.backbone_fingerprint()
        if current != ref:
            raise BackboneChangedError(f"shared backbone changed: fingerprint {current[:12]} != {ref[:12]}")

    # ------------------------------------------------------------------ expert files
    def _expert_tensors(self, name: str) -> dict[str, torch.Tensor]:
        self._require_expert(name)
        out: dict[str, torch.Tensor] = {}
        for i, ffn in sorted(self._ffns.items()):
            for k, v in ffn.experts[name].state_dict().items():
                out[f"{i}.{k}"] = v.detach().cpu().contiguous().clone()
        return out

    def _compat_metadata(self) -> dict[str, str]:
        base = self._expert_tensors(BASE_EXPERT)
        return {
            "layers": json.dumps(self.layers),
            "model_class": model_class_name(self.model),
            "config_hash": config_hash(getattr(self.model, "config", {})),
            "backbone_fingerprint": self.backbone_fingerprint(),
            "base_expert_fingerprint": fingerprint(base),
            "shapes": json.dumps({k: list(v.shape) for k, v in base.items()}, sort_keys=True),
            "dtypes": json.dumps({k: str(v.dtype) for k, v in base.items()}, sort_keys=True),
        }

    def save_expert(self, name: str, path: str | Path) -> Path:
        """Write one expert (all attached layers) as safetensors with compatibility metadata."""
        from safetensors.torch import save_file

        tensors = self._expert_tensors(name)
        meta = {
            "format": EXPERT_FILE_FORMAT,
            "expert_name": name,
            "content_sha256": fingerprint(tensors),
            **self._compat_metadata(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata=meta)
        return path

    @staticmethod
    def read_expert_file(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
        from safetensors import safe_open

        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                meta = dict(f.metadata() or {})
                tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 - safe_open has no __iter__
        except Exception as exc:  # safetensors raises its own error types for malformed files
            raise ExpertCompatibilityError(f"cannot read expert file {path}: {exc}") from exc
        return tensors, meta

    def load_expert(
        self,
        path: str | Path,
        name: str | None = None,
        *,
        overwrite: bool = False,
        check_backbone: bool = True,
    ) -> str:
        """Load an expert file into this model; validates everything before changing the model.

        Raises :class:`ExpertCompatibilityError` on a corrupted file or any mismatch in format,
        model class, config, layers, tensor names/shapes/dtypes, base expert or (unless
        ``check_backbone=False``) backbone fingerprint. Returns the expert name used.
        """
        self._require_attached()
        tensors, meta = self.read_expert_file(path)
        if meta.get("format") != EXPERT_FILE_FORMAT:
            raise ExpertCompatibilityError(f"{path}: not an expert file (format={meta.get('format')!r})")
        name = name if name is not None else meta.get("expert_name", "")
        _check_name(name)
        if name == BASE_EXPERT:
            raise ExpertCompatibilityError("refusing to overwrite the base expert")
        if name in self._expert_names and not overwrite:
            raise ExpertError(f"expert {name!r} exists; pass overwrite=True to replace its weights")
        if fingerprint(tensors) != meta.get("content_sha256"):
            raise ExpertCompatibilityError(f"{path}: tensor content does not match its checksum (corrupted)")
        expected = self._compat_metadata()
        checks = ["layers", "model_class", "config_hash", "shapes", "dtypes", "base_expert_fingerprint"]
        if check_backbone:
            checks.append("backbone_fingerprint")
        for key in checks:
            if meta.get(key) != expected[key]:
                raise ExpertCompatibilityError(
                    f"{path}: {key} mismatch (file {str(meta.get(key))[:80]!r} vs model {expected[key][:80]!r})"
                )
        base = self._expert_tensors(BASE_EXPERT)
        if set(tensors) != set(base):
            raise ExpertCompatibilityError(f"{path}: tensor names differ from the attached FFNs")
        for k, v in tensors.items():
            if v.shape != base[k].shape or v.dtype != base[k].dtype:
                raise ExpertCompatibilityError(f"{path}: tensor {k} has shape/dtype {v.shape}/{v.dtype}")
        # validated: now mutate
        if name not in self._expert_names:
            self.create_expert(name)
        with torch.no_grad():
            for i, ffn in self._ffns.items():
                for k, t in ffn.experts[name].state_dict().items():
                    t.copy_(tensors[f"{i}.{k}"].to(device=t.device))
        return name

    # ------------------------------------------------------------------ export / detach
    def _vanilla_key(self, key: str, name: str) -> str | None:
        """Map an attached state-dict key to its vanilla name (``None``: drop it)."""
        for i in self.layers:
            prefix = f"{self._layers_path}.{i}.mlp.experts."
            if key.startswith(prefix):
                rest = key[len(prefix) :]
                expert, _, tail = rest.partition(".")
                return f"{self._layers_path}.{i}.mlp.{tail}" if expert == name else None
        return key

    def merged_state_dict(self, name: str) -> dict[str, torch.Tensor]:
        """State dict with vanilla keys in which expert ``name`` is the FFN (no copies made)."""
        self._require_expert(name)
        out: dict[str, torch.Tensor] = {}
        for k, v in self.model.state_dict().items():
            vk = self._vanilla_key(k, name)
            if vk is not None:
                out[vk] = v
        return out

    def merged_model(self, name: str) -> nn.Module:
        """A deep copy of the model with expert ``name`` as a plain ``layer.mlp`` (all params trainable)."""
        self._require_expert(name)
        clone = copy.deepcopy(self.model)
        _, blocks = get_decoder_layers(clone)
        for i in self.layers:
            ffn = blocks[i].mlp
            assert isinstance(ffn, ExpertFFN)
            blocks[i].mlp = ffn.experts[name]
        for p in clone.parameters():
            p.requires_grad_(True)
        return clone

    def export_merged(self, name: str, path: str | Path, **save_kwargs: Any) -> Path:
        """Save a *vanilla* HF checkpoint (standard ``model.layers.i.mlp.*`` keys) with expert ``name``.

        Loadable with ``AutoModelForCausalLM.from_pretrained`` without ResearchRL. No model copy is
        made: the attached model is saved with :meth:`merged_state_dict`.
        """
        path = Path(path)
        state = self.merged_state_dict(name)
        self.model.save_pretrained(str(path), state_dict=state, **save_kwargs)  # type: ignore[operator]
        return path

    def detach(self, name: str = BASE_EXPERT) -> None:
        """Put expert ``name`` back as the vanilla ``layer.mlp`` (same parameter objects) and drop the
        other experts. Parameter ``requires_grad`` flags are left as they are."""
        self._require_expert(name)
        for i, ffn in self._ffns.items():
            self._blocks[i].mlp = ffn.experts[name]
        self._ffns.clear()
        self._expert_names = []
        self._trainable = None
        self._frozen = False
        self._backbone_ref = None

    # ------------------------------------------------------------------ checkpoint integration
    def state_dict(self) -> dict[str, Any]:
        """Structure only (no tensors: expert weights live in the model state dict).

        A detached manager (never attached, or after :meth:`detach`) returns a state with
        ``layers == []`` so it can stay registered as a checkpoint component.
        """
        if not self._ffns:
            return {
                "format": MANAGER_STATE_FORMAT,
                "layers": [],
                "experts": [],
                "active": None,
                "routing": None,
                "trainable": None,
                "frozen": False,
                "backbone_fingerprint": None,
            }
        routing = self.routing
        return {
            "format": MANAGER_STATE_FORMAT,
            "layers": self.layers,
            "experts": self.expert_names,
            "active": self.active,
            "routing": None if routing is None else list(routing),
            "trainable": self._trainable,
            "frozen": self._frozen,
            "backbone_fingerprint": self._backbone_ref,
        }

    def _validate_state(self, state: Mapping[str, Any]) -> tuple[list[int], list[str]]:
        if state.get("format") != MANAGER_STATE_FORMAT:
            raise ExpertCompatibilityError(f"unknown expert manager state format {state.get('format')!r}")
        layers = [int(i) for i in state["layers"]]
        names = [str(n) for n in state["experts"]]
        if not layers:
            if self._ffns:
                raise ExpertCompatibilityError(
                    "state is of a detached manager but experts are attached here (which expert would "
                    "be the vanilla FFN is unknown); detach() before loading it"
                )
            return layers, names
        if self._ffns and self.layers != layers:
            raise ExpertCompatibilityError(f"state is for layers {layers}, manager has {self.layers}")
        if not names or names[0] != BASE_EXPERT or len(set(names)) != len(names):
            raise ExpertCompatibilityError(f"invalid expert list in state: {names}")
        for n in names:
            _check_name(n)
        extra = set(self._expert_names) - set(names)
        if extra:
            raise ExpertCompatibilityError(f"model has experts {sorted(extra)} that the state does not know")
        trainable, frozen = state.get("trainable"), bool(state.get("frozen"))
        if trainable is not None and (trainable not in names or trainable == BASE_EXPERT or not frozen):
            raise ExpertCompatibilityError(f"invalid trainable expert {trainable!r} (frozen={frozen})")
        selected = [state.get("active"), *(state.get("routing") or [])]
        unknown = [n for n in selected if n not in names]
        if unknown:
            raise ExpertCompatibilityError(f"state selects unknown experts {unknown}")
        if state.get("routing") is not None and len(state["routing"]) == 0:
            raise ExpertCompatibilityError("state has an empty routing")
        return layers, names

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Recreate the expert structure (attach, create experts, trainability, selection).

        The whole state is validated before anything changes. Newly created experts are copies of
        ``base``; the caller (e.g. HF Trainer resume, or :func:`load_model_with_experts`) must load
        the model weights afterwards. The recorded backbone fingerprint is restored, not recomputed
        (and not checked, since the weights may not be loaded yet), so integrity checks after a
        resume compare against the original backbone.

        ``frozen=True`` restores the manager's ``requires_grad`` flags exactly. ``frozen=False``
        means the manager did not control ``requires_grad``: if *this* manager had frozen the model,
        its freezing is undone (every parameter trainable again); otherwise flags are left as they
        are (so a user's own freezing survives a resume). A detached state (``layers == []``) is a
        no-op on a detached manager and is rejected on an attached one.
        """
        layers, names = self._validate_state(state)
        if not layers:
            return
        if not self._ffns:
            self.attach(layers)
        was_frozen = self._frozen
        self._frozen = False  # create without re-freezing; trainability is applied below
        for n in names:
            if n not in self._expert_names:
                self.create_expert(n)
        self._expert_names = names
        self._trainable = state.get("trainable")
        self._frozen = bool(state.get("frozen"))
        if self._frozen:
            self._apply_trainability()
        elif was_frozen:
            for p in self.model.parameters():
                p.requires_grad_(True)
        self.activate(state["active"])
        if state.get("routing") is not None:
            self.set_routing(state["routing"])
        self._backbone_ref = state.get("backbone_fingerprint")

    def save_pretrained(self, path: str | Path, **save_kwargs: Any) -> Path:
        """Save the model *with experts attached* plus a manager sidecar (see :func:`load_model_with_experts`)."""
        self._require_attached()
        path = Path(path)
        self.model.save_pretrained(str(path), **save_kwargs)  # type: ignore[operator]
        (path / SIDECAR_FILE).write_text(json.dumps(self.state_dict(), indent=2))
        return path


# ---------------------------------------------------------------------- loading attached checkpoints
def _read_safetensors_dir(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    single = path / "model.safetensors"
    index = path / "model.safetensors.index.json"
    if single.exists():
        return load_file(str(single), device="cpu")
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        out: dict[str, torch.Tensor] = {}
        for shard in shards:
            out.update(load_file(str(path / shard), device="cpu"))
        return out
    raise ExpertCompatibilityError(f"no safetensors weights in {path}")


def load_model_with_experts(
    path: str | Path,
    model_factory: Callable[[Any, torch.dtype], nn.Module] | None = None,
) -> tuple[nn.Module, ExpertManager]:
    """Load a checkpoint saved with experts attached (``ExpertManager.save_pretrained`` or an HF
    Trainer checkpoint) and re-attach its experts. Fails if keys do not match exactly.

    ``model_factory(config, dtype)`` builds the empty model (default:
    ``AutoModelForCausalLM.from_config``). Layer/expert structure comes from the sidecar file when
    present, otherwise it is inferred from the ``...mlp.experts.<name>.*`` keys.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    path = Path(path)
    state = _read_safetensors_dir(path)
    config = AutoConfig.from_pretrained(str(path))
    float_dtypes = {t.dtype for t in state.values() if t.is_floating_point()}
    if len(float_dtypes) > 1:
        raise ExpertCompatibilityError(f"mixed floating dtypes in checkpoint: {float_dtypes}")
    dtype = float_dtypes.pop() if float_dtypes else torch.float32
    if model_factory is None:
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    else:
        model = model_factory(config, dtype)
    manager = ExpertManager(model)

    sidecar = path / SIDECAR_FILE
    manager_state: dict[str, Any] | None = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if manager_state is None:
        pat = re.compile(rf"^{re.escape(manager._layers_path)}\.(\d+)\.mlp\.experts\.([^.]+)\.")
        per_layer: dict[int, list[str]] = {}
        for key in state:
            m = pat.match(key)
            if m:
                names = per_layer.setdefault(int(m.group(1)), [])
                if m.group(2) not in names:
                    names.append(m.group(2))
        if not per_layer:
            raise ExpertCompatibilityError(f"{path} has no expert keys; load it with from_pretrained")
        name_sets = {tuple(sorted(v)) for v in per_layer.values()}
        if len(name_sets) != 1 or BASE_EXPERT not in next(iter(name_sets)):
            raise ExpertCompatibilityError(f"inconsistent expert names across layers: {per_layer}")
        first = per_layer[min(per_layer)]
        manager.attach(sorted(per_layer))
        for n in first:
            if n != BASE_EXPERT:
                manager.create_expert(n)
    else:
        if not manager_state.get("layers"):
            raise ExpertCompatibilityError(f"{path}: sidecar describes a detached manager (no experts)")
        manager.load_state_dict(manager_state)

    # Tied weights are stored once; any other missing or unexpected key is an error.
    model_sd = model.state_dict()
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ExpertCompatibilityError(f"unexpected keys in {path}: {result.unexpected_keys[:5]}")
    loaded_ptrs = {model_sd[k].data_ptr() for k in state if k in model_sd}
    untied = [k for k in result.missing_keys if model_sd[k].data_ptr() not in loaded_ptrs]
    if untied:
        raise ExpertCompatibilityError(f"missing keys in {path}: {untied[:5]}")
    if (path / "generation_config.json").exists() and hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_pretrained(str(path))
    model.eval()
    return model, manager


__all__ = [
    "BASE_EXPERT",
    "BackboneChangedError",
    "ExpertCompatibilityError",
    "ExpertError",
    "ExpertFFN",
    "ExpertManager",
    "config_hash",
    "load_model_with_experts",
]
