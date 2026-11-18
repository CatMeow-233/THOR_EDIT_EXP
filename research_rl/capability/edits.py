"""Declarative FFN-channel edits and an immutable, gate-checked version graph (``CAPABILITY.md`` 1.12).

An edit is data, never code: an :class:`EditRecord` names one :class:`EditOp`, a channel
selection ``{layer: [channel, ...]}`` and a small, strictly validated parameter dict. A
:class:`VersionStore` applies edits to a live model, evaluates them with the capability gate
against the parent version and records accepted edits as immutable versions on disk.

Operations (every one writes only the slices owned by the selected channels, through
:func:`~research_rl.models.ffn_channels.rollback_channels` /
:func:`~research_rl.models.ffn_channels.ablate_channels`, so validation runs before any write):

* ``ablate`` - ``params == {}``. Zero the channels' gate row, up row, down column (and gate/up
  bias entries) with :func:`~research_rl.models.ffn_channels.ablate_channels`.
* ``scale`` - ``params == {"factor": float, "up": bool = False}``. ``down_proj.weight[:, c] *=
  factor``; with ``up`` also ``up_proj.weight[c, :]`` (and ``up_proj.bias[c]`` if present). The
  arithmetic is that of :class:`~research_rl.capability.spaces.ChannelGainSpace`: computed in
  float32 (float64 for float64 parameters) and cast (round-to-nearest) to the parameter dtype;
  ``factor == 1`` leaves the bits unchanged. ``factor`` is any finite number (0 and negative
  values are allowed).
* ``revert`` - ``params == {"version": id}``. Copy the channels' slices from the stored channel
  state of version ``id`` (bit-exact copies).
* ``interpolate`` - ``params == {"version": id, "lam": float in [0, 1]}``. Write
  ``reference + lam * (current - reference)`` (the convention of
  :func:`~research_rl.capability.weights.interpolate`: ``lam`` is the share of the *current*
  weights kept). ``lam == 0`` is exactly ``revert`` (copies), ``lam == 1`` leaves the bits
  unchanged; interior values are computed with ``torch.lerp`` in float32 (float64 for float64
  parameters) and cast.
* ``apply_delta`` - ``params == {"delta": id, "factor": float = 1.0}``. Write
  ``current + factor * delta`` for the selected channels from a delta stored in the same store
  (:meth:`VersionStore.add_delta` / :meth:`VersionStore.delta_between`); every selected channel
  must be present in the delta. Computed in float32 (float64 for float64 parameters) and cast:
  applying ``delta_between(a, b)`` to version ``a`` reproduces ``b`` only up to rounding.

Ids (versions, states, deltas) are lowercase SHA-256 hex digests and must match
``[0-9a-f]{64}`` in full (``re.fullmatch``: no trailing newline or other character) wherever they
reach a path, so an id can never contain a path separator or ``..``.

Records are data the store copies. :meth:`VersionStore.stage` rebuilds every edit from its JSON
form (re-validating it, so a record whose dicts were mutated after construction into an invalid
value is refused before any write) and keeps a private canonical copy of exactly what it applied;
:meth:`VersionStore.commit` records that copy and the private copy of the last verdict and its
evaluation context, never the caller's (mutable) objects or ``StagedEdit``'s public lists. Every
:class:`Version` holds its own copies of its edits and JSON fields, and the store's accessors
(:meth:`VersionStore.get`, :meth:`~VersionStore.history`, ...) return fresh copies, so mutating a
returned object never changes the store's history.

Store layout (all writes stay inside ``root``; every write is a temporary file in ``root/tmp``
followed by ``os.replace``)::

    root/store.json                  immutable: format, channel layers, model signature, gate nonce
    root/head.json                   mutable: the version the live model was last put at
    root/ledger.json                 mutable: the store's own AlphaLedger + evaluation counter
    root/versions/<id>.json          immutable version records; id = SHA-256 of the record
    root/states/<id>.safetensors     channel states, content-addressed (id covers tensors + metadata)
    root/deltas/<id>.safetensors     deltas for apply_delta, content-addressed
    root/tmp/                        incomplete writes only; emptied when the store is opened

Opening a store rejects any other file or directory, any symlink, any ``.json`` that is not a
valid record of its kind and any tensor file that does not parse as safetensors (a ``.pt`` or a
pickle file is never loaded: tensors are read with :func:`safetensors.safe_open` only and
``torch.load`` is never called). A version record's id is recomputed on load (any edit of the
JSON, e.g. a tampered fingerprint, is rejected) and a tensor file's id is recomputed from its
tensors and metadata before use. Shapes, dtypes and module paths are checked against the model,
and :meth:`VersionStore.restore` computes the fingerprint the model *would* have and compares it
with the record before it writes anything.

What a version captures. A version stores the channel state
(:func:`~research_rl.models.ffn_channels.channel_state`) of the store's ``channel_layers`` and
the fingerprint (:func:`~research_rl.models.model_interface.fingerprint`) of the **whole**
model. Every other parameter (embeddings, attention, norms, ``lm_head``, ``down_proj.bias``, the
FFNs of other layers) is the *frame*: edits never touch it, its fingerprint is recorded, and
:meth:`VersionStore.restore` refuses (before writing) when the live frame differs, because the
store cannot restore it. Expert containers are addressed through their active expert (the
recorded module paths must match); quantised or sharded weights are not supported.

Gate evaluation. :meth:`VersionStore.evaluate` scores the staged weights against the parent's
weights (the snapshot taken at :meth:`VersionStore.stage`, swapped in by
:meth:`~research_rl.capability.probe.CapabilityProbe.evidence` with
:func:`~research_rl.capability.weights.swapped`) on fresh items, with
``decision_index = evaluation counter of the store``, ``attempt = 0`` and
``namespace = VersionStore.gate_namespace``, and decides with
:class:`~research_rl.capability.gate.CapabilityGate` using the **store's own** persisted
:class:`~research_rl.capability.stats.AlphaLedger` (never a training run's). The ledger's
``(total_alpha, schedule)`` is pinned by the first evaluation; later evaluations with another
``alpha`` / ``alpha_schedule`` are refused, so the spending sequence cannot be restarted by
changing the config. The ledger file is cross-checked on open against the hash-protected
version records: ``ledger.count == evaluations`` (``ledger`` is ``None`` iff ``evaluations ==
0``), every recorded evaluation has ``evaluation_index < evaluations`` and the verdict's
``decision_index`` equal to it, and its ``alpha`` / ``alpha_schedule`` equal the ledger's pin. This
is a lower bound only: evaluations of discarded edits leave no version record, so a ledger reset
to a count at or above the highest recorded index is not detectable.

Gate items per store. The items are a function of the probe's sampler and the key
``(f"edit/{gate_nonce}/gate/0", evaluation counter)``. ``gate_nonce`` is 128 random bits drawn
when the store is created and kept in ``store.json`` (and the namespace is recorded in every
evaluation context), so a new store draws an item stream of its own even with the same probe and
sampler seed: its evaluation ``k`` never replays the items of evaluation ``k`` of another store
or of a training run's decision ``k`` (purpose ``gate/<attempt>``). The streams are independent
draws from the same gate pool, so single items still repeat by chance, as between two decisions
of one run (:mod:`~research_rl.capability.probe`, "Item pools"). A legacy store whose
``store.json`` has no ``gate_nonce`` (created before it existed) still opens and keeps the old key
``("gate/0", evaluation counter)`` (:attr:`VersionStore.gate_namespace` is ``None``): its items
coincide with those of any other legacy store, or training run, on a sampler with the same
``root_seed``.

Commit and init are all-or-nothing in memory: if the head write fails after the version record
was written, the record written by that call is removed again (best effort; the content-addressed
state file is kept, it may be shared), so a retry does not leave an orphan sibling. A crash in
between can still leave a version that no head points to; :meth:`VersionStore.stage` then names
the :meth:`~VersionStore.restore` call that recovers. Writing a content-addressed tensor file whose
name already exists verifies the existing file and rewrites it if it is corrupted.

Not supported: concurrent writers on one root (single process, single thread), weights that
are not full local tensors. Everything is verified on CPU with tiny models
(``tests/research_rl/test_capability_edits.py``); nothing has run on a GPU (NOT_TESTED).
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import os
import re
import secrets
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from research_rl.capability.gate import CapabilityGate, GateConfig
from research_rl.capability.stats import AlphaLedger
from research_rl.capability.types import GateVerdict
from research_rl.capability.weights import StateSnapshot, capture
from research_rl.capability.weights import restore as restore_snapshot
from research_rl.models.ffn_channels import (
    ChannelSelection,
    ChannelSelectionError,
    ChannelState,
    LayerChannels,
    ScopeReport,
    ablate_channels,
    channel_diff,
    channel_state,
    extract_channels,
    rollback_channels,
    verify_modification_scope,
)
from research_rl.models.model_interface import find_swiglu_ffns, fingerprint, snapshot

if TYPE_CHECKING:  # the probe is only called, never constructed, here
    from research_rl.capability.probe import CapabilityProbe

__all__ = [
    "STORE_FORMAT",
    "CommitRefusedError",
    "EditError",
    "EditOp",
    "EditRecord",
    "StagedEdit",
    "StoreError",
    "Version",
    "VersionStore",
]

STORE_FORMAT = "research_rl.capability.edits/v1"
_STATE_FORMAT = "research_rl.capability.edits.channels/v1"
_ID_RE = re.compile(r"[0-9a-f]{64}")  # always used with fullmatch
_NONCE_RE = re.compile(r"[0-9a-f]{32}")  # always used with fullmatch
_STORE_KEYS = frozenset({"format", "channel_layers", "signature", "created", "gate_nonce"})
_LEGACY_STORE_KEYS = _STORE_KEYS - {"gate_nonce"}  # stores created before the per-store gate nonce


def _new_gate_nonce() -> str:
    """Fresh random gate nonce of a new store (128 bits, hex). Tests may monkeypatch this."""
    return secrets.token_hex(16)


MAX_TEXT = 10_000  # maximum length of a note / override reason (characters)

_DIRS = ("versions", "states", "deltas", "tmp")
_FILES = ("store.json", "head.json", "ledger.json")
_FIELDS = ("gate", "up", "down", "gate_bias", "up_bias")


# ============================================================================== errors
class EditError(ValueError):
    """An edit (or delta) is malformed or does not fit the model / store. Raised before any write."""


class StoreError(ValueError):
    """The store's files are missing, unsafe, corrupted, or do not fit the model."""


class CommitRefusedError(RuntimeError):
    """``commit`` without a passing gate verdict and without an override reason, or out of scope."""


# ============================================================================== helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _check_created(value: Any) -> str:
    if not isinstance(value, str):
        raise EditError(f"created must be an ISO-8601 string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise EditError(f"created is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise EditError(f"created must carry a timezone: {value!r}")
    return value


def _check_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise EditError(f"{name} must be a string, got {type(value).__name__}")
    if len(value) > MAX_TEXT:
        raise EditError(f"{name} is longer than {MAX_TEXT} characters")
    return value


def _check_id(value: Any, what: str = "id") -> str:
    """Return ``value`` if it is a 64-character lowercase hex digest (so it is path-safe)."""
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise EditError(f"{what} must be a 64-character lowercase hex digest, got {value!r}")
    return value


def _canonical(obj: Any) -> str:
    """Canonical JSON of a JSON-shaped object (string keys only), used for every content hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _number(name: str, value: Any, *, lo: float | None = None, hi: float | None = None) -> float:
    """``float(value)`` for a finite int / float (not bool) within ``[lo, hi]``; else :class:`EditError`.

    An int too large for a float (``OverflowError``) is rejected as non-finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditError(f"{name} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise EditError(f"{name} must be a finite number, got an int too large for a float") from None
    if not math.isfinite(number):
        raise EditError(f"{name} must be a finite number, got {value!r}")
    if (lo is not None and number < lo) or (hi is not None and number > hi):
        raise EditError(f"{name} must be in [{lo}, {hi}], got {value!r}")
    return number


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


# ============================================================================== edits
class EditOp(str, Enum):
    """The closed set of edit operations (see the module docstring for their exact semantics)."""

    ABLATE = "ablate"
    REVERT = "revert"
    SCALE = "scale"
    INTERPOLATE = "interpolate"
    APPLY_DELTA = "apply_delta"


def _normalise_params(op: EditOp, params: Any) -> dict[str, Any]:
    """Validate ``params`` against the schema of ``op``; return the normalised dict (defaults filled)."""
    if not isinstance(params, Mapping):
        raise EditError(f"params must be a mapping, got {type(params).__name__}")
    for key in params:
        if not isinstance(key, str):
            raise EditError(f"params keys must be strings, got {key!r}")
    allowed: dict[EditOp, tuple[set[str], set[str]]] = {  # (required, optional)
        EditOp.ABLATE: (set(), set()),
        EditOp.SCALE: ({"factor"}, {"up"}),
        EditOp.REVERT: ({"version"}, set()),
        EditOp.INTERPOLATE: ({"version", "lam"}, set()),
        EditOp.APPLY_DELTA: ({"delta"}, {"factor"}),
    }
    required, optional = allowed[op]
    unknown = sorted(set(params) - required - optional)
    if unknown:
        raise EditError(f"{op.value}: unknown params {unknown}; allowed {sorted(required | optional)}")
    missing = sorted(required - set(params))
    if missing:
        raise EditError(f"{op.value}: missing params {missing}")
    out: dict[str, Any] = {}
    if op is EditOp.SCALE:
        out["factor"] = _number("scale factor", params["factor"])
        up = params.get("up", False)
        if not isinstance(up, bool):
            raise EditError(f"scale 'up' must be a bool, got {up!r}")
        out["up"] = up
    elif op is EditOp.REVERT:
        out["version"] = _check_id(params["version"], "revert version")
    elif op is EditOp.INTERPOLATE:
        out["version"] = _check_id(params["version"], "interpolate version")
        out["lam"] = _number("interpolate lam", params["lam"], lo=0.0, hi=1.0)
    elif op is EditOp.APPLY_DELTA:
        out["delta"] = _check_id(params["delta"], "delta id")
        out["factor"] = _number("apply_delta factor", params.get("factor", 1.0))
    return out


def _normalise_selection(selection: Any) -> dict[int, list[int]]:
    """Strict ``{layer: [channel, ...]}``: int (or integer-literal string) layer keys, non-empty
    lists of distinct non-negative ints; returned sorted. A :class:`ChannelSelection` is accepted."""
    if isinstance(selection, ChannelSelection):
        selection = selection.to_dict()
    if not isinstance(selection, Mapping):
        raise EditError(f"selection must be a mapping layer -> channels, got {type(selection).__name__}")
    raw: dict[int | str, list[int]] = {}
    for layer, chans in selection.items():
        if isinstance(chans, (str, bytes)) or not isinstance(chans, (list, tuple)):
            raise EditError(f"selection[{layer!r}] must be a list of channel indices, got {chans!r}")
        if not chans:
            raise EditError(f"selection[{layer!r}] is empty")
        if len(set(chans)) != len(chans) or any(isinstance(c, bool) for c in chans):
            raise EditError(f"selection[{layer!r}] has duplicate or boolean entries: {chans!r}")
        raw[layer] = list(chans)
    try:
        sel = ChannelSelection.from_dict(raw)
    except ChannelSelectionError as exc:
        raise EditError(f"invalid selection: {exc}") from exc
    if not sel:
        raise EditError("selection must name at least one channel")
    return sel.to_dict()


@dataclass(frozen=True)
class EditRecord:
    """One declarative edit. Validated and normalised on construction (:class:`EditError`).

    * ``op``: an :class:`EditOp` (or its string value).
    * ``selection``: ``{layer: [channel, ...]}``; JSON-style string layer keys (``"3"``) are
      accepted; normalised to sorted int keys and sorted channel lists; empty selections, empty
      per-layer lists, duplicates, booleans and negative indices are rejected. Whether the
      indices exist in a model is checked when the edit is staged.
    * ``params``: exactly the schema of ``op`` (module docstring); unknown or missing keys,
      wrong types, booleans as numbers and non-finite numbers are rejected; optional keys are
      filled with their defaults and ints become floats, so equal edits compare equal.
    * ``note``: free text (at most :data:`MAX_TEXT` characters); never interpreted.
    * ``created``: ISO-8601 timestamp with a timezone (default: now, UTC).

    The dataclass is frozen but ``selection`` / ``params`` are plain (mutable) dicts, private
    copies of the arguments; validation runs on construction only. Treat a record as immutable:
    the store never relies on it (:meth:`VersionStore.stage` re-validates a copy and records what
    it applied). Unhashable because of the dicts. :meth:`to_dict` / :meth:`from_dict` and
    :meth:`to_json` / :meth:`from_json` round-trip exactly.
    """

    op: EditOp
    selection: dict[int, list[int]]
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    created: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        try:
            op = EditOp(self.op)
        except ValueError as exc:
            raise EditError(f"unknown edit op {self.op!r}; allowed {[o.value for o in EditOp]}") from exc
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "selection", _normalise_selection(self.selection))
        object.__setattr__(self, "params", _normalise_params(op, self.params))
        object.__setattr__(self, "note", _check_text("note", self.note))
        object.__setattr__(self, "created", _check_created(self.created))

    def copy(self) -> EditRecord:
        """A re-validated, independent copy (``from_dict(to_dict())``): :class:`EditError` if the
        record's dicts were mutated into an invalid or non-normalised value."""
        try:
            fresh = EditRecord.from_dict(json.loads(self.to_json()))
        except (TypeError, ValueError) as exc:  # EditError is a ValueError; json may raise TypeError
            if isinstance(exc, EditError):
                raise
            raise EditError(f"edit is not JSON-serialisable: {exc}") from exc
        if fresh.to_dict() != self.to_dict():
            raise EditError(
                f"edit was modified after construction and is not in normal form: {self.to_dict()}"
            )
        return fresh

    def channel_selection(self) -> ChannelSelection:
        """The selection as a :class:`ChannelSelection`."""
        return ChannelSelection(self.selection)

    def references(self) -> tuple[str | None, str | None]:
        """``(version id, delta id)`` this edit reads (``None`` where not applicable)."""
        return self.params.get("version"), self.params.get("delta")

    def to_dict(self) -> dict[str, Any]:
        """JSON-shaped dict (string layer keys)."""
        return {
            "op": self.op.value,
            "selection": {str(k): list(v) for k, v in self.selection.items()},
            "params": dict(self.params),
            "note": self.note,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EditRecord:
        """Inverse of :meth:`to_dict`. ``op``, ``selection`` and ``params`` are required; unknown
        keys raise :class:`EditError`."""
        if not isinstance(data, Mapping):
            raise EditError(f"an edit must be a mapping, got {type(data).__name__}")
        unknown = sorted(set(data) - {"op", "selection", "params", "note", "created"})
        if unknown:
            raise EditError(f"unknown edit keys {unknown}")
        missing = sorted({"op", "selection", "params"} - set(data))
        if missing:
            raise EditError(f"edit is missing {missing}")
        kwargs: dict[str, Any] = {"op": data["op"], "selection": data["selection"], "params": data["params"]}
        if "note" in data:
            kwargs["note"] = data["note"]
        if "created" in data:
            kwargs["created"] = data["created"]
        return cls(**kwargs)

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, no whitespace) of :meth:`to_dict`."""
        return _canonical(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> EditRecord:
        """Inverse of :meth:`to_json` (:class:`EditError` on invalid JSON or content)."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EditError(f"edit is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


# ============================================================================== versions
def _scope_dict(report: ScopeReport, outside_ok: bool) -> dict[str, Any]:
    return {
        "ok": bool(report.ok and outside_ok),
        "summary": report.summary(),
        "params_checked": report.params_checked,
        "elements_checked": report.elements_checked,
        "changed_allowed": report.changed_allowed,
        "violations": {
            name: {"count": v.count, "max_abs": v.max_abs} for name, v in sorted(report.violations.items())
        },
        "outside_unchanged": bool(outside_ok),
    }


@dataclass(frozen=True)
class Version:
    """An immutable version record (one JSON file ``versions/<id>.json``).

    * ``id``: SHA-256 of the canonical JSON of every other field (:meth:`record`).
    * ``parent``: parent version id (``None`` for the root).
    * ``edits``: the edits applied to the parent, in order (empty for the root).
    * ``verdict``: :meth:`GateVerdict.to_dict` of the last evaluation of the staged edit, or
      ``None``; ``evaluation``: its context (evaluation index, gate config, probe description).
    * ``override``: the override reason when the edit was committed without a passing verdict
      (or with one, if the caller also gave a reason), else ``None``.
    * ``fingerprint``: fingerprint of all model parameters at this version;
      ``frame_fingerprint``: of the parameters outside the store's channel layers.
    * ``state``: id of the channel-state file ``states/<state>.safetensors``.
    * ``scope``: the scope report of the commit (``verify_modification_scope`` against the
      parent, restricted to the edits' channels, plus the check that every other parameter kept
      its bits); ``None`` for the root.
    * ``note``, ``created``.

    Invariants (checked on construction and on load): the root has no edits / verdict / scope;
    every other version has edits, a passing verdict (``transfer_ok`` and ``retention_ok``, no
    alarm) or a non-empty override, and an ``ok`` scope report; a verdict comes with its
    evaluation context (``evaluation_index`` equal to the verdict's ``decision_index`` and the
    ``gate_config`` with its ``alpha`` / ``alpha_schedule``). On construction every edit is
    replaced by :meth:`EditRecord.copy` (re-validated) and ``verdict`` / ``evaluation`` / ``scope``
    by JSON round-trip copies, so a version never shares mutable state with its inputs.
    """

    id: str
    parent: str | None
    edits: tuple[EditRecord, ...]
    verdict: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    override: str | None
    fingerprint: str
    frame_fingerprint: str
    state: str
    scope: dict[str, Any] | None
    note: str
    created: str

    def __post_init__(self) -> None:
        if not isinstance(self.edits, tuple) or not all(isinstance(e, EditRecord) for e in self.edits):
            raise EditError("edits must be a tuple of EditRecord")
        object.__setattr__(self, "edits", tuple(e.copy() for e in self.edits))
        for name in ("verdict", "evaluation", "scope"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, dict):
                    raise EditError(f"{name} must be a dict or None")
                try:
                    object.__setattr__(self, name, json.loads(_canonical(value)))
                except (TypeError, ValueError) as exc:
                    raise EditError(f"{name} is not JSON-shaped: {exc}") from exc
        _check_id(self.id, "version id")
        if self.parent is not None:
            _check_id(self.parent, "parent id")
        _check_id(self.state, "state id")
        for name in ("fingerprint", "frame_fingerprint"):
            _check_id(getattr(self, name), name)
        _check_text("note", self.note)
        _check_created(self.created)
        if self.override is not None and (not isinstance(self.override, str) or not self.override.strip()):
            raise EditError("override must be None or a non-empty reason")
        if self.override is not None:
            _check_text("override", self.override)
        if self.verdict is not None:
            self.evaluation_pin()  # a verdict needs a well-formed evaluation context
        elif self.evaluation is not None:
            raise EditError("an evaluation context without a verdict")
        if self.parent is None:
            if self.edits or self.verdict is not None or self.scope is not None or self.override is not None:
                raise EditError("the root version has no edits, verdict, override or scope")
        else:
            if not self.edits:
                raise EditError("a non-root version needs at least one edit")
            if not (self.passed or self.override is not None):
                raise EditError("a non-root version needs a passing verdict or an override reason")
            if self.scope is None or self.scope.get("ok") is not True:
                raise EditError("a non-root version needs an ok scope report")

    def evaluation_pin(self) -> tuple[int, float, str] | None:
        """``(evaluation_index, alpha, alpha_schedule)`` of the recorded evaluation, or ``None``
        without a verdict. :class:`EditError` if the context is malformed or its index differs
        from the verdict's ``decision_index``."""
        if self.verdict is None:
            return None
        ev = self.evaluation
        if not isinstance(ev, dict):
            raise EditError("a verdict needs its evaluation context")
        index = ev.get("evaluation_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise EditError(f"evaluation_index must be an int >= 0, got {index!r}")
        if self.verdict.get("decision_index") != index or isinstance(
            self.verdict.get("decision_index"), bool
        ):
            raise EditError(
                f"verdict decision_index {self.verdict.get('decision_index')!r} != evaluation_index {index}"
            )
        gc = ev.get("gate_config")
        if not isinstance(gc, dict):
            raise EditError("evaluation gate_config must be a dict")
        alpha, schedule = gc.get("alpha"), gc.get("alpha_schedule")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not isinstance(schedule, str):
            raise EditError(
                f"evaluation gate_config has malformed alpha / alpha_schedule: {alpha!r}, {schedule!r}"
            )
        return index, float(alpha), schedule

    @property
    def passed(self) -> bool:
        """Whether the recorded verdict passed (transfer and retention, no alarm)."""
        v = self.verdict
        return bool(
            v is not None
            and v.get("transfer_ok") is True
            and v.get("retention_ok") is True
            and v.get("alarm") is False
        )

    def record(self) -> dict[str, Any]:
        """JSON-shaped dict of every field except ``id`` (the hashed content)."""
        return {
            "format": STORE_FORMAT,
            "parent": self.parent,
            "edits": [e.to_dict() for e in self.edits],
            "verdict": self.verdict,
            "evaluation": self.evaluation,
            "override": self.override,
            "fingerprint": self.fingerprint,
            "frame_fingerprint": self.frame_fingerprint,
            "state": self.state,
            "scope": self.scope,
            "note": self.note,
            "created": self.created,
        }

    def to_dict(self) -> dict[str, Any]:
        """The JSON record as stored in ``versions/<id>.json``."""
        return {"id": self.id, **self.record()}

    @staticmethod
    def compute_id(record: Mapping[str, Any]) -> str:
        """SHA-256 hex of the canonical JSON of ``record`` (a JSON-shaped dict without ``id``)."""
        return _sha(_canonical(dict(record)))

    @classmethod
    def build(cls, **fields_: Any) -> Version:
        """Construct a version and compute its id from the other fields."""
        probe = cls(id="0" * 64, **fields_)
        return cls(id=cls.compute_id(probe.record()), **fields_)

    @classmethod
    def from_dict(cls, data: Any) -> Version:
        """Strict inverse of :meth:`to_dict`: the id must equal the hash of the stored content and
        the content must already be in normal form (``from_dict(d).to_dict() == d``)."""
        if not isinstance(data, Mapping):
            raise StoreError("a version record must be a JSON object")
        expected = {"id", *cls(id="0" * 64, **_ROOT_TEMPLATE).record()}
        if set(data) != expected:
            raise StoreError(f"version record keys {sorted(data)} != {sorted(expected)}")
        if data["format"] != STORE_FORMAT:
            raise StoreError(f"unknown version format {data['format']!r}")
        edits_raw = data["edits"]
        if not isinstance(edits_raw, list):
            raise StoreError("version edits must be a list")
        try:
            version = cls(
                id=data["id"],
                parent=data["parent"],
                edits=tuple(EditRecord.from_dict(e) for e in edits_raw),
                verdict=data["verdict"],
                evaluation=data["evaluation"],
                override=data["override"],
                fingerprint=data["fingerprint"],
                frame_fingerprint=data["frame_fingerprint"],
                state=data["state"],
                scope=data["scope"],
                note=data["note"],
                created=data["created"],
            )
        except EditError as exc:
            raise StoreError(f"invalid version record: {exc}") from exc
        content = {k: v for k, v in data.items() if k != "id"}
        if cls.compute_id(content) != data["id"]:
            raise StoreError(f"version {data['id']}: content hash mismatch (record was modified)")
        if _canonical(version.to_dict()) != _canonical(dict(data)):
            raise StoreError(f"version {data['id']}: record is not in normal form")
        return version


_ROOT_TEMPLATE: dict[str, Any] = {
    "parent": None,
    "edits": (),
    "verdict": None,
    "evaluation": None,
    "override": None,
    "fingerprint": "0" * 64,
    "frame_fingerprint": "0" * 64,
    "state": "0" * 64,
    "scope": None,
    "note": "",
    "created": "2000-01-01T00:00:00+00:00",
}


def _fresh(version: Version) -> Version:
    """An independent copy of ``version`` (re-parsed from its JSON record, so re-verified)."""
    return Version.from_dict(json.loads(_canonical(version.to_dict())))


# ============================================================================== staged edits
@dataclass(eq=False)
class StagedEdit:
    """Edits applied to the live model on top of ``parent``, not yet committed.

    * ``incumbent``: :func:`~research_rl.capability.weights.capture` of the FFN parameters of the
      affected layers *before* the edits (the parent's weights there): the incumbent of
      :meth:`VersionStore.evaluate`, the restore point of :meth:`VersionStore.discard` and the
      ``before`` of the commit's scope check.
    * ``outside_fingerprint``: fingerprint of every other parameter at stage time.
    * ``fingerprint``: fingerprint of the whole model right after staging; evaluate / commit
      refuse if the model no longer has it.
    * ``verdicts`` / ``evaluations``: every evaluation of this staged edit, in order; a commit
      without override needs the **last** one to pass.
    * ``status``: ``"staged"``, ``"discarded"`` or ``"committed"`` (only ``"staged"`` is usable).

    Every field is informational for the caller: the store keeps its own private copy of the
    parent, the applied edits (canonical JSON), the selection, both fingerprints, the last verdict
    and its evaluation context, and :meth:`VersionStore.evaluate` / :meth:`VersionStore.commit`
    use only that copy. Mutating ``edits``, ``verdicts``, ``evaluations``, ``fingerprint`` etc.
    therefore changes neither what is evaluated nor what is committed (``passed`` reflects the
    public list only). Limit: ``incumbent`` holds the parent's tensors and is shared, not copied
    again; writing into its tensors changes the incumbent and the discard restore point.
    """

    parent: str
    edits: tuple[EditRecord, ...]
    selection: ChannelSelection
    incumbent: StateSnapshot
    outside_fingerprint: str
    fingerprint: str
    verdicts: list[GateVerdict] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "staged"

    @property
    def last_verdict(self) -> GateVerdict | None:
        """The verdict of the most recent evaluation, or ``None``."""
        return self.verdicts[-1] if self.verdicts else None

    @property
    def passed(self) -> bool:
        """True iff the last evaluation passed transfer and retention without an alarm."""
        v = self.last_verdict
        return bool(v is not None and v.transfer_ok and v.retention_ok and not v.alarm)


@dataclass(eq=False)
class _StagedFacts:
    """The store's private record of the active staged edit (see :class:`StagedEdit`)."""

    parent: str
    edits_json: str  # canonical JSON list of EditRecord.to_dict(): exactly what was applied
    selection: ChannelSelection
    outside_fingerprint: str
    fingerprint: str
    verdict: GateVerdict | None = None
    evaluation_json: str | None = None

    def edits(self) -> tuple[EditRecord, ...]:
        return tuple(EditRecord.from_dict(d) for d in json.loads(self.edits_json))

    @property
    def passed(self) -> bool:
        v = self.verdict
        return bool(v is not None and v.transfer_ok and v.retention_ok and not v.alarm)


# ============================================================================== the store
class VersionStore:
    """Immutable version graph of a model's FFN channel state, persisted under ``root``.

    ``channel_layers``: the FFN layers whose channel state is versioned (default: every SwiGLU
    layer). Edits may only select channels of these layers. Opening an existing store checks the
    model's signature (parameter names, shapes, dtypes) and, if given, ``channel_layers``
    against ``store.json``; a new store is created only in an empty or missing directory. The
    live model is **not** assumed to be at the head on open: :meth:`stage` requires
    ``fingerprint(model) == head.fingerprint`` (use :meth:`restore` first).
    """

    def __init__(
        self, root: str | Path, model: nn.Module, *, channel_layers: Iterable[int] | None = None
    ) -> None:
        self.model = model
        self.root = Path(root).absolute()
        given = None if channel_layers is None else sorted(set(channel_layers))
        exists = (self.root / "store.json").exists()
        wanted = given if given is not None or not exists else self._peek_layers()
        refs = find_swiglu_ffns(model, wanted)
        layers = sorted(r.layer_index for r in refs)
        self.channel_layers: list[int] = layers
        if not layers:
            raise EditError("the store needs at least one FFN channel layer")
        self._refs = {r.layer_index: r for r in refs}
        self._channel_names: list[str] = []
        for r in refs:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                self._channel_names.append(f"{r.module_path}.{proj}.weight")
            for proj, lin in (("gate_proj", r.gate), ("up_proj", r.up)):
                if lin.bias is not None:
                    self._channel_names.append(f"{r.module_path}.{proj}.bias")
        named = dict(model.named_parameters())
        missing = [n for n in self._channel_names if n not in named]
        if missing:
            raise EditError(f"FFN parameters not found by name in named_parameters(): {missing[:5]}")
        self._channel_name_set = frozenset(self._channel_names)
        self._frame_names = [n for n in named if n not in self._channel_name_set]
        self._signature = _sha(
            _canonical([[n, list(p.shape), str(p.dtype)] for n, p in named.items()] + [layers])
        )
        self._tmp_counter = itertools.count()
        self._staged: StagedEdit | None = None
        self._facts: _StagedFacts | None = None
        self._versions: dict[str, Version] = {}
        self._head: str | None = None
        self._ledger: AlphaLedger | None = None
        self._evaluations = 0
        self._gate_namespace: str | None = None
        if exists:
            self._open(given)
        else:
            self._create(layers)

    def _peek_layers(self) -> list[int]:
        meta = self._read_json(self.root / "store.json")
        layers = meta.get("channel_layers") if isinstance(meta, dict) else None
        if (
            not isinstance(layers, list)
            or not layers
            or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in layers)
        ):
            raise StoreError("store.json channel_layers is malformed")
        return sorted(set(layers))

    # ------------------------------------------------------------------ open / create
    def _create(self, layers: list[int]) -> None:
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise StoreError(f"{self.root} exists and is not an empty directory (and has no store.json)")
        nonce = _new_gate_nonce()
        if not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
            raise StoreError(f"gate nonce must be 32 lowercase hex characters, got {nonce!r}")
        self.root.mkdir(parents=True, exist_ok=True)
        for d in _DIRS:
            (self.root / d).mkdir(exist_ok=True)
        meta = {
            "format": STORE_FORMAT,
            "channel_layers": layers,
            "signature": self._signature,
            "created": _now(),
            "gate_nonce": nonce,
        }
        self._write_json("store.json", meta)
        self._write_head(None)
        self._write_ledger(None, 0)
        self._gate_namespace = f"edit/{nonce}"

    def _open(self, layers: list[int] | None) -> None:
        if self.root.is_symlink() or not self.root.is_dir():
            raise StoreError(f"{self.root} is not a directory")
        self._scan_layout()
        meta = self._read_json(self.root / "store.json")
        if not isinstance(meta, dict) or set(meta) not in (_STORE_KEYS, _LEGACY_STORE_KEYS):
            raise StoreError("store.json is malformed")
        if meta["format"] != STORE_FORMAT:
            raise StoreError(f"unknown store format {meta['format']!r}")
        nonce = meta.get("gate_nonce")
        if nonce is not None and (not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce)):
            raise StoreError(f"store.json gate_nonce {nonce!r} is malformed (32 lowercase hex characters)")
        if "gate_nonce" in meta and nonce is None:
            raise StoreError("store.json gate_nonce is null")
        stored_layers = meta["channel_layers"]
        if not isinstance(stored_layers, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in stored_layers
        ):
            raise StoreError("store.json channel_layers is malformed")
        if (layers is not None and layers != stored_layers) or stored_layers != sorted(self._refs):
            raise StoreError(f"channel_layers {sorted(self._refs)} != the store's {stored_layers}")
        if meta["signature"] != self._signature:
            raise StoreError("the model does not match the store (parameter names / shapes / dtypes differ)")
        self.channel_layers = list(stored_layers)
        for d in ("states", "deltas"):
            for path in sorted((self.root / d).iterdir()):
                self._read_header(path)
        versions: dict[str, Version] = {}
        for path in sorted((self.root / "versions").iterdir()):
            v = Version.from_dict(self._read_json(path))
            if path.name != f"{v.id}.json":
                raise StoreError(f"{path.name}: file name does not match version id {v.id}")
            versions[v.id] = v
        roots = [v for v in versions.values() if v.parent is None]
        if versions and len(roots) != 1:
            raise StoreError(f"a store has exactly one root version, found {len(roots)}")
        for v in versions.values():
            if v.parent is not None and v.parent not in versions:
                raise StoreError(f"version {v.id}: parent {v.parent} is missing")
            if not (self.root / "states" / f"{v.state}.safetensors").is_file():
                raise StoreError(f"version {v.id}: state file {v.state} is missing")
            for e in v.edits:
                ver, delta = e.references()
                if ver is not None and ver not in versions:
                    raise StoreError(f"version {v.id}: edit references missing version {ver}")
                if delta is not None and not (self.root / "deltas" / f"{delta}.safetensors").is_file():
                    raise StoreError(f"version {v.id}: edit references missing delta {delta}")
        self._versions = versions
        head = self._read_json(self.root / "head.json")
        if not isinstance(head, dict) or set(head) != {"format", "head"} or head["format"] != STORE_FORMAT:
            raise StoreError("head.json is malformed")
        if head["head"] is not None and head["head"] not in versions:
            raise StoreError(f"head {head['head']!r} is not a version of this store")
        self._head = head["head"]
        led = self._read_json(self.root / "ledger.json")
        if not isinstance(led, dict) or set(led) != {"format", "ledger", "evaluations"}:
            raise StoreError("ledger.json is malformed")
        ev = led["evaluations"]
        if isinstance(ev, bool) or not isinstance(ev, int) or ev < 0:
            raise StoreError(f"ledger.json evaluations {ev!r} is invalid")
        self._evaluations = ev
        if led["ledger"] is not None:
            state = led["ledger"]
            if not isinstance(state, dict):
                raise StoreError("ledger.json ledger is malformed")
            try:
                ledger = AlphaLedger(float(state["total_alpha"]), state["schedule"])
                ledger.load_state_dict(state)
            except (KeyError, TypeError, ValueError) as exc:
                raise StoreError(f"ledger.json ledger state is invalid: {exc}") from exc
            self._ledger = ledger
        try:
            self._check_evaluation(None, self._ledger, self._evaluations)  # also for a store without versions
            for v in versions.values():
                self._check_evaluation(v, self._ledger, self._evaluations)
        except EditError as exc:
            raise StoreError(f"ledger.json is inconsistent with the version records: {exc}") from exc
        self._gate_namespace = None if nonce is None else f"edit/{nonce}"
        for v in versions.values():
            ev = v.evaluation
            if (
                isinstance(ev, dict)
                and "gate_namespace" in ev
                and ev["gate_namespace"] != self._gate_namespace
            ):
                raise StoreError(
                    f"version {v.id} was evaluated on gate namespace {ev['gate_namespace']!r}, "
                    f"not this store's {self._gate_namespace!r}"
                )
        shutil.rmtree(self.root / "tmp")
        (self.root / "tmp").mkdir()

    @staticmethod
    def _check_evaluation(version: Version | None, ledger: AlphaLedger | None, evaluations: int) -> None:
        """Cross-check the store's ledger state against one version record (:class:`EditError`).

        ``ledger`` is ``None`` iff ``evaluations == 0``, else ``ledger.count == evaluations``; a
        recorded evaluation needs ``evaluation_index < evaluations`` and the ledger's pin
        ``(total_alpha, schedule)`` equal to its ``(alpha, alpha_schedule)``. ``version=None``
        checks the ledger / counter pair only."""
        if (ledger is None) != (evaluations == 0):
            raise EditError(
                f"ledger is {'missing' if ledger is None else 'present'} after {evaluations} evaluations"
            )
        if ledger is not None and ledger.count != evaluations:
            raise EditError(f"ledger count {ledger.count} != evaluation counter {evaluations}")
        if version is None:
            return
        pin = version.evaluation_pin()
        if pin is None:
            return
        index, alpha, schedule = pin
        if index >= evaluations:
            raise EditError(
                f"version {version.id} records evaluation {index}, but the counter is {evaluations}"
            )
        assert ledger is not None  # evaluations > index >= 0
        if (ledger.total_alpha, ledger.schedule) != (alpha, schedule):
            raise EditError(
                f"version {version.id} was evaluated with alpha={alpha}, schedule={schedule!r}; the "
                f"ledger is pinned to alpha={ledger.total_alpha}, schedule={ledger.schedule!r}"
            )

    def _scan_layout(self) -> None:
        """Reject anything in ``root`` that is not part of the store layout (no symlinks)."""
        patterns = {
            "versions": re.compile(r"[0-9a-f]{64}\.json"),
            "states": re.compile(r"[0-9a-f]{64}\.safetensors"),
            "deltas": re.compile(r"[0-9a-f]{64}\.safetensors"),
        }
        for entry in sorted(self.root.iterdir()):
            if entry.is_symlink():
                raise StoreError(f"symlinks are not allowed in a store: {entry}")
            if entry.name in _FILES and entry.is_file():
                continue
            if entry.name in _DIRS and entry.is_dir():
                if entry.name == "tmp":
                    continue  # incomplete writes; removed after a successful open
                for sub in sorted(entry.iterdir()):
                    if sub.is_symlink() or not sub.is_file() or not patterns[entry.name].fullmatch(sub.name):
                        raise StoreError(
                            f"unexpected entry {sub!r} (only {patterns[entry.name].pattern} files belong "
                            f"in {entry.name}/; tensors are loaded from safetensors only)"
                        )
                continue
            raise StoreError(
                f"unexpected entry {entry} in the store (tensor files other than safetensors are rejected)"
            )
        for name in (*_FILES, *_DIRS):
            if not (self.root / name).exists():
                raise StoreError(f"store is incomplete: {name} is missing")

    # ------------------------------------------------------------------ paths and atomic writes
    def _path(self, directory: str, file_id: str, suffix: str) -> Path:
        _check_id(file_id)
        if directory not in ("versions", "states", "deltas"):
            raise StoreError(f"unknown store directory {directory!r}")
        base = (self.root / directory).resolve()
        path = base / f"{file_id}{suffix}"
        if path.resolve().parent != base or not base.is_relative_to(self.root.resolve()):
            raise StoreError(f"path {path} escapes the store")
        return path

    def _atomic_write(self, path: Path, data: bytes) -> None:
        root = self.root.resolve()
        target = path.resolve()
        if not target.is_relative_to(root) or target == root:
            raise StoreError(f"refusing to write outside the store: {path}")
        tmp = root / "tmp" / f"{os.getpid()}-{time.time_ns()}-{next(self._tmp_counter)}.tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _write_json(self, name: str, obj: Any) -> None:
        self._atomic_write(self.root / name, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read {path}: {exc}") from exc

    def _write_head(self, head: str | None) -> None:
        self._write_json("head.json", {"format": STORE_FORMAT, "head": head})
        self._head = head

    def _write_ledger(self, ledger: AlphaLedger | None, evaluations: int) -> None:
        """Persist ``(ledger, evaluations)``; the caller adopts them in memory only after this."""
        self._write_json(
            "ledger.json",
            {
                "format": STORE_FORMAT,
                "ledger": None if ledger is None else ledger.state_dict(),
                "evaluations": evaluations,
            },
        )

    # ------------------------------------------------------------------ tensor files
    def _write_channels(self, directory: str, kind: str, state: ChannelState) -> str:
        tensors: dict[str, torch.Tensor] = {}
        paths: dict[str, str] = {}
        for layer in sorted(state.layers):
            lc = state[layer]
            tensors[f"{layer}.channels"] = lc.channels.detach().cpu().to(torch.long).contiguous()
            for name in _FIELDS:
                t = getattr(lc, name)
                if t is not None:
                    tensors[f"{layer}.{name}"] = t.detach().cpu().contiguous()
            paths[str(layer)] = lc.module_path or ""
        meta = {"format": _STATE_FORMAT, "kind": kind, "module_paths": _canonical(paths)}
        file_id = _sha(_canonical({"tensors": fingerprint(tensors), "meta": meta}))
        path = self._path(directory, file_id, ".safetensors")
        if path.is_symlink():
            raise StoreError(f"symlinks are not allowed in a store: {path}")
        if path.exists():
            try:
                self._read_channels(directory, kind, file_id)  # verifies the content hash
                return file_id
            except StoreError:
                pass  # corrupted: the in-memory content is by definition what the id names; rewrite
        from safetensors.torch import save

        self._atomic_write(path, save(tensors, metadata=meta))
        return file_id

    @staticmethod
    def _read_header(path: Path) -> dict[str, str]:
        from safetensors import safe_open

        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                meta = dict(f.metadata() or {})
        except Exception as exc:  # safetensors raises its own error types for malformed files
            raise StoreError(f"{path} is not a valid safetensors file: {exc}") from exc
        if meta.get("format") != _STATE_FORMAT or meta.get("kind") not in ("state", "delta"):
            raise StoreError(f"{path} is not a channel file of this store (metadata {meta})")
        return meta

    def _read_channels(self, directory: str, kind: str, file_id: str) -> ChannelState:
        """Load a channel file with ``safe_open`` only; verify its id from tensors and metadata."""
        from safetensors import safe_open

        path = self._path(directory, file_id, ".safetensors")
        if path.is_symlink() or not path.is_file():
            raise StoreError(f"{directory} file {file_id} does not exist")
        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                meta = dict(f.metadata() or {})
                tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 - safe_open has no __iter__
        except Exception as exc:  # safetensors raises its own error types for malformed files
            raise StoreError(f"{path} is not a valid safetensors file: {exc}") from exc
        if set(meta) != {"format", "kind", "module_paths"} or meta["format"] != _STATE_FORMAT:
            raise StoreError(f"{path}: unexpected metadata {meta}")
        if meta["kind"] != kind:
            raise StoreError(f"{path}: is a {meta['kind']!r} file, expected {kind!r}")
        if _sha(_canonical({"tensors": fingerprint(tensors), "meta": meta})) != file_id:
            raise StoreError(f"{path}: content hash mismatch (file was modified)")
        try:
            paths = json.loads(meta["module_paths"])
        except json.JSONDecodeError as exc:
            raise StoreError(f"{path}: module_paths metadata is not JSON") from exc
        if not isinstance(paths, dict):
            raise StoreError(f"{path}: module_paths metadata is malformed")
        layers: dict[int, LayerChannels] = {}
        seen: set[str] = set()
        for key, mpath in paths.items():
            if not (isinstance(key, str) and re.fullmatch(r"[0-9]+", key) and isinstance(mpath, str)):
                raise StoreError(f"{path}: malformed layer entry {key!r}")
            parts = {n: tensors.get(f"{key}.{n}") for n in ("channels", *_FIELDS)}
            seen.update(f"{key}.{n}" for n, t in parts.items() if t is not None)
            ch, gate, up, down = parts["channels"], parts["gate"], parts["up"], parts["down"]
            if ch is None or gate is None or up is None or down is None:
                raise StoreError(f"{path}: layer {key} lacks channels/gate/up/down tensors")
            if ch.dtype != torch.long or ch.dim() != 1:
                raise StoreError(f"{path}: layer {key} channels must be a 1-D int64 tensor")
            k = int(ch.numel())
            if gate.dim() != 2 or up.dim() != 2 or down.dim() != 2:
                raise StoreError(f"{path}: layer {key} weights must be 2-D")
            if gate.shape[0] != k or up.shape[0] != k or down.shape[1] != k:
                raise StoreError(f"{path}: layer {key} tensor shapes do not match {k} channels")
            for bname in ("gate_bias", "up_bias"):
                b = parts[bname]
                if b is not None and tuple(b.shape) != (k,):
                    raise StoreError(f"{path}: layer {key} {bname} shape {tuple(b.shape)} != ({k},)")
            layers[int(key)] = LayerChannels(
                channels=ch,
                gate=gate,
                up=up,
                down=down,
                gate_bias=parts["gate_bias"],
                up_bias=parts["up_bias"],
                module_path=mpath or None,
            )
        extra = sorted(set(tensors) - seen)
        if extra:
            raise StoreError(f"{path}: unexpected tensors {extra[:5]}")
        return ChannelState(layers)

    def _check_against_model(self, state: ChannelState, *, full: bool, what: str) -> None:
        """Layers, channels, module paths, bias presence, shapes and dtypes against the model.

        ``full``: a version state (every channel of every channel layer, parameter dtypes);
        otherwise a delta (a subset of channels of channel layers, floating dtype)."""
        if full and sorted(state.layers) != self.channel_layers:
            raise StoreError(f"{what}: layers {sorted(state.layers)} != store layers {self.channel_layers}")
        for layer, lc in state.layers.items():
            ref = self._refs.get(layer)
            if ref is None:
                raise StoreError(f"{what}: layer {layer} is not a channel layer of this store")
            if lc.module_path is not None and lc.module_path != ref.module_path:
                raise StoreError(
                    f"{what}: layer {layer} module path {lc.module_path!r} != {ref.module_path!r}"
                )
            chans = lc.channels.tolist()
            n = ref.intermediate_size
            if full and chans != list(range(n)):
                raise StoreError(f"{what}: layer {layer} must hold channels 0..{n - 1} in order")
            if not full and (chans != sorted(set(chans)) or not chans or chans[0] < 0 or chans[-1] >= n):
                raise StoreError(f"{what}: layer {layer} channels must be sorted, unique and < {n}")
            k = len(chans)
            pairs: list[tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...]]] = [
                ("gate", lc.gate, ref.gate.weight, (k, ref.gate.weight.shape[1])),
                ("up", lc.up, ref.up.weight, (k, ref.up.weight.shape[1])),
                ("down", lc.down, ref.down.weight, (ref.down.weight.shape[0], k)),
            ]
            for bname, b, lin in (("gate_bias", lc.gate_bias, ref.gate), ("up_bias", lc.up_bias, ref.up)):
                if (b is None) != (lin.bias is None):
                    raise StoreError(f"{what}: layer {layer} {bname} presence differs from the model")
                if b is not None and lin.bias is not None:
                    pairs.append((bname, b, lin.bias, (k,)))
            for name, t, p, shape in pairs:
                if tuple(t.shape) != tuple(shape):
                    raise StoreError(f"{what}: layer {layer} {name} shape {tuple(t.shape)} != {tuple(shape)}")
                if full and t.dtype != p.dtype:
                    raise StoreError(f"{what}: layer {layer} {name} dtype {t.dtype} != parameter {p.dtype}")
                if not full and not t.dtype.is_floating_point:
                    raise StoreError(f"{what}: layer {layer} {name} must be floating point, got {t.dtype}")

    # ------------------------------------------------------------------ fingerprints
    def _frame_fingerprint(self) -> str:
        return fingerprint(self.model, self._frame_names)

    def _expected_fingerprint(self, state: ChannelState) -> str:
        """Fingerprint the model would have with the live frame and ``state`` in the channel layers."""
        tensors: dict[str, torch.Tensor] = {
            n: p for n, p in self.model.named_parameters() if n not in self._channel_name_set
        }
        for layer, lc in state.layers.items():
            mp = self._refs[layer].module_path
            tensors[f"{mp}.gate_proj.weight"] = lc.gate
            tensors[f"{mp}.up_proj.weight"] = lc.up
            tensors[f"{mp}.down_proj.weight"] = lc.down
            if lc.gate_bias is not None:
                tensors[f"{mp}.gate_proj.bias"] = lc.gate_bias
            if lc.up_bias is not None:
                tensors[f"{mp}.up_proj.bias"] = lc.up_bias
        return fingerprint(tensors)

    def _ffn_names(self, layers: Iterable[int]) -> list[str]:
        """Every gate/up/down weight and bias of ``layers`` (``down_proj.bias`` included: it is
        owned by no channel, so the scope check must see it to verify it did not change)."""
        names: list[str] = []
        for layer in layers:
            r = self._refs[layer]
            for proj, lin in (("gate_proj", r.gate), ("up_proj", r.up), ("down_proj", r.down)):
                names.append(f"{r.module_path}.{proj}.weight")
                if lin.bias is not None:
                    names.append(f"{r.module_path}.{proj}.bias")
        return names

    # ------------------------------------------------------------------ queries
    @property
    def head(self) -> str | None:
        """Id of the version the live model was last put at by this store (``None`` before init)."""
        return self._head

    @property
    def staged(self) -> StagedEdit | None:
        """The active staged edit, if any."""
        return self._staged

    @property
    def evaluation_count(self) -> int:
        """Evaluations run by this store so far (the next evaluation's ``decision_index``)."""
        return self._evaluations

    @property
    def gate_namespace(self) -> str | None:
        """Sampler namespace of this store's gate items, ``"edit/<gate_nonce>"`` (the nonce is drawn
        at random when the store is created and kept in ``store.json``), so two stores never share
        an item stream; ``None`` for a legacy store created without a nonce (its items use the
        training-run purpose ``gate/0``)."""
        return self._gate_namespace

    @property
    def ledger(self) -> AlphaLedger | None:
        """A copy of the store's own alpha ledger (``None`` before the first evaluation)."""
        if self._ledger is None:
            return None
        copy = AlphaLedger(self._ledger.total_alpha, self._ledger.schedule)
        copy.load_state_dict(self._ledger.state_dict())
        return copy

    def _version(self, version_id: str) -> Version:
        """The store's own record (internal; never handed out)."""
        _check_id(version_id, "version id")
        try:
            return self._versions[version_id]
        except KeyError:
            raise KeyError(f"no version {version_id} in this store") from None

    def get(self, version_id: str) -> Version:
        """A copy of version ``version_id`` (:class:`EditError` for a malformed id, ``KeyError`` if
        unknown). Mutating the copy's dicts never changes the store."""
        return _fresh(self._version(version_id))

    def versions(self) -> list[Version]:
        """Copies of every version, ordered by ``(created, id)``."""
        return [_fresh(v) for v in sorted(self._versions.values(), key=lambda v: (v.created, v.id))]

    def history(self, version_id: str | None = None) -> list[Version]:
        """Copies of the lineage root -> ``version_id`` (default: the head; ``[]`` for an empty store)."""
        vid = self._head if version_id is None else version_id
        if vid is None:
            return []
        chain: list[Version] = []
        cur: str | None = vid
        while cur is not None:
            v = self._version(cur)
            chain.append(v)
            cur = v.parent
        return [_fresh(v) for v in chain[::-1]]

    def load_state(self, version_id: str) -> ChannelState:
        """The stored channel state of a version (validated against the model)."""
        v = self._version(version_id)
        state = self._read_channels("states", "state", v.state)
        self._check_against_model(state, full=True, what=f"version {version_id}")
        return state

    def load_delta(self, delta_id: str) -> ChannelState:
        """A stored delta (validated against the model)."""
        _check_id(delta_id, "delta id")
        delta = self._read_channels("deltas", "delta", delta_id)
        self._check_against_model(delta, full=False, what=f"delta {delta_id}")
        return delta

    def deltas(self) -> list[str]:
        """Ids of every stored delta, sorted."""
        return sorted(p.name[: -len(".safetensors")] for p in (self.root / "deltas").iterdir())

    def diff(self, a: str, b: str) -> dict[int, list[int]]:
        """Per layer, the channels whose stored slices differ between versions ``a`` and ``b``
        (any element ``!=``; ``-0.0 == 0.0``, and a NaN at the same place in both counts as
        changed). Layers without a changed channel are omitted."""
        diffs = channel_diff(self.load_state(a), self.load_state(b))
        return {layer: d.changed_channels() for layer, d in sorted(diffs.items()) if d.changed_channels()}

    # ------------------------------------------------------------------ deltas
    def add_delta(self, delta: ChannelState) -> str:
        """Store a delta (per layer: ``channels`` and floating ``gate``/``up``/``down`` (+ biases)
        increments, shaped like :class:`LayerChannels`); returns its content id. ``module_path``
        ``None`` is taken as the model's path. Idempotent for identical content."""
        if not isinstance(delta, ChannelState) or not delta.layers:
            raise EditError("a delta must be a non-empty ChannelState")
        fixed: dict[int, LayerChannels] = {}
        for layer, lc in delta.layers.items():
            if layer not in self._refs:
                raise EditError(f"delta layer {layer} is not a channel layer of this store")
            fixed[layer] = LayerChannels(
                channels=lc.channels,
                gate=lc.gate,
                up=lc.up,
                down=lc.down,
                gate_bias=lc.gate_bias,
                up_bias=lc.up_bias,
                module_path=lc.module_path or self._refs[layer].module_path,
            )
        state = ChannelState(fixed)
        try:
            self._check_against_model(state, full=False, what="delta")
        except StoreError as exc:
            raise EditError(str(exc)) from exc
        return self._write_channels("deltas", "delta", state)

    def delta_between(self, a: str, b: str, selection: Mapping[int, Sequence[int]] | None = None) -> str:
        """Store ``state(b) - state(a)`` on ``selection`` (default: :meth:`diff` of ``a``, ``b``),
        computed in float32 (float64 for float64 parameters); returns the delta id."""
        sa, sb = self.load_state(a), self.load_state(b)
        sel = ChannelSelection(self.diff(a, b) if selection is None else _normalise_selection(selection))
        if not sel:
            raise EditError(f"versions {a} and {b} have identical channel states")
        layers: dict[int, LayerChannels] = {}
        for layer, chans in sel.items():
            if layer not in self._refs:
                raise EditError(f"layer {layer} is not a channel layer of this store")
            ca, cb = sa[layer].select(chans), sb[layer].select(chans)

            def sub(x: torch.Tensor | None, y: torch.Tensor | None) -> torch.Tensor | None:
                if x is None or y is None:
                    return None
                dt = _compute_dtype(x.dtype)
                return y.to(dt) - x.to(dt)

            gate, up, down = sub(ca.gate, cb.gate), sub(ca.up, cb.up), sub(ca.down, cb.down)
            assert gate is not None and up is not None and down is not None
            layers[layer] = LayerChannels(
                channels=ca.channels.clone(),
                gate=gate,
                up=up,
                down=down,
                gate_bias=sub(ca.gate_bias, cb.gate_bias),
                up_bias=sub(ca.up_bias, cb.up_bias),
                module_path=self._refs[layer].module_path,
            )
        return self.add_delta(ChannelState(layers))

    # ------------------------------------------------------------------ init
    def init_from_model(self, note: str = "") -> Version:
        """Record the live model as the root version and make it the head. Only on an empty store.

        If the head write fails, the root record written by this call is removed again and the
        store is empty (a retry works). Returns a copy of the record."""
        if self._versions:
            hint = "" if self._head is not None else f"; its head is unset: call restore({self._root_id()!r})"
            raise StoreError(f"the store already has a root version{hint}")
        state = channel_state(self.model, self.channel_layers)
        state_id = self._write_channels("states", "state", state)
        version = Version.build(
            parent=None,
            edits=(),
            verdict=None,
            evaluation=None,
            override=None,
            fingerprint=fingerprint(self.model),
            frame_fingerprint=self._frame_fingerprint(),
            state=state_id,
            scope=None,
            note=_check_text("note", note),
            created=_now(),
        )
        self._write_version_and_head(version)
        return _fresh(version)

    def _root_id(self) -> str | None:
        roots = [v.id for v in self._versions.values() if v.parent is None]
        return roots[0] if roots else None

    def _disk_head(self) -> str | None:
        """The head named by ``head.json`` (``None`` if unset or unreadable)."""
        try:
            data = self._read_json(self.root / "head.json")
        except StoreError:
            return None
        head = data.get("head") if isinstance(data, dict) else None
        return head if isinstance(head, str) else None

    def _write_version_and_head(self, version: Version) -> None:
        """Write the record, then move the head to it. If the head write fails, the record this
        call created is dropped from memory and unlinked (best effort) before the error propagates.
        Exception: if ``head.json`` already names the new version (an interrupt after the replace),
        the record is kept and the in-memory head follows the file."""
        created = self._write_version(version)
        try:
            self._write_head(version.id)
        except BaseException:
            if self._disk_head() == version.id:
                self._head = version.id
                raise
            del self._versions[version.id]
            if created:
                # on failure: an orphan record no head points to; harmless, see the module docstring
                with contextlib.suppress(OSError):
                    self._path("versions", version.id, ".json").unlink()
            raise

    def _write_version(self, version: Version) -> bool:
        """Write ``versions/<id>.json`` and register it; returns whether this call created the file.

        Guard before writing: the record must re-parse with :meth:`Version.from_dict` exactly, its
        parent and references must be in the store and its evaluation must be consistent with the
        ledger (:meth:`_check_evaluation`), so the store never writes a record that would make it
        fail to open."""
        path = self._path("versions", version.id, ".json")
        text = json.dumps(version.to_dict(), indent=1, sort_keys=True) + "\n"
        try:
            parsed = Version.from_dict(json.loads(text))
            if _canonical(parsed.to_dict()) != _canonical(version.to_dict()):  # NaN-safe comparison
                raise StoreError(f"version {version.id} does not round-trip through JSON")
            self._check_evaluation(parsed, self._ledger, self._evaluations)
        except (EditError, StoreError) as exc:
            raise StoreError(f"refusing to write an invalid version record: {exc}") from exc
        if parsed.parent is not None and parsed.parent not in self._versions:
            raise StoreError(f"refusing to write version {parsed.id}: parent {parsed.parent} is unknown")
        for e in parsed.edits:
            ver, delta = e.references()
            if ver is not None and ver not in self._versions:
                raise StoreError(f"refusing to write version {parsed.id}: references unknown version {ver}")
            if delta is not None and delta not in self.deltas():
                raise StoreError(f"refusing to write version {parsed.id}: references unknown delta {delta}")
        if path.exists():
            if _canonical(Version.from_dict(self._read_json(path)).to_dict()) != _canonical(
                version.to_dict()
            ):
                raise StoreError(f"version {version.id} exists with different content")
            created = False
        else:
            self._atomic_write(path, text.encode())
            created = True
        self._versions[version.id] = parsed
        return created

    # ------------------------------------------------------------------ edits on the live model
    def _validate_edit(self, edit: EditRecord) -> None:
        sel = edit.channel_selection()
        outside = [layer for layer in sel.layers() if layer not in self._refs]
        if outside:
            raise EditError(
                f"edit selects layers {outside} outside the store's channel layers {self.channel_layers}"
            )
        try:
            sel.validate(self.model)
        except ChannelSelectionError as exc:
            raise EditError(str(exc)) from exc
        ver, delta_id = edit.references()
        if ver is not None:
            if ver not in self._versions:
                raise EditError(f"{edit.op.value}: reference version {ver} is not in this store")
            self.load_state(ver)
        if delta_id is not None:
            if delta_id not in self.deltas():
                raise EditError(f"apply_delta: delta {delta_id} is not in this store")
            delta = self.load_delta(delta_id)
            for layer, chans in sel.items():
                have = set(delta[layer].channels.tolist()) if layer in delta else set()
                missing = sorted(set(chans) - have)
                if missing:
                    raise EditError(f"apply_delta: delta {delta_id} has no layer {layer} channels {missing}")

    def _write_mapped(
        self,
        sel: ChannelSelection,
        fn: Callable[[int, str, torch.Tensor], torch.Tensor],
    ) -> None:
        """Write ``fn(layer, field, current)`` (computed per owned slice) into the selected channels."""
        cur = extract_channels(self.model, sel)
        target: dict[int, LayerChannels] = {}
        for layer, lc in cur.layers.items():
            vals: dict[str, torch.Tensor | None] = {}
            for name in _FIELDS:
                t = getattr(lc, name)
                vals[name] = None if t is None else fn(layer, name, t).to(t.dtype)
            gate, up, down = vals["gate"], vals["up"], vals["down"]
            assert gate is not None and up is not None and down is not None
            target[layer] = LayerChannels(
                channels=lc.channels,
                gate=gate,
                up=up,
                down=down,
                gate_bias=vals["gate_bias"],
                up_bias=vals["up_bias"],
                module_path=lc.module_path,
            )
        rollback_channels(self.model, ChannelState(target), sel)

    def _apply(self, edit: EditRecord) -> None:
        sel = edit.channel_selection()
        op, params = edit.op, edit.params
        if op is EditOp.ABLATE:
            ablate_channels(self.model, sel)
        elif op is EditOp.SCALE:
            factor = params["factor"]
            scaled = {"down", "up", "up_bias"} if params["up"] else {"down"}

            def scale(layer: int, name: str, t: torch.Tensor) -> torch.Tensor:
                if name not in scaled:
                    return t
                return t.to(_compute_dtype(t.dtype)) * factor

            self._write_mapped(sel, scale)
        elif op is EditOp.REVERT:
            rollback_channels(self.model, self.load_state(params["version"]), sel)
        elif op is EditOp.INTERPOLATE:
            ref = self.load_state(params["version"])
            lam = params["lam"]
            refs = {layer: ref[layer].select(chans) for layer, chans in sel.items()}

            def lerp(layer: int, name: str, t: torch.Tensor) -> torch.Tensor:
                r = getattr(refs[layer], name)
                if lam == 0.0:
                    return r
                if lam == 1.0:
                    return t
                dt = _compute_dtype(t.dtype)
                return torch.lerp(r.to(dt), t.to(dt), lam)

            self._write_mapped(sel, lerp)
        elif op is EditOp.APPLY_DELTA:
            delta = self.load_delta(params["delta"])
            factor = params["factor"]
            parts = {layer: delta[layer].select(chans) for layer, chans in sel.items()}

            def add(layer: int, name: str, t: torch.Tensor) -> torch.Tensor:
                d = getattr(parts[layer], name)
                dt = _compute_dtype(t.dtype)
                return t.to(dt) + factor * d.to(dt)

            self._write_mapped(sel, add)
        else:  # pragma: no cover - EditOp is closed
            raise EditError(f"unknown op {op!r}")

    # ------------------------------------------------------------------ stage / discard
    def _require_staged(self, staged: StagedEdit) -> _StagedFacts:
        if staged is not self._staged or self._facts is None:
            raise StoreError("this staged edit is not the active staged edit of this store")
        return self._facts

    def _require_fingerprint(self, expected: str, what: str) -> None:
        if fingerprint(self.model) != expected:
            raise StoreError(f"the model's weights changed outside the store ({what})")

    def stage(self, edit: EditRecord | Sequence[EditRecord]) -> StagedEdit:
        """Apply one edit (or several, in order) to the live model on top of the head version.

        Requires an initialised store, no other active staged edit and ``fingerprint(model) ==
        head.fingerprint``. Every edit is validated (layers inside the store's channel layers,
        channels exist, referenced versions / deltas exist and fit the model) before the first
        write; if applying any edit fails anyway, the parent's weights are restored bitwise
        before the exception propagates. Each edit is first replaced by :meth:`EditRecord.copy`
        (re-validated: a record mutated into an invalid value raises :class:`EditError` before
        any write); the store applies and later commits only those copies, so changing the
        caller's records after ``stage`` has no effect. Returns the :class:`StagedEdit`.
        """
        given = (edit,) if isinstance(edit, EditRecord) else tuple(edit)
        if not given or not all(isinstance(e, EditRecord) for e in given):
            raise EditError("stage needs one EditRecord or a non-empty sequence of them")
        edits = tuple(e.copy() for e in given)
        if self._staged is not None:
            raise StoreError("another edit is staged; commit or discard it first")
        if self._head is None:
            if self._versions:
                raise StoreError(
                    f"the store's head is unset (an interrupted write); call restore({self._root_id()!r}) "
                    "or restore another version first"
                )
            raise StoreError("the store has no version; call init_from_model first")
        parent = self._versions[self._head]
        self._require_fingerprint(parent.fingerprint, "the model is not at the head version; restore first")
        for e in edits:
            self._validate_edit(e)
        selection = ChannelSelection()
        for e in edits:
            selection = selection.union(e.channel_selection())
        names = self._ffn_names(selection.layers())
        name_set = set(names)
        others = [n for n, _ in self.model.named_parameters() if n not in name_set]
        incumbent = capture(self.model, names)
        outside = fingerprint(self.model, others)
        try:
            for e in edits:
                self._apply(e)
        except BaseException:
            restore_snapshot(self.model, incumbent)
            raise
        after = fingerprint(self.model)
        self._facts = _StagedFacts(
            parent=parent.id,
            edits_json=_canonical([e.to_dict() for e in edits]),
            selection=selection,
            outside_fingerprint=outside,
            fingerprint=after,
        )
        staged = StagedEdit(
            parent=parent.id,
            edits=tuple(e.copy() for e in edits),
            selection=selection,
            incumbent=incumbent,
            outside_fingerprint=outside,
            fingerprint=after,
        )
        self._staged = staged
        return staged

    def discard(self, staged: StagedEdit) -> None:
        """Undo a staged edit: the affected parameters are restored bitwise to the parent's."""
        facts = self._require_staged(staged)
        restore_snapshot(self.model, staged.incumbent)
        staged.status = "discarded"
        self._staged = None
        self._facts = None
        self._require_fingerprint(
            self._versions[facts.parent].fingerprint, "parameters outside the edit changed while staged"
        )

    # ------------------------------------------------------------------ evaluate / commit
    def evaluate(self, staged: StagedEdit, gate_config: GateConfig, probe: CapabilityProbe) -> GateVerdict:
        """Gate verdict of the staged weights (candidate) against the parent's (incumbent).

        ``probe.evidence(model, staged.incumbent, decision_index=evaluation_count,
        namespace=gate_namespace)`` scores both on the same fresh items of this store's own stream
        (the incumbent through ``weights.swapped``), then
        ``CapabilityGate(gate_config, copy of the store ledger).verdict`` decides; the advanced
        ledger and counter are written to ``ledger.json`` first and adopted in memory only after
        that write succeeded (an error before that write leaves the ledger, the counter and the
        private verdict unchanged; ``ledger.count == evaluation_count`` always holds; the final
        fingerprint check raises after the ledger was advanced). ``gate_config.enabled`` must be
        True and its ``alpha`` / ``alpha_schedule`` must match the store's ledger once one exists
        (``ValueError``). ``require_transfer_gain`` is not consulted: a commit without override
        always needs ``transfer_ok`` and ``retention_ok``. Refused (:class:`StoreError`) if the
        model no longer holds the staged weights, before and after scoring.
        """
        facts = self._require_staged(staged)
        if not isinstance(gate_config, GateConfig):
            raise TypeError(f"gate_config must be a GateConfig, got {type(gate_config).__name__}")
        if not gate_config.enabled:
            raise ValueError("an edit evaluation needs gate_config.enabled=True (use an override otherwise)")
        if self._ledger is not None and (
            self._ledger.total_alpha != gate_config.alpha
            or self._ledger.schedule != gate_config.alpha_schedule
        ):
            raise ValueError(
                f"the store's ledger is pinned to alpha={self._ledger.total_alpha}, "
                f"schedule={self._ledger.schedule!r}; got alpha={gate_config.alpha}, "
                f"schedule={gate_config.alpha_schedule!r}"
            )
        self._require_fingerprint(facts.fingerprint, "the staged weights were modified")
        ledger = self.ledger if self._ledger is not None else gate_config.make_ledger()  # a copy
        assert ledger is not None
        index = self._evaluations
        evidence = probe.evidence(self.model, staged.incumbent, index, namespace=self._gate_namespace)
        verdict = CapabilityGate(gate_config, ledger).verdict(evidence)
        to_dict = getattr(probe, "to_dict", None)
        context = {
            "evaluation_index": index,
            "gate_namespace": self._gate_namespace,
            "gate_config": gate_config.to_dict(),
            "probe": to_dict() if callable(to_dict) else None,
        }
        context_json = _canonical(context)
        self._write_ledger(ledger, index + 1)
        self._ledger = ledger
        self._evaluations = index + 1
        self._require_fingerprint(facts.fingerprint, "the probe modified the weights")
        facts.verdict = verdict
        facts.evaluation_json = context_json
        staged.verdicts.append(verdict)
        staged.evaluations.append(json.loads(context_json))
        return verdict

    def commit(self, staged: StagedEdit, *, override: str | None = None, note: str = "") -> Version:
        """Record the staged edit as a new version (child of its parent) and make it the head.

        Refused with :class:`CommitRefusedError` (the edit stays staged) unless the **last**
        evaluation of this staged edit passed (``transfer_ok`` and ``retention_ok``, no alarm) or
        ``override`` is a non-empty reason (``""`` / whitespace raise ``ValueError``), and unless
        the scope check passes: :func:`verify_modification_scope` of the parent's values of the
        affected FFN parameters against the current ones, allowed = the edits' channels, plus an
        unchanged fingerprint of every other parameter. The report is stored in the version.
        Writes, in order: the channel-state file, the version record, the head pointer; if the
        head write fails, the record is removed again and the edit stays staged (a retry does not
        leave an orphan sibling). The recorded edits, verdict and evaluation context are the
        store's private copies taken at :meth:`stage` / :meth:`evaluate` (see :class:`StagedEdit`).
        Returns a copy of the new record.
        """
        facts = self._require_staged(staged)
        if override is not None:
            if not isinstance(override, str) or not override.strip():
                raise ValueError("override must be a non-empty reason (or None)")
            _check_text("override", override)
        note = _check_text("note", note)
        if not facts.passed and override is None:
            last = facts.verdict
            why = (
                "it was never evaluated"
                if last is None
                else (
                    f"its last verdict did not pass (transfer_ok={last.transfer_ok}, "
                    f"retention_ok={last.retention_ok}, alarm={last.alarm})"
                )
            )
            raise CommitRefusedError(f"commit refused: {why}; give an override reason to commit anyway")
        self._require_fingerprint(facts.fingerprint, "the staged weights were modified")
        if self._head != facts.parent:
            raise StoreError("the head moved since the edit was staged")
        names = list(staged.incumbent.params)
        name_set = set(names)
        others = [n for n, _ in self.model.named_parameters() if n not in name_set]
        report = verify_modification_scope(
            staged.incumbent.params, snapshot(self.model, names), facts.selection, self.model
        )
        outside_ok = fingerprint(self.model, others) == facts.outside_fingerprint
        scope = _scope_dict(report, outside_ok)
        if not scope["ok"]:
            raise CommitRefusedError(
                f"commit refused: modification outside the edit's scope: {report.summary()}"
            )
        parent = self._versions[facts.parent]
        state_id = self._write_channels("states", "state", channel_state(self.model, self.channel_layers))
        last = facts.verdict
        version = Version.build(
            parent=parent.id,
            edits=facts.edits(),
            verdict=None if last is None else json.loads(_canonical(last.to_dict())),
            evaluation=None if facts.evaluation_json is None else json.loads(facts.evaluation_json),
            override=override,
            fingerprint=facts.fingerprint,
            frame_fingerprint=parent.frame_fingerprint,
            state=state_id,
            scope=scope,
            note=note,
            created=_now(),
        )
        self._write_version_and_head(version)
        staged.status = "committed"
        self._staged = None
        self._facts = None
        return _fresh(version)

    # ------------------------------------------------------------------ restore
    def restore(self, version_id: str) -> Version:
        """Put the live model at ``version_id`` bitwise and make it the head.

        Refused (:class:`StoreError`, nothing written) while an edit is staged, for an unknown or
        malformed id, when the state file fails its checks, and when the model would not end up
        with the version's fingerprint (a tampered record or state, or a live frame that differs
        from the version's: the store cannot restore parameters outside its channel layers).
        """
        if self._staged is not None:
            raise StoreError("an edit is staged; commit or discard it first")
        version = self._version(version_id)
        state = self.load_state(version_id)
        if self._expected_fingerprint(state) != version.fingerprint:
            frame = self._frame_fingerprint()
            why = (
                "the live parameters outside the channel layers differ from the version's"
                if frame != version.frame_fingerprint
                else "the stored state does not reproduce the recorded fingerprint"
            )
            raise StoreError(f"cannot restore {version_id}: {why}")
        rollback_channels(self.model, state)
        self._require_fingerprint(version.fingerprint, "restore did not reproduce the fingerprint")
        self._write_head(version.id)
        return _fresh(version)
