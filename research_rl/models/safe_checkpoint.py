"""Safetensors-only check of a local Hugging Face model directory (no weights are read).

``AutoModelForCausalLM.from_pretrained(dir, use_safetensors=True)`` opens ``model.safetensors``
if it exists, else ``model.safetensors.index.json`` and **every shard its ``weight_map`` names,
whatever the shard's file name**: an index can point at a pickle file (a suffixless name, or a
path outside the directory) that ``transformers`` then reads with ``torch.load``. A suffix
blocklist over the directory cannot see either case. :func:`check_safetensors_dir` therefore
checks exactly the files the loader opens, as an allowlist:

* ``config.json`` must exist and be a JSON object that does not set ``transformers_weights``
  (which overrides the weight-file selection and may name a pickle file);
* the directory must not be a PEFT adapter (``adapter_config.json``: the load would be redirected
  to its base model, a directory not checked here);
* ``model.safetensors`` must exist, or ``model.safetensors.index.json`` whose non-empty
  ``weight_map`` lists only bare ``*.safetensors`` file names present in the directory.

Callers: :func:`research_rl.ui.loading.check_model_dir` (UI ``--model-dir``) and
``scripts/gpu/capability_p1.py`` (``validate_model_dir``). Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PICKLE_SUFFIXES",
    "SAFE_WEIGHTS_FILE",
    "SAFE_WEIGHTS_INDEX_FILE",
    "SafetensorsDir",
    "UnsafeCheckpointError",
    "check_safetensors_dir",
    "is_plain_file_name",
]

SAFE_WEIGHTS_FILE = "model.safetensors"
"""Single-file safetensors weights: the first file ``transformers`` opens with ``use_safetensors``."""
SAFE_WEIGHTS_INDEX_FILE = "model.safetensors.index.json"
"""Sharded safetensors index: opened when :data:`SAFE_WEIGHTS_FILE` is absent."""
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
"""Pickle-based checkpoint formats: never loaded (they can execute code on load)."""


class UnsafeCheckpointError(ValueError):
    """The directory is not a checkpoint whose weights ``transformers`` would read from safetensors only."""


@dataclass(frozen=True)
class SafetensorsDir:
    """Result of :func:`check_safetensors_dir`.

    ``safetensors``: the weight files the loader will open (bare names, sorted for an index).
    ``ignored_pickle_files``: top-level files with a :data:`PICKLE_SUFFIXES` suffix; the loader never
    opens them (callers may still refuse them).
    """

    path: Path
    safetensors: tuple[str, ...]
    ignored_pickle_files: tuple[str, ...]


def is_plain_file_name(name: str) -> bool:
    """``True`` for a bare file name inside the directory (no separator, ``..`` or absolute path).

    The check is lexical on purpose: shards of a Hugging Face cache snapshot are symlinks into
    ``../../blobs``, which a ``resolve()``-based containment check would wrongly refuse."""
    return bool(name) and name not in (".", "..") and "/" not in name and "\\" not in name


def _index_shards(path: Path, label: str) -> list[str]:
    where = f"{label}: {SAFE_WEIGHTS_INDEX_FILE}"
    try:
        index = json.loads((path / SAFE_WEIGHTS_INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise UnsafeCheckpointError(f"{where} is not readable JSON: {e}") from e
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise UnsafeCheckpointError(f"{where} has no non-empty weight_map")
    values = list(weight_map.values())
    not_names = sorted({repr(v) for v in values if not isinstance(v, str) or not is_plain_file_name(v)})
    if not_names:
        raise UnsafeCheckpointError(
            f"{where} lists shards that are not bare file names in the directory: {not_names[:5]}"
        )
    shards = sorted(set(values))
    not_safetensors = [s for s in shards if not s.endswith(".safetensors")]
    if not_safetensors:
        raise UnsafeCheckpointError(
            f"{where} lists non-safetensors shards {not_safetensors[:5]} (pickle weights are never loaded); "
            "convert the checkpoint to safetensors first"
        )
    missing = [s for s in shards if not (path / s).is_file()]
    if missing:
        raise UnsafeCheckpointError(f"{where} lists missing shards {missing[:5]}")
    return shards


def check_safetensors_dir(path: str | Path, *, label: str | None = None) -> SafetensorsDir:
    """Check ``path`` without reading any weights (see the module docstring for the rules).

    ``label`` prefixes every message (default ``"model dir '<path>'"``). Raises
    :class:`UnsafeCheckpointError` when ``path`` is not a directory, ``config.json`` is missing, is
    not a JSON object or sets ``transformers_weights``, the directory holds ``adapter_config.json``,
    neither :data:`SAFE_WEIGHTS_FILE` nor :data:`SAFE_WEIGHTS_INDEX_FILE` exists, or the index is
    malformed or lists a shard that is not a bare ``*.safetensors`` file name present in the directory.
    """
    root = Path(path)
    name = label if label is not None else f"model dir {str(path)!r}"
    if not root.is_dir():
        raise UnsafeCheckpointError(
            f"{name} is not an existing local directory (hub ids are not accepted; nothing is downloaded)"
        )
    if not (root / "config.json").is_file():
        raise UnsafeCheckpointError(f"{name} has no config.json")
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise UnsafeCheckpointError(f"{name}: config.json is not readable JSON: {e}") from e
    if not isinstance(config, dict):
        raise UnsafeCheckpointError(f"{name}: config.json is not a JSON object")
    if config.get("transformers_weights") is not None:
        raise UnsafeCheckpointError(
            f"{name}: config.json sets transformers_weights="
            f"{config['transformers_weights']!r}, which overrides the weight-file selection (it may name "
            f"a pickle file); remove it and use {SAFE_WEIGHTS_FILE} or {SAFE_WEIGHTS_INDEX_FILE}"
        )
    if (root / "adapter_config.json").is_file():
        raise UnsafeCheckpointError(
            f"{name} is a PEFT adapter directory (adapter_config.json): the load would "
            "be redirected to its base model; merge the adapter into a full safetensors checkpoint first"
        )
    pickles = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in PICKLE_SUFFIXES)
    if (root / SAFE_WEIGHTS_FILE).is_file():
        tensors = [SAFE_WEIGHTS_FILE]
    elif (root / SAFE_WEIGHTS_INDEX_FILE).is_file():
        tensors = _index_shards(root, name)
    else:
        others = sorted(p.name for p in root.glob("*.safetensors") if p.is_file())
        found = [
            f"pickle-format files {pickles}, which are never loaded" if pickles else "",
            f"non-standard safetensors files {others}, which transformers does not load" if others else "",
        ]
        detail = "; ".join(f for f in found if f)
        raise UnsafeCheckpointError(
            f"{name} has no *.safetensors weights under a standard name "
            f"({SAFE_WEIGHTS_FILE} or {SAFE_WEIGHTS_INDEX_FILE})"
            + (f" (found {detail})" if detail else "")
            + "; convert or rename the checkpoint to safetensors first"
        )
    return SafetensorsDir(path=root, safetensors=tuple(tensors), ignored_pickle_files=tuple(pickles))
