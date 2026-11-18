"""Model loading for the UI: local safetensors checkpoints or an in-process tiny demo model.

:func:`load_model_dir` loads a Hugging Face model directory **from local files only** and **from
safetensors only**:

* the directory must exist, contain ``config.json`` and at least one ``*.safetensors`` file;
* any file that could hold a pickle (``*.bin``, ``*.pt``, ``*.pth``, ``*.pkl``, ``*.pickle``,
  ``*.ckpt``, ``*.joblib``, ``*.npy``/``*.npz`` with objects, ...) anywhere in the directory makes
  the load **refused** before ``from_pretrained`` is called (refusing is safer than hoping the
  loader ignores it);
* ``from_pretrained`` is called with ``local_files_only=True``, ``use_safetensors=True`` and
  ``trust_remote_code=False`` (no network, no pickle, no code from the checkpoint).

:func:`build_tiny_demo` builds a randomly initialised tiny Llama with the character tokenizer of
:mod:`research_rl.models.tiny` in-process. Its numbers are those of a random tiny model on this
machine and say nothing about real large models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama

__all__ = [
    "UNSAFE_SUFFIXES",
    "LoadedModel",
    "ModelLoadError",
    "build_tiny_demo",
    "check_model_dir",
    "load_model_dir",
]

UNSAFE_SUFFIXES = frozenset(
    {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib", ".npy", ".npz", ".h5", ".msgpack"}
)
"""File suffixes refused in a model directory (pickle-capable or non-safetensors weight formats)."""


class ModelLoadError(ValueError):
    """The model directory is missing, incomplete, or contains a non-safetensors weight file."""


@dataclass
class LoadedModel:
    """A model with its tokenizer and a short description of where it came from."""

    model: nn.Module
    tokenizer: Any
    source: str
    tiny_demo: bool


def check_model_dir(path: str | Path) -> list[Path]:
    """Validate a model directory and return its ``*.safetensors`` files (sorted).

    Raises :class:`ModelLoadError` if the path is not a directory, has no ``config.json``, has no
    ``*.safetensors`` file, or contains any file with a suffix in :data:`UNSAFE_SUFFIXES`
    (recursively; symlinks are followed only for this name check).
    """
    root = Path(path)
    if not root.is_dir():
        raise ModelLoadError(f"model dir {root} is not a directory")
    if not (root / "config.json").is_file():
        raise ModelLoadError(f"model dir {root} has no config.json")
    unsafe = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in UNSAFE_SUFFIXES)
    if unsafe:
        raise ModelLoadError(
            f"model dir {root} contains non-safetensors weight / pickle files {unsafe[:5]}; "
            "only *.safetensors weights are loaded (convert the checkpoint and remove these files)"
        )
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ModelLoadError(f"model dir {root} has no *.safetensors weight file")
    return weights


def load_model_dir(path: str | Path) -> LoadedModel:
    """Load a causal LM and its tokenizer from a local directory (safetensors only, offline).

    See the module docstring for the checks. Every refusal or loader failure (including a corrupt
    ``*.safetensors`` file) raises :class:`ModelLoadError`. The model is put in eval mode on the CPU
    (a GPU placement is not done here; NOT_TESTED).
    """
    root = Path(path)
    check_model_dir(root)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(root), local_files_only=True, use_safetensors=True, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    except Exception as exc:  # any loader failure (SafetensorError, shape RuntimeError, ...) is a refusal
        raise ModelLoadError(f"cannot load model dir {root}: {type(exc).__name__}: {exc}") from exc
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, source=f"model dir {root.name}", tiny_demo=False)


def build_tiny_demo(seed: int = 0) -> LoadedModel:
    """A randomly initialised tiny Llama (``research_rl.models.tiny``) and its char tokenizer."""
    tokenizer = build_char_tokenizer()
    model = build_tiny_llama(len(tokenizer), seed=seed)
    model.eval()
    return LoadedModel(
        model=model, tokenizer=tokenizer, source=f"tiny demo (random tiny Llama, seed {seed})", tiny_demo=True
    )
