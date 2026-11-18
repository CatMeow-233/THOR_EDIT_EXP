#!/usr/bin/env python
"""P1 capability pilot: the P0 pipeline on a LOCAL 7-8B checkpoint on one GPU.

**Script status: NOT_TESTED.** This script has never been run on a GPU or on a 7-8B checkpoint
by its authors. Only ``--dry-run`` (a tiny in-process Llama on CPU, outputs labelled
``DRY_RUN_CPU``) is exercised by the test suite; a dry run is a plumbing check and says nothing
about 7-8B models. GPU numbers are written only when they were measured on the CUDA device of the
run (``measured_on`` = ``torch.cuda.get_device_name``); everything else GPU-related is
``NOT_TESTED``. Memory figures printed before the run are *computed* from tensor shapes, not
measured.

Pipeline (``docs/research_rl/CAPABILITY.md`` 1.14 and the kill criteria of section 3); every step
reuses :mod:`research_rl.capability.pilot`:

1. **Model.** ``--model-dir`` must be an existing local directory with ``config.json`` and
   ``model.safetensors`` or ``model.safetensors.index.json`` whose shards are all ``*.safetensors``
   files in the directory (:func:`validate_model_dir`); pickle formats (``.bin``, ``.pt``,
   ``.pth``, ``.pkl``, ``.ckpt``) are never loaded (a directory with pickle weights only, an index
   pointing at pickle shards, a ``transformers_weights`` override in ``config.json`` and a PEFT
   adapter directory are refused). ``HF_HUB_OFFLINE=1`` and
   ``TRANSFORMERS_OFFLINE=1`` are set before ``transformers`` is imported and every load passes
   ``local_files_only=True``, ``use_safetensors=True`` and ``trust_remote_code=False``: nothing is
   ever downloaded and no repository code is executed. Weights are loaded in bf16 on the GPU.
2. **Tasks.** The same synthetic operator-composition splits as P0
   (:func:`~research_rl.capability.pilot.build_splits`), rendered with the model's own tokenizer.
   :func:`check_tokenizer` verifies that every character of the task alphabet and every sampled
   prompt and answer round-trips through encode/decode and sets ``max_new_tokens`` to at least the
   longest answer in tokens + 1 (and at least the character contract ``max_len + 1``).
3. **Low start.** SFT on the P0 atomic-skill data (:func:`~research_rl.capability.tasks.sft_examples`)
   through a :class:`~research_rl.optim.optimizer_wrapper.HookedOptimizer` (AdamW, fp32 master
   weights) over a parameter subset (``--sft-scope``/``--sft-layers``); ``--sft-steps 0`` skips it.
   The SFT-trained tensors are saved to ``start_delta.safetensors``. The start map is measured with
   :class:`~research_rl.capability.probe.StartMap`.
4. **Channels.** :func:`~research_rl.capability.pilot.select_informed_channels` (forward-only),
   optionally restricted to ``--channel-layers`` (memory), and the density-matched random control.
5. **Arms** (P0 arm keys, sequential, one GPU): ``grpo`` = :class:`TorchGRPOOperator` with a
   ``HookedOptimizer(AdamW, master_weights=True)`` and a :class:`~research_rl.optim.update_interface.MaskRule`
   restricting the update to the informed channel subset
   (:func:`~research_rl.models.ffn_channels.setup_channel_training`); ``es_informed`` /
   ``es_random`` / ``es_full`` = :class:`~research_rl.capability.es.ESOperator` over
   :class:`~research_rl.capability.spaces.ChannelGainSpace` (``es_full`` = every channel). Every arm
   runs behind :class:`~research_rl.capability.loop.GatedLoop` with the pilot's oracle, tracker and
   alarm; between arms the start weights are restored and checked bit-exactly.
6. **Outputs** in ``--output-dir``: ``results.json``, ``summary.md``, ``memory_plan.json``,
   ``p1_config.json``, ``start_delta.safetensors`` (when SFT ran) and
   ``decisions/<arm>.<label>.jsonl``. Every output carries the label (``DRY_RUN_CPU`` or
   ``P1_GPU``) and the script status ``NOT_TESTED``.

Refusals: exit code 3 (``"status": "REFUSED"``) for a missing or
invalid ``--model-dir``, a tokenizer that does not round-trip the tasks or a computed memory plan
over budget; exit code 2 (``"status": "NOT_TESTED"``) when CUDA is unavailable and ``--dry-run``
was not given. Every check that does not need the trained start model runs before SFT and before
the output directory is written: model directory, CUDA, configuration, layer ranges, tokenizer,
retention horizon and a memory upper bound (the GRPO / ES phases bounded by every channel of
``--channel-layers``). Only the exact memory plan after channel selection can refuse later.

Examples::

    python scripts/gpu/capability_p1.py --dry-run --output-dir runs/p1_dry
    python scripts/gpu/capability_p1.py --model-dir /models/llama-3.1-8b --output-dir runs/p1 \\
        --channel-layers 20-27 --sft-layers 20-27 --arms grpo+gate_on,es_informed+gate_on --seeds 0
"""

from __future__ import annotations

import os

# Offline before anything imports transformers / huggingface_hub (research_rl imports them).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import dataclasses
import json
import math
import platform
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from research_rl.capability.alarm import DisagreementAlarm
from research_rl.capability.gate import CapabilityGate
from research_rl.capability.loop import GatedLoop
from research_rl.capability.operators import TorchGRPOOperator
from research_rl.capability.pilot import (
    BANNER,
    GATES,
    NOT_TESTED,
    OPERATORS,
    ORACLE_PURPOSE,
    ArmSpec,
    PilotConfig,
    _arm_result,
    _ArmMonitor,
    _budget,
    _CountingScoreFn,
    _encode_example,
    _finite,
    _make_operator,
    _PilotController,
    _RecordingOperator,
    _report_counts,
    _rule_slices,
    _sampler_root,
    _seed,
    _selection_from,
    _slices,
    _torch_threads,
    _verifiers,
    arm_specs,
    build_model,
    build_splits,
    choose_retention_items,
    evaluate_criteria,
    excluded_uids,
    gate_exercise,
    oracle_exclusion_steps,
    parse_arm,
    render_summary,
    select_informed_channels,
    smoke_config,
)
from research_rl.capability.probe import TRACK_POOL, CapabilityProbe, StartMap
from research_rl.capability.tasks import (
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    VerifierScoreFn,
    sft_examples,
)
from research_rl.capability.tracker import CapabilityTracker
from research_rl.capability.types import Item, OperatorReport, Split, UpdateOperator
from research_rl.capability.weights import StateSnapshot, capture, restore
from research_rl.models.ffn_channels import ChannelSelection, setup_channel_training
from research_rl.models.model_interface import count_parameters, find_swiglu_ffns, fingerprint
from research_rl.models.tiny import build_char_tokenizer
from research_rl.optim import HookedOptimizer

SCRIPT_STATUS = NOT_TESTED
SCRIPT_NOTE = (
    "NOT_TESTED: scripts/gpu/capability_p1.py has never been run on a GPU or on a 7-8B checkpoint by its "
    "authors; only --dry-run (tiny in-process model on CPU) is exercised by the test suite."
)
LABEL_DRY_RUN = "DRY_RUN_CPU"
LABEL_GPU = "P1_GPU"
P1_BANNERS = {
    LABEL_DRY_RUN: (
        "DRY_RUN_CPU: P1 plumbing check on a tiny in-process Llama on CPU. Not a P1 result, not evidence "
        "about 7-8B models, no GPU numbers."
    ),
    LABEL_GPU: (
        "P1 on a local checkpoint on one GPU. Script status NOT_TESTED (never run on a GPU by its authors): "
        "check every number before trusting it."
    ),
}
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
"""Pickle-based checkpoint formats: never loaded (they can execute code on load)."""
TOKENIZER_CHECK_PURPOSE = "p1_tokenizer_check"
"""``ItemSampler`` purpose of the tokenizer round-trip sample (index 0; no other stream uses it)."""
EXIT_REFUSED = 3
EXIT_NOT_TESTED = 2
P1_GPU_DEFAULTS: dict[str, Any] = {
    "es_space": "gain",
    "es_full_space": "all_channels",
    "jobs": 1,
    "threads_per_arm": 1,
    "eta_probe": False,
    "sft_steps": 2000,
    "sft_batch_size": 16,
    "sft_lr": 1e-4,
    "grpo_lr": 1e-5,
    "grpo_prompts_per_update": 8,
    "eval_batch_size": 64,
}
"""PilotConfig overrides of a ``--model-dir`` run: NOT_TESTED starting values, not tuned on any GPU."""
_RESULTS_VERSION = 1


class P1Refusal(Exception):
    """The script refuses to run: ``status`` (``REFUSED`` / ``NOT_TESTED``), ``reason``, ``exit_code``."""

    def __init__(self, reason: str, *, status: str = "REFUSED", exit_code: int = EXIT_REFUSED) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.exit_code = exit_code


# ============================================================================== options
@dataclass
class P1Options:
    """P1-only knobs (everything else is a :class:`~research_rl.capability.pilot.PilotConfig` field).

    * ``model_dir``: local checkpoint directory (required unless ``dry_run``).
    * ``dry_run``: tiny in-process Llama on CPU, outputs labelled ``DRY_RUN_CPU``.
    * ``device``: CUDA device of a real run.
    * ``channel_layers``: restrict the informed selection (and therefore the GRPO optimizer, which
      holds the whole FFN tensors of every touched layer) to these decoder layers; ``None`` = all.
    * ``sft_scope``: ``"ffn"`` (gate/up/down of ``sft_layers``) or ``"all"`` (every parameter).
    * ``sft_layers``: layers of an ``"ffn"`` SFT; ``None`` = ``channel_layers``, else every layer.
    * ``memory_fraction``: share of the device's total memory the computed plan may use.
    * ``allow_over_budget``: run (with a warning) although the computed plan exceeds the budget.
    * ``tokenizer_check_items``: items per non-AUDIT slice in the tokenizer round-trip check.
    """

    model_dir: str | None = None
    dry_run: bool = False
    device: str = "cuda:0"
    channel_layers: tuple[int, ...] | None = None
    sft_scope: str = "ffn"
    sft_layers: tuple[int, ...] | None = None
    memory_fraction: float = 0.9
    allow_over_budget: bool = False
    tokenizer_check_items: int = 64

    def __post_init__(self) -> None:
        if self.sft_scope not in ("ffn", "all"):
            raise P1Refusal(f"sft_scope must be 'ffn' or 'all', got {self.sft_scope!r}")
        if not 0.0 < self.memory_fraction <= 1.0:
            raise P1Refusal(f"memory_fraction must be in (0, 1], got {self.memory_fraction}")
        if self.tokenizer_check_items < 1:
            raise P1Refusal(f"tokenizer_check_items must be >= 1, got {self.tokenizer_check_items}")
        for name in ("channel_layers", "sft_layers"):
            value = getattr(self, name)
            if value is not None:
                if not value:
                    raise P1Refusal(f"{name} must be None or a non-empty list of layers")
                setattr(self, name, tuple(sorted({int(x) for x in value})))

    @property
    def label(self) -> str:
        return LABEL_DRY_RUN if self.dry_run else LABEL_GPU

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        for key in ("channel_layers", "sft_layers"):
            if out[key] is not None:
                out[key] = list(out[key])
        return out


def parse_layers(spec: str) -> tuple[int, ...]:
    """``"20-23,30"`` -> ``(20, 21, 22, 23, 30)`` (``ValueError`` for malformed parts)."""
    out: list[int] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad layer range {part!r}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out or any(x < 0 for x in out):
        raise ValueError(f"bad layer specification {spec!r}")
    return tuple(sorted(set(out)))


def parse_arms(text: str) -> tuple[str, ...]:
    """``grpo,es_informed+gate_on`` -> ``operator+gate`` strings (a bare operator means both gates)."""
    out: list[str] = []
    for token in (t.strip() for t in text.split(",")):
        if not token:
            continue
        if "+" in token:
            parse_arm(token)
            out.append(token)
        elif token in OPERATORS:
            out.extend(f"{token}+{g}" for g in GATES)
        else:
            raise ValueError(f"unknown arm {token!r}; operators: {OPERATORS}, gates: {GATES}")
    return tuple(dict.fromkeys(out))


def check_p1_config(config: PilotConfig) -> None:
    """P1 constraints on a :class:`PilotConfig`: ES over :class:`ChannelGainSpace` only (``es_space
    == "gain"``, ``es_full_space == "all_channels"``; a full FFN-weight space does not fit one GPU
    for a 7-8B model) and sequential arms (``jobs == 1``: one model on one GPU)."""
    if config.es_space != "gain":
        raise P1Refusal(f"P1 runs ES over ChannelGainSpace only (es_space='gain'), got {config.es_space!r}")
    if config.es_full_space != "all_channels":
        raise P1Refusal(
            f"P1 es_full uses ChannelGainSpace over every channel (es_full_space='all_channels'), "
            f"got {config.es_full_space!r}"
        )
    if config.jobs != 1:
        raise P1Refusal(f"P1 runs arms sequentially on one model (jobs=1), got jobs={config.jobs}")


# ============================================================================== model directory
SAFE_WEIGHTS_FILE = "model.safetensors"
"""Single-file safetensors weights: the first file ``transformers`` opens with ``use_safetensors``."""
SAFE_WEIGHTS_INDEX_FILE = "model.safetensors.index.json"
"""Sharded safetensors index: opened when :data:`SAFE_WEIGHTS_FILE` is absent."""


def _is_plain_file_name(name: str) -> bool:
    """``True`` for a bare file name inside the directory (no separator, ``..`` or absolute path).

    The check is lexical on purpose: shards of a Hugging Face cache snapshot are symlinks into
    ``../../blobs``, which a ``resolve()``-based containment check would wrongly refuse."""
    return bool(name) and name not in (".", "..") and "/" not in name and "\\" not in name


def _index_shards(path: Path, model_dir: str) -> list[str]:
    """Shard file names of ``path / SAFE_WEIGHTS_INDEX_FILE``; :class:`P1Refusal` for a malformed
    index, an empty or non-dict ``weight_map``, or any shard that is not a bare ``*.safetensors``
    file name present in the directory (an index can point at pickle shards, which ``transformers``
    would then ``torch.load``)."""
    where = f"--model-dir {model_dir!r}: {SAFE_WEIGHTS_INDEX_FILE}"
    try:
        index = json.loads((path / SAFE_WEIGHTS_INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise P1Refusal(f"{where} is not readable JSON: {e}") from e
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise P1Refusal(f"{where} has no non-empty weight_map")
    values = list(weight_map.values())
    not_names = sorted({repr(v) for v in values if not isinstance(v, str) or not _is_plain_file_name(v)})
    if not_names:
        raise P1Refusal(
            f"{where} lists shards that are not bare file names in the directory: {not_names[:5]}"
        )
    shards = sorted(set(values))
    not_safetensors = [s for s in shards if not s.endswith(".safetensors")]
    if not_safetensors:
        raise P1Refusal(
            f"{where} lists non-safetensors shards {not_safetensors[:5]} (pickle weights are never loaded); "
            "convert the checkpoint to safetensors first"
        )
    missing = [s for s in shards if not (path / s).is_file()]
    if missing:
        raise P1Refusal(f"{where} lists missing shards {missing[:5]}")
    return shards


def validate_model_dir(model_dir: str | None) -> dict[str, Any]:
    """Check a local checkpoint directory without loading any weights.

    The check covers exactly the files ``transformers`` opens with ``use_safetensors=True``:
    ``model.safetensors`` if present, else ``model.safetensors.index.json`` and every shard it
    lists. Refuses (:class:`P1Refusal`) when ``model_dir`` is missing or not an existing directory;
    when ``config.json`` is missing, is not a JSON object or sets ``transformers_weights`` (which
    overrides the file selection and may name a pickle file); when the directory is a PEFT adapter
    (``adapter_config.json``: the load would be redirected to its base model, a directory not
    checked here); when neither of the two standard safetensors files exists (a directory with only
    pickle weights such as ``pytorch_model.bin``, or only a non-standard name such as
    ``consolidated.safetensors``); or when the index is malformed or lists a shard that is not a
    bare ``*.safetensors`` file name present in the directory. Pickle files in the directory are
    reported in ``ignored_pickle_files``; since only the files above are opened, they are never
    read. ``safetensors`` lists the weight files that will be loaded.
    """
    if not model_dir:
        raise P1Refusal(
            "--model-dir is required (a local directory with config.json and model.safetensors or "
            "model.safetensors.index.json); this script never downloads a model. Use --dry-run for the "
            "tiny CPU plumbing check."
        )
    path = Path(model_dir)
    if not path.is_dir():
        raise P1Refusal(
            f"--model-dir {model_dir!r} is not an existing local directory (hub ids are not accepted; "
            "nothing is downloaded)"
        )
    if not (path / "config.json").is_file():
        raise P1Refusal(f"--model-dir {model_dir!r} has no config.json")
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise P1Refusal(f"--model-dir {model_dir!r}: config.json is not readable JSON: {e}") from e
    if not isinstance(config, dict):
        raise P1Refusal(f"--model-dir {model_dir!r}: config.json is not a JSON object")
    if config.get("transformers_weights") is not None:
        raise P1Refusal(
            f"--model-dir {model_dir!r}: config.json sets transformers_weights="
            f"{config['transformers_weights']!r}, which overrides the weight-file selection (it may name "
            f"a pickle file); remove it and use {SAFE_WEIGHTS_FILE} or {SAFE_WEIGHTS_INDEX_FILE}"
        )
    if (path / "adapter_config.json").is_file():
        raise P1Refusal(
            f"--model-dir {model_dir!r} is a PEFT adapter directory (adapter_config.json): the load would "
            "be redirected to its base model; merge the adapter into a full safetensors checkpoint first"
        )
    pickles = sorted(p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in PICKLE_SUFFIXES)
    if (path / SAFE_WEIGHTS_FILE).is_file():
        tensors = [SAFE_WEIGHTS_FILE]
    elif (path / SAFE_WEIGHTS_INDEX_FILE).is_file():
        tensors = _index_shards(path, model_dir)
    else:
        others = sorted(p.name for p in path.glob("*.safetensors") if p.is_file())
        found = [
            f"pickle-format files {pickles}, which are never loaded" if pickles else "",
            f"non-standard safetensors files {others}, which transformers does not load" if others else "",
        ]
        detail = "; ".join(f for f in found if f)
        raise P1Refusal(
            f"--model-dir {model_dir!r} has no *.safetensors weights under a standard name "
            f"({SAFE_WEIGHTS_FILE} or {SAFE_WEIGHTS_INDEX_FILE})"
            + (f" (found {detail})" if detail else "")
            + "; convert or rename the checkpoint to safetensors first"
        )
    return {
        "model_dir": str(path.resolve()),
        "safetensors": tensors,
        "ignored_pickle_files": pickles,
    }


def require_cuda(device: str) -> dict[str, Any]:
    """``{"device", "name", "total_memory_bytes", "capability"}`` of ``device``; :class:`P1Refusal`
    with status ``NOT_TESTED`` (exit code 2) when CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise P1Refusal(
            "torch.cuda.is_available() is False: the P1 run needs one CUDA GPU (use --dry-run for the CPU "
            "plumbing check)",
            status=NOT_TESTED,
            exit_code=EXIT_NOT_TESTED,
        )
    dev = torch.device(device)
    if dev.type != "cuda":
        raise P1Refusal(f"--device must be a CUDA device, got {device!r}")
    props = torch.cuda.get_device_properties(dev)
    return {
        "device": str(dev),
        "name": torch.cuda.get_device_name(dev),
        "total_memory_bytes": int(props.total_memory),
        "capability": f"{props.major}.{props.minor}",
    }


def ensure_pad_token(tokenizer: Any) -> str | None:
    """Use EOS as the pad token (in memory only) when the tokenizer has none; returns a note."""
    if tokenizer.eos_token_id is None:
        raise P1Refusal("the tokenizer has no eos_token_id (completions must end at EOS)")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        return "tokenizer had no pad token: EOS is used for padding (in memory, not saved)"
    return None


def load_local_checkpoint(
    model_dir: str, *, device: str | torch.device, dtype: torch.dtype = torch.bfloat16
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Validate ``model_dir`` (:func:`validate_model_dir`) and load tokenizer and causal LM from it
    offline: ``local_files_only=True``, ``use_safetensors=True``, ``trust_remote_code=False``,
    weights in ``dtype`` on ``device``, eval mode. Refuses a model without SwiGLU FFNs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = validate_model_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()
    try:
        refs = find_swiglu_ffns(model)
    except ValueError as e:
        raise P1Refusal(f"the model has no SwiGLU FFNs the capability tools can address: {e}") from e
    note = ensure_pad_token(tokenizer)
    info.update(
        {
            "model_type": getattr(model.config, "model_type", None),
            "num_layers": len(refs),
            "dtype": str(dtype),
            "pad_note": note,
        }
    )
    return model, tokenizer, info


# ============================================================================== tokenizer check
def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return str(tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False))


def check_tokenizer(
    tokenizer: Any, splits: CompositionSplits, *, items_per_slice: int, root_seed: int
) -> dict[str, Any]:
    """Round-trip check of the task rendering with ``tokenizer`` and the token budget.

    Sample: ``items_per_slice`` items of every non-AUDIT slice (purpose
    :data:`TOKENIZER_CHECK_PURPOSE`, index 0) plus every constant digit string of every input length
    (``"0" * L`` ... ``"9" * L``). Checks (``add_special_tokens=False``, decoding without skipping
    special tokens and without clean-up):

    * every character of the alphabet (all prompt / answer characters and the ten digits) and every
      sampled prompt and answer decodes back to itself; a single leading space added by the
      tokenizer is tolerated (the verifiers strip whitespace) and counted;
    * no encoding contains the unknown token.

    Any failure raises :class:`P1Refusal`. ``required_max_new_tokens`` = ``max(longest answer in
    tokens + 1, max_len + 1)``: the first is the budget :class:`VerifierScoreFn` needs (it flags a
    completion truncated when it re-encodes to at least ``max_new_tokens`` tokens), the second the
    character contract that the GRPO operator and ``check_truncation_budget`` enforce.
    """
    ensure_pad_token(tokenizer)
    sampler = ItemSampler(splits, root_seed=root_seed)
    items: list[Item] = []
    for name, split in splits.slices().items():
        if split is Split.AUDIT:
            continue
        items.extend(
            sampler.draw(split, items_per_slice, purpose=TOKENIZER_CHECK_PURPOSE, index=0, slice=name)
        )
    ops = splits.operators
    answers = [it.answer for it in items]
    answers += [d * length for length in range(ops.min_len, ops.max_len + 1) for d in "0123456789"]
    prompts = [it.prompt for it in items]
    alphabet = sorted(set("".join(prompts) + "".join(answers)) | set("0123456789"))
    unk = getattr(tokenizer, "unk_token_id", None)
    failures: list[str] = []
    tolerated = 0

    def check(text: str, what: str) -> None:
        nonlocal tolerated
        ids = _encode(tokenizer, text)
        back = _decode(tokenizer, ids)
        if unk is not None and unk in ids:
            failures.append(f"{what} {text!r} encodes to the unknown token")
        elif back != text:
            if back.startswith(" ") and back[1:] == text:
                tolerated += 1
            else:
                failures.append(f"{what} {text!r} decodes to {back!r}")

    for ch in alphabet:
        check(ch, "character")
    for text in prompts:
        check(text, "prompt")
    for text in answers:
        check(text, "answer")
    if failures:
        raise P1Refusal(
            f"the tokenizer does not round-trip the task text ({len(failures)} failure(s)): "
            + "; ".join(failures[:5])
        )
    answer_tokens = max(len(_encode(tokenizer, a)) for a in answers)
    required = max(answer_tokens + 1, ops.max_len + 1)
    return {
        "items_per_slice": items_per_slice,
        "items": len(items),
        "answers_checked": len(answers),
        "alphabet": "".join(alphabet),
        "leading_space_tolerated": tolerated,
        "max_answer_chars": max(len(a) for a in answers),
        "max_answer_tokens": answer_tokens,
        "max_prompt_tokens": max(len(_encode(tokenizer, p)) for p in prompts),
        "required_max_new_tokens": required,
        "rule": "max(longest answer in tokens + 1, max_len + 1)",
    }


# ============================================================================== memory plan
def _names_bytes(model: nn.Module, names: set[str]) -> tuple[int, int]:
    """``(elements, bytes)`` of the named parameters."""
    elements = size = 0
    for name, p in model.named_parameters():
        if name in names:
            elements += p.numel()
            size += p.numel() * p.element_size()
    return elements, size


def ffn_parameter_names(model: nn.Module, layers: Sequence[int] | None) -> list[str]:
    """gate/up/down weights (and biases) of the SwiGLU FFNs of ``layers`` (``None`` = every layer)."""
    out: list[str] = []
    for ref in find_swiglu_ffns(model, layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out.append(f"{ref.module_path}.{proj}.weight")
            if getattr(ref.module, proj).bias is not None:
                out.append(f"{ref.module_path}.{proj}.bias")
    return out


def sft_parameter_names(model: nn.Module, options: P1Options) -> list[str]:
    """Parameters the SFT stage trains (see :class:`P1Options`)."""
    if options.sft_scope == "all":
        return [n for n, _ in model.named_parameters()]
    layers = options.sft_layers if options.sft_layers is not None else options.channel_layers
    return ffn_parameter_names(model, layers)


def _down_names(model: nn.Module, layers: Sequence[int]) -> list[str]:
    return [f"{ref.module_path}.down_proj.weight" for ref in find_swiglu_ffns(model, layers)]


def _optimizer_phase(model: nn.Module, names: Sequence[str], *, masked: bool) -> dict[str, Any]:
    """Device bytes of a ``HookedOptimizer(AdamW, master_weights=True)`` over ``names``: per element
    an fp32 master (4), AdamW ``exp_avg`` + ``exp_avg_sq`` in fp32 (8), the gradient in the served
    dtype and its transient fp32 copy during ``step()`` (4); with ``masked`` (MaskRule) also the
    wrapper's per-step fp32 clone of tracked parameters (4), the bool mask cached on the device by
    the rule and the gradient-hook mask (2) and the device clone :func:`swapped` makes to evaluate the
    incumbent (served dtype)."""
    elements, served = _names_bytes(model, set(names))
    parts = {
        "fp32_master": 4 * elements,
        "adamw_moments_fp32": 8 * elements,
        "grad_served_dtype": served,
        "grad_fp32_transient": 4 * elements,
    }
    if masked:
        parts["mask_rule_fp32_clone"] = 4 * elements
        parts["bool_masks"] = 2 * elements
        parts["swapped_incumbent_clone"] = served
    return {
        "tensors": len(set(names)),
        "elements": elements,
        "served_bytes": served,
        "parts": parts,
        "total_bytes": sum(parts.values()),
    }


def _es_phase(model: nn.Module, selection: ChannelSelection, population: int) -> dict[str, Any]:
    """Device bytes of an ``ESOperator`` over ``ChannelGainSpace(selection)``: the base copy of every
    selected ``down_proj`` column (served dtype), the :func:`swapped` clone of the touched
    ``down_proj`` tensors and fp32 vectors (``population + 2`` of ``dim`` elements)."""
    base = 0
    for ref in find_swiglu_ffns(model, selection.layers()):
        w = ref.down.weight
        base += int(w.shape[0]) * len(selection.channels(ref.layer_index)) * w.element_size()
    _, swap = _names_bytes(model, set(_down_names(model, selection.layers())))
    vectors = 4 * selection.count() * (population + 2)
    parts = {"gain_space_base": base, "swapped_incumbent_clone": swap, "es_vectors_fp32": vectors}
    return {"dim": selection.count(), "parts": parts, "total_bytes": sum(parts.values())}


def memory_plan(
    model: nn.Module,
    config: PilotConfig,
    *,
    sft_names: Sequence[str],
    selections: Mapping[str, ChannelSelection],
    reset_names: Sequence[str],
    device_total_bytes: int | None,
    fraction: float,
) -> dict[str, Any]:
    """Computed (never measured) memory plan of the run.

    ``selections`` maps operator names present in the run (``grpo``, ``es_informed``,
    ``es_random``, ``es_full``) to their channel selection; the GRPO optimizer holds the **whole**
    FFN tensors of every layer the selection touches (the MaskRule does not shrink optimizer state).
    Device peak estimate = parameter bytes + the largest phase (SFT, each operator). Host bytes:
    the start-state snapshot and the size of one gate snapshot per operator (parameters in their
    dtype; for GRPO also fp32 masters and AdamW moments); the gate controller can hold two of them
    at once (incumbent and the SHRINK candidate, or old and new incumbent while promoting), so
    ``host_peak_estimate_bytes`` = start snapshot + 2 x the largest gate snapshot (arms run one at a
    time). Excluded: activations, autograd buffers, KV cache,
    logits, CUDA context and allocator fragmentation. ``over_budget`` compares the device estimate
    with ``fraction * device_total_bytes`` (``None`` when the total is unknown, e.g. a dry run).
    """
    param_elements = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    phases: dict[str, Any] = {}
    if config.sft_steps > 0 and sft_names:
        phases["sft"] = _optimizer_phase(model, sft_names, masked=False)
    host: dict[str, int] = {"start_reset_snapshot": _names_bytes(model, set(reset_names))[1]}
    for op, selection in selections.items():
        if op == "grpo":
            names = ffn_parameter_names(model, selection.layers())
            phase = _optimizer_phase(model, names, masked=True)
            phase["layers"] = selection.layers()
            phases["grpo"] = phase
            host["grpo_gate_snapshot"] = phase["served_bytes"] + 12 * phase["elements"]
        else:
            phases[op] = _es_phase(model, selection, config.es_population)
            host[f"{op}_gate_snapshot"] = _names_bytes(model, set(_down_names(model, selection.layers())))[1]
    largest = max((p["total_bytes"] for p in phases.values()), default=0)
    peak = param_bytes + buffer_bytes + largest
    gate_snapshots = [v for k, v in host.items() if k.endswith("_gate_snapshot")]
    host_peak = host["start_reset_snapshot"] + 2 * max(gate_snapshots, default=0)
    budget = None if device_total_bytes is None else int(fraction * device_total_bytes)
    return {
        "computed_not_measured": True,
        "status": "computed from tensor shapes; NOT measured",
        "parameters": param_elements,
        "parameter_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "phases": phases,
        "device_peak_estimate_bytes": peak,
        "host_bytes": host,
        "host_peak_estimate_bytes": host_peak,
        "host_rule": "start snapshot + 2 x largest gate snapshot",
        "device_total_bytes": device_total_bytes,
        "budget_fraction": fraction,
        "budget_bytes": budget,
        "over_budget": None if budget is None else peak > budget,
        "excluded": (
            "activations, autograd buffers, KV cache, logits, CUDA context and allocator fragmentation are "
            "not included"
        ),
    }


def _gib(value: int | None) -> str:
    """Bytes as GiB (MiB / KiB / B below one GiB / MiB / KiB); ``-`` for ``None``."""
    if value is None:
        return "-"
    for unit, size in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if abs(value) >= size:
            return f"{value / size:.2f} {unit}"
    return f"{value} B"


def format_memory_plan(plan: Mapping[str, Any]) -> list[str]:
    """Human-readable lines of :func:`memory_plan` (also used in ``summary.md``)."""
    lines = [
        f"memory plan ({plan['status']}): parameters {plan['parameters']} = {_gib(plan['parameter_bytes'])}",
    ]
    for name, phase in plan["phases"].items():
        parts = ", ".join(f"{k} {_gib(v)}" for k, v in phase["parts"].items())
        lines.append(f"  {name}: {_gib(phase['total_bytes'])} ({parts})")
    lines.append(
        f"  device peak estimate {_gib(plan['device_peak_estimate_bytes'])}; budget {_gib(plan['budget_bytes'])} "
        f"(over budget: {plan['over_budget']})"
    )
    lines.append(
        "  host: "
        + ", ".join(f"{k} {_gib(v)}" for k, v in plan["host_bytes"].items())
        + f"; host peak estimate {_gib(plan['host_peak_estimate_bytes'])} ({plan['host_rule']})"
    )
    lines.append(f"  {plan['excluded']}")
    return lines


def enforce_budget(plan: Mapping[str, Any], options: P1Options, warnings_out: list[str]) -> None:
    """:class:`P1Refusal` when the computed plan exceeds the budget (a warning with
    ``allow_over_budget``)."""
    if not plan["over_budget"]:
        return
    msg = (
        f"computed device peak {_gib(plan['device_peak_estimate_bytes'])} exceeds the budget "
        f"{_gib(plan['budget_bytes'])} ({plan['excluded']}); restrict --channel-layers / --sft-layers, "
        "lower informed_k or pass --allow-over-budget"
    )
    if not options.allow_over_budget:
        raise P1Refusal(msg)
    warnings_out.append(msg)


# ============================================================================== SFT
def train_sft_subset(
    model: nn.Module, tokenizer: Any, items: Sequence[Item], config: PilotConfig, names: Sequence[str]
) -> dict[str, Any]:
    """The P0 SFT recipe (batches drawn with replacement from ``_seed("sft_batches", seed)``, loss on
    answer + EOS tokens, right-padded batches with an attention mask; see
    :func:`~research_rl.capability.pilot.train_sft`) on the parameters ``names`` only, through a
    ``HookedOptimizer(AdamW(lr=sft_lr, weight_decay=0), master_weights=True)``: bf16 weights keep
    fp32 masters, so small updates are not rounded away. Every other parameter has
    ``requires_grad=False`` during SFT; the flags are restored and gradients dropped afterwards. The
    model is left in eval mode."""
    if config.sft_steps == 0 or not items or not names:
        model.eval()
        return {"steps": 0, "loss_curve": [], "trained_parameters": 0}
    params = dict(model.named_parameters())
    wanted = set(names)
    flags = {n: p.requires_grad for n, p in params.items()}
    trainable = [(n, p) for n, p in params.items() if n in wanted]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    curve: list[dict[str, float]] = []
    try:
        for n, p in params.items():
            p.requires_grad_(n in wanted)
        inner = torch.optim.AdamW([p for _, p in trainable], lr=config.sft_lr, weight_decay=0.0)
        optimizer = HookedOptimizer(inner, trainable, master_weights=True)
        encoded = [_encode_example(tokenizer, it) for it in items]
        gen = torch.Generator().manual_seed(_seed("sft_batches", config.seed))
        model.train()
        for step in range(config.sft_steps):
            idx = torch.randint(len(encoded), (config.sft_batch_size,), generator=gen).tolist()
            rows = [encoded[i] for i in idx]
            length = max(len(r[0]) for r in rows)
            ids = torch.tensor([r[0] + [pad] * (length - len(r[0])) for r in rows], dtype=torch.long)
            labels = torch.tensor([r[1] + [-100] * (length - len(r[1])) for r in rows], dtype=torch.long)
            mask = torch.tensor(
                [[1] * len(r[0]) + [0] * (length - len(r[0])) for r in rows], dtype=torch.long
            )
            loss = model(
                input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)
            ).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 50 == 0 or step == config.sft_steps - 1:
                curve.append({"step": step, "loss": float(loss.detach())})
        master_resyncs = optimizer.master_resyncs
    finally:
        for n, p in params.items():
            p.grad = None
            p.requires_grad_(flags[n])
        model.eval()
    return {
        "steps": config.sft_steps,
        "loss_curve": curve,
        "trained_parameters": sum(p.numel() for _, p in trainable),
        "trained_tensors": len(trainable),
        "optimizer": "HookedOptimizer(AdamW, master_weights=True)",
        "master_resyncs": master_resyncs,
    }


# ============================================================================== channels
def select_channels(
    model: nn.Module,
    tokenizer: Any,
    sampler: ItemSampler,
    config: PilotConfig,
    budget: int,
    channel_layers: Sequence[int] | None,
) -> dict[str, Any]:
    """:func:`~research_rl.capability.pilot.select_informed_channels`, optionally restricted to
    ``channel_layers``: the global top-``k'`` selection with ``k' = ceil(informed_k * layers /
    len(channel_layers))`` (so about ``informed_k`` channels fall into the allowed layers) is
    intersected with ``channel_layers``. The global ranking restricted to a layer subset keeps its
    order, so the result is the top channels within those layers; its size is recorded (it is not
    exactly ``informed_k``)."""
    if channel_layers is None:
        info = select_informed_channels(model, tokenizer, sampler, config, budget)
        info["restriction"] = None
        return info
    num_layers = len(find_swiglu_ffns(model))
    bad = [layer for layer in channel_layers if layer >= num_layers]
    if bad:
        raise P1Refusal(f"--channel-layers {bad} out of range (the model has {num_layers} layers)")
    k_global = math.ceil(config.informed_k * num_layers / len(channel_layers))
    info = select_informed_channels(
        model, tokenizer, sampler, dataclasses.replace(config, informed_k=k_global), budget
    )
    full = _selection_from(info)
    allowed = set(channel_layers)
    restricted = ChannelSelection(
        {layer: full.channels(layer) for layer in full.layers() if layer in allowed}
    )
    if restricted.count() == 0:
        raise P1Refusal(f"no informed channel falls into --channel-layers {list(channel_layers)}")
    info["restriction"] = {
        "channel_layers": list(channel_layers),
        "requested_k": config.informed_k,
        "global_k": k_global,
        "global_count": full.count(),
        "count": restricted.count(),
    }
    info["selection"] = {str(k): list(v) for k, v in restricted.to_dict().items()}
    info["count"] = restricted.count()
    return info


def check_layers(model: nn.Module, options: P1Options) -> None:
    """:class:`P1Refusal` when ``channel_layers`` / ``sft_layers`` name layers the model lacks."""
    num_layers = len(find_swiglu_ffns(model))
    for flag, layers in (("--channel-layers", options.channel_layers), ("--sft-layers", options.sft_layers)):
        bad = [layer for layer in layers or () if layer >= num_layers]
        if bad:
            raise P1Refusal(f"{flag} {bad} out of range (the model has {num_layers} layers)")


def upper_bound_selections(
    model: nn.Module, config: PilotConfig, channel_layers: Sequence[int] | None
) -> dict[str, ChannelSelection]:
    """Selections that bound every operator's memory before the informed selection exists: every
    channel of ``channel_layers`` (all layers if ``None``) for ``grpo`` / ``es_informed`` /
    ``es_random``, every channel for ``es_full``."""
    bound = ChannelSelection.all_channels(model, channel_layers)
    out: dict[str, ChannelSelection] = {}
    for spec in arm_specs(config):
        out[spec.operator] = ChannelSelection.all_channels(model) if spec.operator == "es_full" else bound
    return out


def arm_selections(
    model: nn.Module, config: PilotConfig, selection_info: Mapping[str, Any]
) -> dict[str, ChannelSelection]:
    """Channel selection of every operator present in ``config.arms`` (for the memory plan), built
    exactly as the arms build them."""
    from research_rl.capability.scores import density_matched_random

    informed = _selection_from(selection_info)
    out: dict[str, ChannelSelection] = {}
    for spec in arm_specs(config):
        if spec.operator in out:
            continue  # es_random: every seed has the same per-layer counts, hence the same layers and sizes
        if spec.operator in ("grpo", "es_informed"):
            out[spec.operator] = informed
        elif spec.operator == "es_full":
            out[spec.operator] = ChannelSelection.all_channels(model)
        else:
            out[spec.operator] = density_matched_random(
                informed, model, _seed("pilot_random_channels", config.seed, spec.seed) % (2**31)
            )
    return out


def reset_parameter_names(model: nn.Module, selections: Mapping[str, ChannelSelection]) -> list[str]:
    """Every tensor an arm can write: the FFN gate/up/down of the GRPO layers and the ``down_proj``
    of every ES layer (ChannelGainSpace with ``up=False`` writes ``down_proj`` columns only)."""
    names: set[str] = set()
    for op, selection in selections.items():
        if op == "grpo":
            names.update(ffn_parameter_names(model, selection.layers()))
        else:
            names.update(_down_names(model, selection.layers()))
    return sorted(names)


# ============================================================================== one arm
def _has_signal(report: OperatorReport) -> bool:
    """True when an update could move the weights: a non-zero GRPO advantage, or ES fitness that
    differs between members (all-equal fitness gives a zero shaped update)."""
    if "advantages" in report.extra:
        return any(a != 0.0 for a in report.extra["advantages"])
    return float(report.extra.get("fitness_std", 0.0)) > 0.0


def _grpo_operator(
    spec: ArmSpec,
    model: nn.Module,
    tokenizer: Any,
    sampler: ItemSampler,
    config: PilotConfig,
    selection: ChannelSelection,
    budget: int,
) -> tuple[TorchGRPOOperator, Any, dict[str, Any]]:
    """GRPO over the informed channel subset: ``setup_channel_training(disable_grad_outside=True)``
    (MaskRule + gradient masks; every parameter outside the touched FFNs gets
    ``requires_grad=False``) and ``HookedOptimizer(AdamW(lr=grpo_lr, weight_decay=0),
    master_weights=True)``. Returns the operator, the channel-training handle (call ``remove()``)
    and a description."""
    v1, v2 = _verifiers(config, spec.variant)
    handle = setup_channel_training(model, selection, disable_grad_outside=True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    try:
        inner = torch.optim.AdamW([p for _, p in named], lr=config.grpo_lr, weight_decay=0.0)
        optimizer = HookedOptimizer(inner, named, rules=handle.rules, master_weights=True)
        op = TorchGRPOOperator(
            model,
            tokenizer,
            sampler,
            v1,
            v2,
            group_size=config.grpo_group_size,
            prompts_per_update=config.grpo_prompts_per_update,
            lr=config.grpo_lr,
            max_new_tokens=budget,
            temperature=config.grpo_temperature,
            optimizer=optimizer,
            screen=config.grpo_screen,
            seed=_seed("pilot_operator", config.seed, spec.seed, spec.operator) % (2**31),
        )
    except BaseException:
        handle.remove()
        raise
    info = {
        "type": "torch_grpo",
        "optimizer": "HookedOptimizer(AdamW, master_weights=True) + MaskRule(channel subset)",
        "channels": selection.count(),
        "layers": selection.layers(),
        "parameters": len(op.parameter_names()),
        "trainable_elements": sum(p.numel() for _, p in named),
        "screen": config.grpo_screen,
    }
    return op, handle, info


def run_arm(
    spec: ArmSpec,
    model: nn.Module,
    tokenizer: Any,
    config: PilotConfig,
    *,
    start: StartMap,
    selection_info: Mapping[str, Any],
    items_per_retention_slice: int,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    """One arm on the (already restored) start weights; the glue of
    :func:`research_rl.capability.pilot._run_arm` with the P1 model and operators (that function
    builds its own tiny model and cannot take a loaded checkpoint). Returns the
    :func:`~research_rl.capability.pilot._arm_result` dict plus the GPU peak memory (measured on CUDA
    only)."""
    t_start = time.perf_counter()
    splits = build_splits(config)
    budget = _budget(config, splits)
    sampler = ItemSampler(splits, root_seed=_sampler_root(config, spec.seed))
    transfer, retention = _slices(config, splits)
    gate_cfg = dataclasses.replace(
        config.base_gate_config(items_per_retention_slice=items_per_retention_slice),
        enabled=spec.gate == "gate_on",
    )
    v2 = ExactAnswerVerifier()
    gate_fn = _CountingScoreFn(tokenizer, v2, max_new_tokens=budget, batch_size=config.eval_batch_size)
    track_fn = _CountingScoreFn(tokenizer, v2, max_new_tokens=budget, batch_size=config.eval_batch_size)
    oracle_fn = _CountingScoreFn(tokenizer, v2, max_new_tokens=budget, batch_size=config.eval_batch_size)
    probe = CapabilityProbe(
        sampler,
        gate_fn,
        transfer_slices=transfer,
        retention_slices=retention,
        items_per_transfer_slice=gate_cfg.items_per_transfer_slice,
        items_per_retention_slice=gate_cfg.items_per_retention_slice,
    )
    tracker = CapabilityTracker(sampler, track_fn, start, items_per_slice=config.track_items_per_slice)
    device = next(model.parameters()).device
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    handle = None
    inner: UpdateOperator
    if spec.operator == "grpo":
        inner, handle, space_info = _grpo_operator(
            spec, model, tokenizer, sampler, config, _selection_from(selection_info), budget
        )
    else:
        inner, space_info = _make_operator(spec, model, tokenizer, sampler, config, selection_info, budget)
    try:
        operator = _RecordingOperator(inner, kind="grpo" if spec.operator == "grpo" else "es")
        alarm_kwargs = dict(config.alarm)
        if spec.operator != "grpo":
            alarm_kwargs["rollouts_per_unit"] = config.es_population
        alarm = DisagreementAlarm(**alarm_kwargs)
        updates = config.updates_for(spec.operator)
        exclude = excluded_uids(
            sampler,
            start,
            track_items_per_slice=config.track_items_per_slice,
            max_step=oracle_exclusion_steps(config, updates),
        )
        controller = _PilotController(CapabilityGate(gate_cfg), probe)
        log_path = output_dir / "decisions" / f"{spec.file_stem}.{label}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()
        loop = GatedLoop(
            model,
            operator,
            controller,
            alarm,
            config=gate_cfg,
            log_path=log_path,
            allow_short_horizon=config.allow_short_horizon,
        )
        horizon = loop.horizon()
        required = loop.required_verdicts(updates)
        monitor = _ArmMonitor(
            model=model,
            operator=operator,
            tracker=tracker,
            oracle_fn=oracle_fn,
            sampler=sampler,
            config=config,
            exclude=exclude,
            label=f"{label} {spec.key}",
            total_blocks=updates // gate_cfg.block_updates,
            require_transfer_gain=gate_cfg.require_transfer_gain,
        )
        controller.monitor = monitor
        point0 = tracker.measure(model, 0, None)
        start_fingerprint = fingerprint(model, operator.parameter_names())
        monitor.start_clock(time.perf_counter() - t_start)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = loop.run(updates)
        rows = [_report_counts(r, operator.kind) for r in operator.reports]
        trailing = loop.updates_since_decision
        trailing_point = None
        if trailing > 0:
            tail = operator.reports[-trailing:]
            reward = sum(float(r.reward_v1) for r in tail) / len(tail)
            trailing_point = tracker.measure(model, len(monitor.blocks) + 1, reward).to_dict()
        final_fingerprint = fingerprint(model, operator.parameter_names())
    finally:
        if handle is not None:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.eval()
    result = _arm_result(
        spec,
        config,
        records=records,
        monitor=monitor,
        rows=rows,
        point0=point0.to_dict(),
        trailing_point=trailing_point,
        trailing_updates=trailing,
        tracker=tracker,
        space_info=space_info,
        alarm=alarm,
        gate_cfg=gate_cfg,
        horizon=horizon,
        required=required,
        updates=updates,
        eval_items={
            "gate": gate_fn.items_scored,
            "tracker": track_fn.items_scored,
            "oracle": oracle_fn.items_scored,
        },
        warnings_seen=list(dict.fromkeys(str(w.message) for w in caught)),
        wall=time.perf_counter() - t_start,
        final_fingerprint=final_fingerprint,
    )
    result["final_fingerprint_scope"] = "the operator's parameter_names() (its rollback scope)"
    result["weights_changed"] = final_fingerprint != start_fingerprint
    result["updates_with_signal"] = sum(1 for r in operator.reports if _has_signal(r))
    result["decision_log"] = str(log_path.relative_to(output_dir))
    if on_cuda:
        result["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    else:
        result["gpu_peak_allocated_bytes"] = NOT_TESTED
        result["gpu_peak_reserved_bytes"] = NOT_TESTED
    return result


# ============================================================================== the run
def _log(message: str) -> None:
    print(f"[p1] {message}", flush=True)


def build_dry_run_model(config: PilotConfig) -> tuple[nn.Module, Any]:
    """The tiny in-process Llama of ``config`` (char tokenizer) in bf16 on CPU, so the dry run
    exercises the same bf16 + fp32-master path as a real run."""
    tokenizer = build_char_tokenizer()
    model = build_model(config, len(tokenizer)).to(torch.bfloat16)
    model.eval()
    return model, tokenizer


def dry_run_config(**overrides: Any) -> PilotConfig:
    """``--dry-run`` configuration: :func:`~research_rl.capability.pilot.smoke_config` with ES over
    ChannelGainSpace, one gate block per arm (``updates = block_updates``) and 400 SFT steps (bf16
    on CPU is slow; the dry run checks plumbing and its start is not meant to be good, but with 400
    steps both default arms get updates with signal - some V1 passes - and move their weights,
    measured: about 15 s on 2 CPU threads)."""
    base = smoke_config()
    block = base.base_gate_config().block_updates
    values: dict[str, Any] = {
        "updates": block,
        "es_space": "gain",
        "es_full_space": "all_channels",
        "sft_steps": 400,
    }
    values.update(overrides)
    return smoke_config(**values)


def gpu_config(**overrides: Any) -> PilotConfig:
    """Configuration of a ``--model-dir`` run: :class:`PilotConfig` defaults, then
    :data:`P1_GPU_DEFAULTS` (NOT_TESTED starting values), then ``overrides``."""
    values = PilotConfig().to_dict()
    values.update(P1_GPU_DEFAULTS)
    values.update(overrides)
    return PilotConfig.from_dict(values)


def run_p1(
    model: nn.Module,
    tokenizer: Any,
    config: PilotConfig,
    options: P1Options,
    output_dir: str | Path,
    *,
    environment: Mapping[str, Any],
    model_info: Mapping[str, Any],
    device_total_bytes: int | None,
) -> dict[str, Any]:
    """Run P1 on a loaded ``model`` (module docstring) and write the outputs; returns the results
    dict. The same code path serves the dry run and the GPU run; only how the model was obtained and
    ``environment`` differ. Every refusal that does not need the trained start model (configuration,
    layer ranges, tokenizer, retention horizon, memory upper bound) happens before anything trains or
    is written."""
    check_p1_config(config)
    label = options.label
    banner = P1_BANNERS[label]
    t0 = time.perf_counter()
    run_warnings: list[str] = []
    timing: dict[str, Any] = {}
    splits = build_splits(config)
    splits.check_disjoint()
    tok_check = check_tokenizer(
        tokenizer,
        splits,
        items_per_slice=options.tokenizer_check_items,
        root_seed=_sampler_root(config, -2),
    )
    required = int(tok_check["required_max_new_tokens"])
    if config.max_new_tokens is not None and config.max_new_tokens < required:
        raise P1Refusal(
            f"max_new_tokens={config.max_new_tokens} < {required} (longest answer in tokens + 1 with this tokenizer)"
        )
    config = dataclasses.replace(config, max_new_tokens=max(required, config.max_new_tokens or 0))
    budget = _budget(config, splits)
    check_layers(model, options)
    sft_names = sft_parameter_names(model, options)
    _, retention = _slices(config, splits)
    gate_choice = choose_retention_items(config, retention)
    if gate_choice["short_horizon"]:
        msg = (
            f"items_per_retention_slice={gate_choice['items_per_retention_slice']} cannot cover the worst-case "
            f"{gate_choice['worst_case_verdicts']} verdicts; later verdicts may be horizon_exhausted"
        )
        if not config.allow_short_horizon:
            raise P1Refusal(msg + " (set allow_short_horizon or add larger retention_items_candidates)")
        run_warnings.append(msg)
    bounds = upper_bound_selections(model, config, options.channel_layers)
    plan = memory_plan(
        model,
        config,
        sft_names=sft_names,
        selections=bounds,
        reset_names=reset_parameter_names(model, bounds),
        device_total_bytes=device_total_bytes,
        fraction=options.memory_fraction,
    )
    plan["stage"] = "upper bound before SFT: every channel of --channel-layers (all layers if unset)"
    _log(f"memory plan stage 1 ({plan['stage']})")
    for line in format_memory_plan(plan):
        _log(line)
    enforce_budget(plan, options, run_warnings)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = {"label": label, "script_status": SCRIPT_STATUS, "script_note": SCRIPT_NOTE, "banner": banner}
    (out / "p1_config.json").write_text(
        json.dumps({**header, "pilot_config": config.to_dict(), "p1_options": options.to_dict()}, indent=2),
        encoding="utf-8",
    )
    _log(f"{label}: {SCRIPT_NOTE}")
    with _torch_threads(config.threads):
        t = time.perf_counter()
        items = sft_examples(splits, config.sft_examples, seed=config.seed, demo_pairs=config.sft_demo_pairs)
        sft = train_sft_subset(model, tokenizer, items, config, sft_names)
        timing["sft_s"] = time.perf_counter() - t
        start_delta = None
        if sft["steps"] > 0:
            wanted = set(sft_names)
            tensors = {
                n: p.detach().cpu().contiguous().clone() for n, p in model.named_parameters() if n in wanted
            }
            start_delta = out / "start_delta.safetensors"
            save_file(
                tensors,
                str(start_delta),
                metadata={
                    "label": label,
                    "script_status": SCRIPT_STATUS,
                    "base_model": str(model_info.get("model_dir", "in-process tiny model")),
                    "content": "SFT-trained tensors only; load the base checkpoint, then these tensors",
                },
            )
        _log(f"SFT: {sft['steps']} steps on {sft['trained_parameters']} parameters")
        base_sampler = ItemSampler(splits, root_seed=_sampler_root(config, -1))
        start_fn = VerifierScoreFn(
            tokenizer, ExactAnswerVerifier(), max_new_tokens=budget, batch_size=config.eval_batch_size
        )
        t = time.perf_counter()
        start = StartMap.measure(
            model,
            base_sampler,
            start_fn,
            items_per_slice=config.start_items_per_slice,
            k=config.start_k,
            num_samples=config.start_num_samples,
            seed=_seed("start_map", config.seed) % (2**31),
        )
        timing["start_map_s"] = time.perf_counter() - t
        _log("start map: " + ", ".join(f"{k}={v['pass1']:.3f}" for k, v in start.summary().items()))
        t = time.perf_counter()
        selection = select_channels(model, tokenizer, base_sampler, config, budget, options.channel_layers)
        timing["selection_s"] = time.perf_counter() - t
        _log(f"informed channels: {selection['scorer']} ({selection['count']} channels)")
        selections = arm_selections(model, config, selection)
        reset_names = reset_parameter_names(model, selections)
        plan = memory_plan(
            model,
            config,
            sft_names=sft_names,
            selections=selections,
            reset_names=reset_names,
            device_total_bytes=device_total_bytes,
            fraction=options.memory_fraction,
        )
        plan["stage"] = "exact after channel selection"
        _log(f"memory plan stage 2 ({plan['stage']})")
        for line in format_memory_plan(plan):
            _log(line)
        (out / "memory_plan.json").write_text(json.dumps({**header, **plan}, indent=2), encoding="utf-8")
        enforce_budget(plan, options, run_warnings)
        for name in retention:
            if name in start.slices and start.pass1(name) < 0.5:
                run_warnings.append(
                    f"start retention {name} pass@1 = {start.pass1(name):.3f} < 0.5: retention evidence is noisy and "
                    "gate verdicts on this slice may reflect the weak start rather than the candidate"
                )
        start_fingerprint = fingerprint(model)
        reset: StateSnapshot = capture(model, reset_names)
        reset_fp = fingerprint(model, reset_names)
        arms: list[dict[str, Any]] = []
        t = time.perf_counter()
        for spec in arm_specs(config):
            if fingerprint(model, reset_names) != reset_fp:
                raise RuntimeError(f"start weights changed before arm {spec.key}")
            arms.append(
                run_arm(
                    spec,
                    model,
                    tokenizer,
                    config,
                    start=start,
                    selection_info=selection,
                    items_per_retention_slice=int(gate_choice["items_per_retention_slice"]),
                    output_dir=out,
                    label=label,
                )
            )
            restore(model, reset)
            if fingerprint(model, reset_names) != reset_fp:
                raise RuntimeError(f"restoring the start weights after arm {spec.key} was not bit-exact")
        timing["arms_s"] = time.perf_counter() - t
        if fingerprint(model) != start_fingerprint:
            raise RuntimeError("the model differs from the start checkpoint after the arms")
    for arm in arms:
        if arm["alarm"]["skipped_updates"] == arm["alarm"]["updates"] and arm["alarm"]["updates"] > 0:
            run_warnings.append(
                f"{arm['key']}: the alarm skipped every update (effective units below min_rollouts); it never could fire"
            )
    exercise = gate_exercise(arms, config.min_improved_blocks)
    run_warnings.extend(f"gate not exercised: {problem}" for problem in exercise["problems"])
    results: dict[str, Any] = {
        "version": _RESULTS_VERSION,
        "pilot": "P1",
        **header,
        "measured_on": environment["measured_on"],
        "status": "complete",
        "config": config.to_dict(),
        "p1_options": options.to_dict(),
        "environment": dict(environment),
        "model": dict(model_info),
        "tokenizer_check": tok_check,
        "memory_plan": plan,
        "splits": {
            "fingerprint": splits.fingerprint(),
            "num_operators": config.num_operators,
            "operators": splits.operators.describe(),
            "rules_per_slice": {k: len(v) for k, v in splits.slice_rules.items()},
            "slices": {k: v.value for k, v in splits.slices().items()},
        },
        "sft": {
            **sft,
            "examples": config.sft_examples,
            "demo_pairs": config.sft_demo_pairs,
            "parameters": count_parameters(model),
            "start_fingerprint": start_fingerprint,
            "start_delta": None if start_delta is None else start_delta.name,
        },
        "start_map": start.to_dict(),
        "start_summary": start.summary(),
        "selection": selection,
        "gate_choice": gate_choice,
        "oracle": {
            "purpose": ORACLE_PURPOSE,
            "pool": TRACK_POOL,
            "items_per_slice": config.oracle_items_per_slice,
            "slices": list(_rule_slices(splits)),
            "improvement_threshold": config.oracle_threshold,
            "definition": "as in research_rl.capability.pilot (results['oracle']['definition'] of a P0 run)",
        },
        "counting": {
            "operator_calls": "as reported by the operators (see research_rl.capability.pilot)",
            "fpe": "sequence level: every generated / training-forward sequence = 1 forward, every backward sequence = 2",
        },
        "arms": arms,
        "gate_exercise": exercise,
        "warnings": run_warnings,
    }
    results["criteria"] = evaluate_criteria(results)
    timing["total_s"] = time.perf_counter() - t0
    results["timing"] = timing
    results = _finite(results)
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    (out / "summary.md").write_text(render_p1_summary(results), encoding="utf-8")
    _log(f"done in {timing['total_s']:.1f}s; wrote {out / 'results.json'} and {out / 'summary.md'} ({label})")
    return results


# ============================================================================== summary
_P0_TITLE = "# Capability pilot P0 summary"
_P0_SCOPE = (
    "Every number below comes from a tiny Llama on synthetic digit-string tasks on CPU. "
    f"GPU metrics: {NOT_TESTED}. Smoke runs (`--smoke`) are plumbing checks, not results."
)
_P0_GPU_ROW = f"| GPU memory / throughput | {NOT_TESTED} |"


def render_p1_summary(results: Mapping[str, Any]) -> str:
    """``summary.md`` of P1: :func:`~research_rl.capability.pilot.render_summary` with the P0 title,
    banner and scope line replaced by P1 ones (``RuntimeError`` if any expected P0 string is
    missing, so a changed P0 layout can never leave a false label), plus a P1 section (script
    status, ``measured_on``, tokenizer check, computed memory plan, measured GPU peaks)."""
    label = str(results["label"])
    text = render_summary(results)
    if label == LABEL_DRY_RUN:
        scope = (
            "DRY_RUN_CPU: every number below comes from a tiny in-process Llama (bf16) on synthetic digit-string "
            f"tasks on CPU. GPU metrics: {NOT_TESTED}. A dry run is a plumbing check, not a result."
        )
        gpu_row = _P0_GPU_ROW
    else:
        scope = (
            f"P1_GPU on {results['measured_on']}: numbers come from the local checkpoint `{results['model'].get('model_dir')}` "
            f"on synthetic digit-string tasks. Script status {SCRIPT_STATUS}. GPU peak memory per arm is measured; "
            f"throughput is {NOT_TESTED}."
        )
        gpu_row = f"| GPU memory / throughput | peak memory measured (P1 section); throughput {NOT_TESTED} |"
    for old, new in (
        (_P0_TITLE, f"# Capability pilot P1 summary ({label})"),
        (f"**{BANNER}**", f"**{results['banner']}**"),
        (_P0_SCOPE, scope),
        (_P0_GPU_ROW, gpu_row),
    ):
        if old == _P0_GPU_ROW and not results.get("arms"):
            continue
        if old not in text:
            raise RuntimeError(f"render_summary layout changed: {old!r} not found")
        text = text.replace(old, new)
    plan = results["memory_plan"]
    tok = results["tokenizer_check"]
    lines = [
        "## P1",
        "",
        f"- label: **{label}**; script status: **{results['script_status']}** ({results['script_note']})",
        f"- measured_on: `{results['measured_on']}`",
        f"- tokenizer check: {tok['items']} items + {tok['answers_checked']} answers round-trip; longest answer "
        f"{tok['max_answer_tokens']} tokens ({tok['max_answer_chars']} chars); max_new_tokens = "
        f"{results['config']['max_new_tokens']} ({tok['rule']})",
        "",
        "Memory plan (computed from tensor shapes, NOT measured):",
        "",
        "```",
        *format_memory_plan(plan),
        "```",
        "",
        "| arm | GPU peak allocated | GPU peak reserved |",
        "|---|---|---|",
    ]
    for arm in results.get("arms", []):
        lines.append(
            f"| {arm['key']} | {arm['gpu_peak_allocated_bytes']} | {arm['gpu_peak_reserved_bytes']} |"
        )
    lines.append("")
    marker = "## Setup"
    if marker not in text:
        raise RuntimeError("render_summary layout changed: '## Setup' not found")
    return text.replace(marker, "\n".join(lines) + "\n" + marker, 1)


# ============================================================================== CLI
_EPILOG = (
    f"Script status: {SCRIPT_NOTE} Without CUDA the script refuses to run (exit 2, status NOT_TESTED) "
    "unless --dry-run is given; --dry-run outputs are labelled DRY_RUN_CPU. Nothing is ever downloaded."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="local checkpoint dir (config.json + model.safetensors or model.safetensors.index.json)",
    )
    p.add_argument("--output-dir", required=True, help="directory for results.json, summary.md, ...")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny in-process model on CPU, 1 block per arm, outputs labelled DRY_RUN_CPU (what the tests run)",
    )
    p.add_argument("--device", default="cuda:0", help="CUDA device of a real run")
    p.add_argument(
        "--config", default=None, help="JSON file of PilotConfig fields (applied before the flags)"
    )
    p.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--arms", default=None, help="comma-separated operator or operator+gate")
    p.add_argument("--variants", default=None, help="comma-separated subset of leak,clean")
    p.add_argument("--updates", type=int, default=None, help="operator updates per arm")
    p.add_argument("--sft-steps", type=int, default=None, help="SFT steps (0 skips SFT)")
    p.add_argument(
        "--sft-scope", choices=("ffn", "all"), default=None, help="SFT parameters (default ffn; dry run all)"
    )
    p.add_argument(
        "--sft-layers", default=None, help="layers of an ffn SFT, e.g. 20-27 (default: --channel-layers)"
    )
    p.add_argument(
        "--channel-layers", default=None, help="restrict the channel subset to these layers, e.g. 20-27"
    )
    p.add_argument(
        "--memory-fraction", type=float, default=0.9, help="share of GPU memory the computed plan may use"
    )
    p.add_argument(
        "--allow-over-budget", action="store_true", help="run although the computed plan exceeds the budget"
    )
    p.add_argument(
        "--tokenizer-check-items", type=int, default=64, help="items per slice in the round-trip check"
    )
    return p


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.config is not None:
        overrides.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.seeds is not None:
        overrides["seeds"] = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if args.arms is not None:
        overrides["arms"] = parse_arms(args.arms)
    if args.variants is not None:
        overrides["variants"] = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    if args.updates is not None:
        overrides["updates"] = args.updates
    if args.sft_steps is not None:
        overrides["sft_steps"] = args.sft_steps
    return overrides


def _environment(label: str, cuda: Mapping[str, Any] | None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "label": label,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    if cuda is None:
        env.update({"device": "cpu", "measured_on": LABEL_DRY_RUN, "gpu": NOT_TESTED})
    else:
        env.update({**cuda, "measured_on": cuda["name"]})
    return env


def _config(factory: Any, overrides: Mapping[str, Any]) -> PilotConfig:
    """``factory(**overrides)`` with configuration errors turned into a :class:`P1Refusal`."""
    try:
        config: PilotConfig = factory(**overrides)
    except (TypeError, ValueError) as e:
        raise P1Refusal(f"invalid configuration: {e}") from e
    check_p1_config(config)
    return config


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, load and run (raises :class:`P1Refusal`); order: ``--model-dir`` checks, then CUDA,
    then loading, then :func:`run_p1`."""
    try:
        options = P1Options(
            model_dir=args.model_dir,
            dry_run=bool(args.dry_run),
            device=args.device,
            channel_layers=None if args.channel_layers is None else parse_layers(args.channel_layers),
            sft_scope=args.sft_scope or ("all" if args.dry_run else "ffn"),
            sft_layers=None if args.sft_layers is None else parse_layers(args.sft_layers),
            memory_fraction=args.memory_fraction,
            allow_over_budget=bool(args.allow_over_budget),
            tokenizer_check_items=args.tokenizer_check_items,
        )
        overrides = _overrides(args)
    except ValueError as e:
        raise P1Refusal(str(e)) from e
    if options.dry_run:
        dir_check = validate_model_dir(options.model_dir) if options.model_dir else None
        config = _config(dry_run_config, overrides)
        model, tokenizer = build_dry_run_model(config)
        model_info: dict[str, Any] = {
            "model_dir": None,
            "source": "in-process tiny Llama (research_rl.models.tiny), bf16, CPU",
            "model_dir_check": dir_check,
            "dtype": str(torch.bfloat16),
        }
        return run_p1(
            model,
            tokenizer,
            config,
            options,
            args.output_dir,
            environment=_environment(options.label, None),
            model_info=model_info,
            device_total_bytes=None,
        )
    validate_model_dir(options.model_dir)
    cuda = require_cuda(options.device)
    config = _config(gpu_config, overrides)
    assert options.model_dir is not None
    model, tokenizer, model_info = load_local_checkpoint(options.model_dir, device=options.device)
    return run_p1(
        model,
        tokenizer,
        config,
        options,
        args.output_dir,
        environment=_environment(options.label, cuda),
        model_info=model_info,
        device_total_bytes=int(cuda["total_memory_bytes"]),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the exit code (0 done, 2 NOT_TESTED / no CUDA, 3 refused)."""
    args = build_parser().parse_args(argv)
    try:
        results = run_from_args(args)
    except P1Refusal as e:
        print(
            json.dumps(
                {
                    "status": e.status,
                    "reason": e.reason,
                    "script_status": SCRIPT_STATUS,
                    "script_note": SCRIPT_NOTE,
                },
                indent=2,
            )
        )
        return e.exit_code
    verdicts = {
        k: v["verdict"] for k, v in results["criteria"].items() if isinstance(v, dict) and "verdict" in v
    }
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "label": results["label"],
                "measured_on": results["measured_on"],
                "script_status": results["script_status"],
                "banner": results["banner"],
                "criteria": verdicts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
