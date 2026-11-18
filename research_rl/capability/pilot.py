"""P0 capability pilot on CPU (``docs/research_rl/CAPABILITY.md`` 1.14 and the kill criteria of section 3).

**CPU tiny-model pilot (P0). Not evidence about 7-8B models; no GPU numbers.** Everything this module
produces comes from a randomly initialised tiny Llama trained in-process on synthetic digit-string
tasks; GPU metrics are never measured here and are written ``NOT_TESTED``.

Pipeline of :func:`run_pilot` (every random draw from an explicit seed):

1. **Testbed.** :class:`~research_rl.capability.tasks.OperatorSet` (``num_operators >= 10``) and
   :meth:`~research_rl.capability.tasks.CompositionSplits.build`; the split fingerprint is recorded.
   A tiny Llama is built from scratch (:func:`~research_rl.models.tiny.build_tiny_llama`, character
   tokenizer :func:`~research_rl.models.tiny.build_char_tokenizer`).
2. **Low start.** A plain PyTorch SFT loop (AdamW, loss on answer + EOS tokens only) on
   :func:`~research_rl.capability.tasks.sft_examples` (atomic operators, copy, the T_TASK
   prerequisites, optional demonstrations from TRAIN pairs only). The start checkpoint is measured
   with :class:`~research_rl.capability.probe.StartMap` (pass@1 greedy, pass@k sampled, per slice)
   and saved with safetensors (``start_model.safetensors``); every arm reloads it from that file.
3. **Arms** = ``operator x gate x variant x seed`` with operator in {``grpo`` (backprop control,
   :class:`~research_rl.capability.operators.TorchGRPOOperator`), ``es_full``, ``es_informed``,
   ``es_random`` (:class:`~research_rl.capability.es.ESOperator`)}, gate in {``gate_off``
   (``GateConfig(enabled=False)``: the ungated arm records ``would=<reason>``), ``gate_on``} and
   variant in {``leak`` (V1 = :class:`~research_rl.capability.tasks.SuffixLeakVerifier`, V2 exact),
   ``clean`` (V1 = V2 = :class:`~research_rl.capability.tasks.ExactAnswerVerifier`)}. Every arm runs
   :class:`~research_rl.capability.loop.GatedLoop` (one ``run(num_updates)`` call, so the loop's
   start-of-run horizon check covers the whole arm) with a
   :class:`~research_rl.capability.alarm.DisagreementAlarm` (``rollouts_per_unit=population`` for
   ES; GRPO reports prompt units) and a :class:`~research_rl.capability.tracker.CapabilityTracker`
   measuring A, R, T_rule, T_task and D(t) after every decision (tracker step = decision index + 1,
   step 0 = before the first update). The **final** weights are measured at the tracker step
   ``oracle_exclusion_steps(config)``, the same for every arm of the config (``final`` and
   ``final_step`` in the arm result; when updates follow the last decision this is also the last
   curve point, otherwise an extra measurement that is not added to the curve).
4. **Oracle** of the gate's error rates: before each decision the candidate and the incumbent
   (via :func:`~research_rl.capability.weights.swapped`) are scored with the exact verifier on the
   same large fresh T_RULE sample (purpose ``"oracle"``, index = decision index, drawn from the
   tracker pool, so it never coincides with gate items, minus every start-map uid and every tracker
   uid of steps ``0..oracle_exclusion_steps(config)``, a bound shared by every arm of the config, so
   arms of one seed score the same oracle items at the same decision index). ``oracle improved`` iff the unweighted mean over the T_RULE slices of
   ``candidate - incumbent`` exceeds :attr:`PilotConfig.oracle_threshold`. "The gate accepted" means
   the **first verdict** classifies as ``accept`` (:func:`~research_rl.capability.loop.classify_verdict`),
   identically for gated arms and for the ungated arm's ``would=`` record. False accept = accepted
   and not improved; false reject = not accepted and improved.
5. **Counts.** Hack rate = share of training rollouts with V1 pass and V2 fail. Operator calls are
   recorded as the operators report them (generation / training-forward calls); the Operator
   criterion uses *sequence-level* forward-pass equivalents: every generated sequence and every
   training-forward sequence is one forward, every backward sequence counts as two forwards.
6. **Outputs** in ``output_dir``: ``results.json``, ``summary.md`` (a table per arm and the automatic
   section-3 evaluation, :func:`evaluate_criteria`), ``decisions/<arm>.jsonl`` (one
   :class:`~research_rl.capability.types.DecisionRecord` per line), ``start_model.safetensors``,
   ``pilot_config.json``.

Criteria (:func:`evaluate_criteria`, a pure function of the results dict): every statistic is a
paired difference over seeds (arms of one seed share the start checkpoint and the sampler root
seed, so they score the same oracle items at every decision index, the same tracker items at every
step, and their ``final`` points on identical tracker items at the shared final step; intermediate
curve points share items only at equal steps, which alarm-shifted blocks can break), with a two-sided ``1 - criteria_alpha`` Student-t
interval (:func:`paired_interval`); fewer than three seeds, or a missing counterpart arm, gives
``INCONCLUSIVE`` - never a verdict. Every seed shares one split, one start checkpoint, one
start map and one informed channel selection; the seed varies only the RL stage (sampler root,
operator seed, the ``es_random`` draw), so the intervals cover RL-stage randomness only, conditional
on that start and selection, and a Subset verdict concerns this one informed selection. A Gate
SUCCESS needs both of its KILL conditions to be evaluable (at least ``min_improved_blocks``
oracle-improved blocks); a KILL condition the run cannot reach is stated in the reason and the
warnings (Operator: ``kill_reachable``). The operational definitions are written into the results.

Determinism: an arm is a function of ``(config, arm key, start weights)`` only. It runs with
``threads_per_arm`` torch threads whether arms run sequentially or in ``jobs`` spawned processes,
and arms are reported in canonical order, so ``--jobs`` does not change any recorded decision.

Expected time (default config, 48 arms, measured block times are printed as an ETA before the arms
start): see the docstring of :class:`PilotConfig`. Verified on CPU with tiny models only.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import math
import multiprocessing
import platform
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from research_rl.capability.alarm import DisagreementAlarm
from research_rl.capability.es import ESConfig, ESOperator, FitnessResult
from research_rl.capability.gate import CapabilityGate, GateConfig
from research_rl.capability.loop import (
    REASON_ACCEPT,
    GateController,
    GatedLoop,
    classify_verdict,
    worst_case_verdicts,
)
from research_rl.capability.operators import TorchGRPOOperator
from research_rl.capability.probe import (
    TRACK_POOL,
    CapabilityProbe,
    StartMap,
    draw_slice,
    sample_completions_with_truncation,
    score_items,
)
from research_rl.capability.scores import (
    ChannelStats,
    collect_channel_stats,
    contrast_scores,
    density_matched_random,
    usage_ratio_scores,
)
from research_rl.capability.spaces import ChannelGainSpace, ChannelWeightSpace, FullParameterSpace
from research_rl.capability.stats import student_t_quantile
from research_rl.capability.tasks import (
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    SuffixLeakVerifier,
    VerifierScoreFn,
    sft_examples,
)
from research_rl.capability.tracker import TRACK_PURPOSE, CapabilityTracker
from research_rl.capability.types import (
    DecisionRecord,
    GateDecision,
    Item,
    OperatorReport,
    RollbackUnit,
    Split,
    UpdateOperator,
)
from research_rl.capability.weights import StateSnapshot, swapped
from research_rl.models.ffn_channels import ChannelSelection
from research_rl.models.model_interface import count_parameters, find_swiglu_ffns, fingerprint
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama
from research_rl.rewards.verifier_interface import Verifier

__all__ = [
    "BANNER",
    "GATES",
    "OPERATORS",
    "ORACLE_PURPOSE",
    "VARIANTS",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_KILL",
    "VERDICT_SUCCESS",
    "ArmSpec",
    "PilotConfig",
    "arm_specs",
    "baseline_standard_error",
    "choose_retention_items",
    "evaluate_criteria",
    "gate_exercise",
    "interval_with_baseline_error",
    "oracle_exclusion_steps",
    "oracle_items",
    "paired_interval",
    "render_summary",
    "run_pilot",
    "smoke_config",
]

BANNER = "CPU tiny-model pilot (P0). Not evidence about 7-8B models; no GPU numbers."
NOT_TESTED = "NOT_TESTED"
OPERATORS = ("grpo", "es_full", "es_informed", "es_random")
GATES = ("gate_off", "gate_on")
VARIANTS = ("leak", "clean")
ORACLE_PURPOSE = "oracle"
"""``ItemSampler`` purpose of the oracle items (index = decision index)."""
ES_PURPOSE = "es"
"""``ItemSampler`` purpose of the ES fitness items (index = update index)."""
SELECT_PURPOSE = "select"
"""``ItemSampler`` purpose of the channel-selection texts (index 0, TRAIN and RETENTION only)."""
VERDICT_SUCCESS = "SUCCESS"
VERDICT_KILL = "KILL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
_RESULTS_VERSION = 1


# ============================================================================== seeds and threads
def _seed(*parts: object) -> int:
    """Deterministic 63-bit seed from ``parts`` (SHA-256 of their ``repr``)."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


@contextlib.contextmanager
def _torch_threads(n: int) -> Iterator[None]:
    """Run the block with ``n`` torch intra-op threads; the previous setting is restored."""
    previous = torch.get_num_threads()
    torch.set_num_threads(int(n))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# ============================================================================== configuration
@dataclass
class PilotConfig:
    """Every knob of the pilot (JSON-able via :meth:`to_dict` / :meth:`from_dict`).

    Defaults are the full P0 configuration: 4 operators x 2 gates x 2 variants x 3 seeds = 48 arms
    of ``updates=48`` updates each (6 decisions of ``block_updates=8``), a 3-layer tiny Llama
    (hidden 64, FFN 256, 196800 parameters) trained by SFT for 4000 steps. Measured once on the
    development container (4 arms in parallel, one torch thread each, 2048 retention items per
    slice): SFT about 3 minutes on 2 threads, start map about 20 s, one block (8 updates plus one
    gate verdict, oracle and tracker) about 11 s for every operator, dominated by the gate's
    13312 evidence generations. A 48-update arm therefore takes about 1-2 minutes (more when
    SHRINK re-tests or alarm decisions add verdicts), so the 48 arms take roughly 20-40 minutes
    with ``--jobs 4`` on 4 cores and roughly 1.5-2.5 hours sequentially (extrapolated from the
    measured block time, not a measured full run). Other machines differ: :func:`run_pilot` times
    one block of every operator first and prints the measured ETA (``eta_probe``), which is
    recorded in ``results.json``.

    **What the default run can and cannot show (measured on CPU, tiny models only).** On four
    seed-0 default arms (three GRPO, one ES) no block was oracle-improved (largest oracle diff
    0.003 against ``oracle_threshold=0.01``) and no first verdict accepted, so the false-reject
    rate is not evaluable and the false-accept rate is 0 by construction. Ungated GRPO from the
    default start gained about 0.01 mean T_rule in 300 updates at ``grpo_lr=2e-4``; at 1e-3
    T_rule gained less than 0.01 in 80 updates while ``retention/atomic`` fell from 0.93 to 0.57
    (leak variant) and at 3e-3 the model collapsed. The per-block T_rule effect is far below what
    the oracle or the gate's transfer test resolves, so the default pilot tests the gate's
    *rejections* and the acquisition limit, not its error rates. The run says so itself:
    :func:`gate_exercise` is written to ``results["gate_exercise"]``, to the warnings, to the gate
    criterion's reason and to ``summary.md``. Without ``min_improved_blocks`` oracle-improved blocks
    the Gate criterion's reject-rate KILL is not evaluable, so its verdict is ``INCONCLUSIVE`` (or the
    clean-cost KILL), never ``SUCCESS``: a gate that rejects every block cannot pass. The testbed is
    not weakened to manufacture gains (section 3: report an acquisition-limited task).

    **Operator KILL at the defaults.** GRPO spends ``4 x grpo_prompts_per_update x grpo_group_size``
    = 512 FPE per update (24 576 over 48 updates; budget ``operator_budget_factor`` x that = 245 760),
    ES ``es_population x es_items_per_update`` = 256 per update (12 288 over ``es_updates=None`` = 48,
    5 % of the budget). The KILL needs ES to spend the whole budget, so at the defaults the Operator
    criterion can only be ``SUCCESS`` or ``INCONCLUSIVE``; the criterion's ``kill_reachable`` statistic,
    its reason and the run warnings say so. ``es_updates >= 960`` spends the budget
    (:func:`choose_retention_items` then picks 3072 retention items per slice); that is 20x the ES
    arms' updates (configuration not run; time extrapolated, not measured), so it is not the default.

    Groups of fields:

    * testbed: ``seed`` (root of every derived seed), ``num_operators`` (>= 10), ``depth3_count``,
      ``leak_k`` (suffix length of the planted V1 leak);
    * model: ``hidden_size``, ``intermediate_size``, ``num_hidden_layers``, ``num_attention_heads``,
      ``num_key_value_heads``;
    * SFT: ``sft_steps``, ``sft_batch_size``, ``sft_lr``, ``sft_examples``, ``sft_demo_pairs``
      (composed demonstrations from TRAIN pairs only; 0 disables);
    * start map: ``start_items_per_slice``, ``start_k``, ``start_num_samples`` (>= ``start_k``);
    * arms: ``seeds``, ``arms`` (``"operator+gate"`` strings; default the full cross product),
      ``variants``, ``updates`` (GRPO and default ES updates), ``es_updates`` (``None`` = ``updates``);
    * gate: ``gate`` (overrides of :class:`~research_rl.capability.gate.GateConfig` fields;
      ``enabled`` is set per arm, ``rollback_unit`` is always ``parameters``), ``transfer_slices`` /
      ``retention_slices`` (``None`` = every T_RULE + T_TASK / RETENTION slice),
      ``retention_items_candidates`` (searched for the smallest ``items_per_retention_slice`` whose
      horizon covers the worst-case verdict count, unless ``gate`` fixes it),
      ``allow_short_horizon``;
    * alarm: ``alarm`` (overrides of :class:`~research_rl.capability.alarm.DisagreementAlarm`
      keyword arguments; ``rollouts_per_unit`` is set per operator);
    * GRPO: ``grpo_prompts_per_update``, ``grpo_group_size``, ``grpo_lr``, ``grpo_temperature``,
      ``grpo_screen`` (default False: the gate, not rollout screening, is the defence under test);
    * ES: ``es_population``, ``es_items_per_update``, ``es_sigma``, ``es_lr``, ``es_space``
      (``"gain"`` = ChannelGainSpace, ``"weight"`` = ChannelWeightSpace, for the subset arms),
      ``es_full_space`` (``"ffn_weights"`` = FullParameterSpace over every FFN weight,
      ``"all_channels"`` = the subset space over every channel), ``es_full_sigma``, ``es_full_lr``;
    * selection: ``informed_k``, ``informed_scorer`` (``"auto"``: contrast if enough V2-correct and
      incorrect samples of the same prompts exist, else usage ratio; or ``"contrast"`` /
      ``"usage_ratio"``), ``scorer_prompts``, ``scorer_samples``, ``scorer_min_texts``,
      ``usage_threshold`` (``None`` = mean ``|h|`` on the retention texts);
    * measurement: ``max_new_tokens`` (``None`` = ``max_len + 1``), ``eval_batch_size``,
      ``track_items_per_slice``, ``oracle_items_per_slice``, ``oracle_threshold``;
    * criteria: ``criteria_alpha``, ``equal_a_tolerance``, ``operator_budget_factor``,
      ``operator_gain_fraction``, ``operator_es_arm``, ``min_improved_blocks``;
    * execution: ``threads`` (main process: SFT, start map, selection), ``threads_per_arm``,
      ``jobs``, ``eta_probe``, ``smoke``, ``progress``.
    """

    seed: int = 0
    num_operators: int = 10
    depth3_count: int = 16
    leak_k: int = 2
    hidden_size: int = 64
    intermediate_size: int = 256
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    sft_steps: int = 4000
    sft_batch_size: int = 64
    sft_lr: float = 3e-3
    sft_examples: int = 20000
    sft_demo_pairs: int = 2000
    start_items_per_slice: int = 128
    start_k: int = 16
    start_num_samples: int = 16
    seeds: tuple[int, ...] = (0, 1, 2)
    arms: tuple[str, ...] = tuple(f"{op}+{g}" for op in OPERATORS for g in GATES)
    variants: tuple[str, ...] = VARIANTS
    updates: int = 48
    es_updates: int | None = None
    gate: dict[str, Any] = field(
        default_factory=lambda: {"block_updates": 8, "items_per_transfer_slice": 128}
    )
    transfer_slices: tuple[str, ...] | None = None
    retention_slices: tuple[str, ...] | None = None
    retention_items_candidates: tuple[int, ...] = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
    allow_short_horizon: bool = False
    alarm: dict[str, Any] = field(default_factory=dict)
    grpo_prompts_per_update: int = 32
    grpo_group_size: int = 4
    grpo_lr: float = 2e-4
    grpo_temperature: float = 1.0
    grpo_screen: bool = False
    es_population: int = 8
    es_items_per_update: int = 32
    es_sigma: float = 0.05
    es_lr: float = 0.005
    es_space: str = "gain"
    es_full_space: str = "ffn_weights"
    es_full_sigma: float = 0.002
    es_full_lr: float = 8e-6
    informed_k: int = 64
    informed_scorer: str = "auto"
    scorer_prompts: int = 64
    scorer_samples: int = 8
    scorer_min_texts: int = 8
    usage_threshold: float | None = None
    max_new_tokens: int | None = None
    eval_batch_size: int = 256
    track_items_per_slice: int = 128
    oracle_items_per_slice: int = 512
    oracle_threshold: float = 0.01
    criteria_alpha: float = 0.05
    equal_a_tolerance: float = 0.02
    operator_budget_factor: float = 10.0
    operator_gain_fraction: float = 0.5
    operator_es_arm: str = "es_informed"
    min_improved_blocks: int = 5
    threads: int = 2
    threads_per_arm: int = 1
    jobs: int = 1
    eta_probe: bool = True
    smoke: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        self.seeds = tuple(int(s) for s in self.seeds)
        self.arms = tuple(str(a) for a in self.arms)
        self.variants = tuple(str(v) for v in self.variants)
        self.retention_items_candidates = tuple(int(n) for n in self.retention_items_candidates)
        if self.transfer_slices is not None:
            self.transfer_slices = tuple(self.transfer_slices)
        if self.retention_slices is not None:
            self.retention_slices = tuple(self.retention_slices)
        self.gate = dict(self.gate)
        self.alarm = dict(self.alarm)
        if self.num_operators < 10:
            raise ValueError(f"num_operators must be >= 10 for the pilot, got {self.num_operators}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be non-empty and distinct, got {self.seeds}")
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError(f"arms must be non-empty and distinct, got {self.arms}")
        for arm in self.arms:
            parse_arm(arm)
        if (
            not self.variants
            or any(v not in VARIANTS for v in self.variants)
            or len(set(self.variants)) != len(self.variants)
        ):
            raise ValueError(f"variants must be distinct values of {VARIANTS}, got {self.variants}")
        for name in (
            "updates",
            "sft_batch_size",
            "start_items_per_slice",
            "start_k",
            "grpo_prompts_per_update",
            "grpo_group_size",
            "es_items_per_update",
            "informed_k",
            "scorer_prompts",
            "scorer_samples",
            "scorer_min_texts",
            "eval_batch_size",
            "track_items_per_slice",
            "oracle_items_per_slice",
            "threads",
            "threads_per_arm",
            "jobs",
            "min_improved_blocks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        for name in ("sft_steps", "sft_examples", "sft_demo_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an int >= 0, got {value!r}")
        if self.sft_examples + self.sft_demo_pairs < 1 and self.sft_steps > 0:
            raise ValueError("SFT needs at least one example")
        if self.es_updates is not None and (isinstance(self.es_updates, bool) or self.es_updates < 1):
            raise ValueError(f"es_updates must be None or an int >= 1, got {self.es_updates!r}")
        if self.start_num_samples < self.start_k:
            raise ValueError("start_num_samples must be >= start_k")
        if self.es_population < 2 or self.es_population % 2:
            raise ValueError(f"es_population must be an even int >= 2, got {self.es_population}")
        if self.es_space not in ("gain", "weight"):
            raise ValueError(f"es_space must be 'gain' or 'weight', got {self.es_space!r}")
        if self.es_full_space not in ("ffn_weights", "all_channels"):
            raise ValueError(
                f"es_full_space must be 'ffn_weights' or 'all_channels', got {self.es_full_space!r}"
            )
        if self.informed_scorer not in ("auto", "contrast", "usage_ratio"):
            raise ValueError(
                f"informed_scorer must be auto, contrast or usage_ratio, got {self.informed_scorer!r}"
            )
        if self.operator_es_arm not in ("es_full", "es_informed", "es_random"):
            raise ValueError(f"operator_es_arm must be an ES operator, got {self.operator_es_arm!r}")
        if not 0.0 < self.criteria_alpha < 1.0:
            raise ValueError(f"criteria_alpha must be in (0, 1), got {self.criteria_alpha}")
        if self.max_new_tokens is not None and self.max_new_tokens < 2:
            raise ValueError(f"max_new_tokens must be None or >= 2, got {self.max_new_tokens}")
        unknown = sorted(set(self.gate) - {f.name for f in dataclasses.fields(GateConfig)})
        if unknown:
            raise ValueError(f"unknown gate keys {unknown}")
        for fixed in ("enabled", "rollback_unit"):
            if fixed in self.gate:
                raise ValueError(f"gate[{fixed!r}] is set per arm by the pilot and may not be overridden")
        if "rollouts_per_unit" in self.alarm:
            raise ValueError("alarm['rollouts_per_unit'] is set per operator by the pilot")
        self.base_gate_config()  # validates the overrides

    # ---------------------------------------------------------------- helpers
    @property
    def es_num_updates(self) -> int:
        return self.updates if self.es_updates is None else int(self.es_updates)

    def updates_for(self, operator: str) -> int:
        """Updates of an arm with ``operator``."""
        return self.updates if operator == "grpo" else self.es_num_updates

    def base_gate_config(self, *, items_per_retention_slice: int | None = None) -> GateConfig:
        """``GateConfig`` with the overrides, ``rollback_unit=parameters`` and (if given) the
        retention item count; ``enabled`` is left at its default (set per arm)."""
        kwargs = dict(self.gate)
        if items_per_retention_slice is not None:
            kwargs["items_per_retention_slice"] = int(items_per_retention_slice)
        if isinstance(kwargs.get("shrink_lambdas"), list):
            kwargs["shrink_lambdas"] = tuple(kwargs["shrink_lambdas"])
        return GateConfig(rollback_unit=RollbackUnit.PARAMETERS, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        for key, value in out.items():
            if isinstance(value, tuple):
                out[key] = list(value)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PilotConfig:
        """Inverse of :meth:`to_dict`; unknown keys raise ``ValueError``."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown PilotConfig keys: {unknown}")
        kwargs: dict[str, Any] = dict(data)
        for key in (
            "seeds",
            "arms",
            "variants",
            "retention_items_candidates",
            "transfer_slices",
            "retention_slices",
        ):
            if isinstance(kwargs.get(key), list):
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


def smoke_config(**overrides: Any) -> PilotConfig:
    """The ``--smoke`` configuration: 1 seed, 2 arms (``grpo+gate_on``, ``es_informed+gate_off``) on
    the leak variant, 4 updates in blocks of 2, a 2-layer hidden-32 model, tiny item counts (16
    retention items per slice with a 0.5 retention margin, so verdicts can pass at all) and
    ``allow_short_horizon=True`` (a short horizon only warns). Runs in well under a minute on 2
    threads (about 13 s measured). Smoke numbers are plumbing checks only, never results.

    The smoke arms must actually train, or a determinism check on them is vacuous: with a
    one-character leak (``leak_k=1``) and 800 SFT steps some training rollouts pass V1 (all of
    them hacks: the smoke model solves no TRAIN pair; 16 of 128 GRPO rollouts measured), so GRPO
    groups (8 prompts x 4 samples per update) have non-zero advantages,
    and ES uses ``es_population=4`` / ``es_sigma=0.5`` so that members' greedy outputs (and hence
    fitness) differ. Both arms then change the weights (``final_fingerprint`` differs from the
    start and depends on the learning rates)."""
    base: dict[str, Any] = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "leak_k": 1,
        "sft_steps": 800,
        "sft_batch_size": 32,
        "sft_examples": 400,
        "sft_demo_pairs": 40,
        "start_items_per_slice": 3,
        "start_k": 2,
        "start_num_samples": 2,
        "seeds": (0,),
        "arms": ("grpo+gate_on", "es_informed+gate_off"),
        "variants": ("leak",),
        "updates": 4,
        "gate": {
            "block_updates": 2,
            "items_per_transfer_slice": 8,
            "items_per_retention_slice": 16,
            "retention_margin": 0.5,
        },
        "allow_short_horizon": True,
        "alarm": {"min_rollouts": 4},
        "grpo_prompts_per_update": 8,
        "grpo_group_size": 4,
        "es_population": 4,
        "es_items_per_update": 8,
        "es_sigma": 0.5,
        "es_lr": 0.05,
        "informed_k": 8,
        "scorer_prompts": 4,
        "scorer_samples": 2,
        "scorer_min_texts": 2,
        "eval_batch_size": 64,
        "track_items_per_slice": 4,
        "oracle_items_per_slice": 12,
        "min_improved_blocks": 1,
        "threads": 1,
        "threads_per_arm": 1,
        "eta_probe": False,
        "smoke": True,
    }
    base.update(overrides)
    return PilotConfig(**base)


# ============================================================================== arms
def parse_arm(arm: str) -> tuple[str, str]:
    """``"operator+gate"`` -> ``(operator, gate)`` (``ValueError`` for unknown parts)."""
    parts = arm.split("+")
    if len(parts) != 2 or parts[0] not in OPERATORS or parts[1] not in GATES:
        raise ValueError(
            f"arm {arm!r} must be '<operator>+<gate>' with operator in {OPERATORS}, gate in {GATES}"
        )
    return parts[0], parts[1]


@dataclass(frozen=True)
class ArmSpec:
    """One arm: ``operator``, ``gate``, ``variant`` and ``seed``."""

    operator: str
    gate: str
    variant: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.operator}/{self.gate}/{self.variant}/s{self.seed}"

    @property
    def file_stem(self) -> str:
        return self.key.replace("/", "__")

    def to_dict(self) -> dict[str, Any]:
        return {"operator": self.operator, "gate": self.gate, "variant": self.variant, "seed": self.seed}


def arm_specs(config: PilotConfig) -> list[ArmSpec]:
    """Every arm of ``config`` in canonical order (variant, seed, operator, gate)."""
    combos = [parse_arm(a) for a in config.arms]
    combos.sort(key=lambda c: (OPERATORS.index(c[0]), GATES.index(c[1])))
    variants = sorted(config.variants, key=VARIANTS.index)
    return [ArmSpec(op, g, v, s) for v in variants for s in sorted(config.seeds) for op, g in combos]


# ============================================================================== shared context
def build_splits(config: PilotConfig) -> CompositionSplits:
    """The pilot's operator set and rule-level splits (deterministic in ``config.seed``)."""
    operators = OperatorSet(seed=config.seed, num_operators=config.num_operators)
    return CompositionSplits.build(operators, seed=config.seed, depth3_count=config.depth3_count)


def build_model(config: PilotConfig, vocab_size: int) -> nn.Module:
    """A fresh tiny Llama with ``config``'s shape (seeded by ``config.seed``)."""
    return build_tiny_llama(
        vocab_size,
        seed=config.seed,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
    )


def _budget(config: PilotConfig, splits: CompositionSplits) -> int:
    need = splits.operators.max_len + 1
    budget = need if config.max_new_tokens is None else int(config.max_new_tokens)
    if budget < need:
        raise ValueError(f"max_new_tokens={budget} violates the truncation contract (need >= {need})")
    return budget


def _verifiers(config: PilotConfig, variant: str) -> tuple[Verifier, Verifier]:
    v2 = ExactAnswerVerifier()
    if variant == "leak":
        return SuffixLeakVerifier(k=config.leak_k), v2
    if variant == "clean":
        return ExactAnswerVerifier(), v2
    raise ValueError(f"unknown variant {variant!r}")


def _sampler_root(config: PilotConfig, seed: int) -> int:
    return _seed("pilot_sampler", config.seed, seed) % (2**31)


def _slices(config: PilotConfig, splits: CompositionSplits) -> tuple[tuple[str, ...], tuple[str, ...]]:
    transfer = (
        config.transfer_slices
        if config.transfer_slices is not None
        else (*splits.slices_of(Split.T_RULE), *splits.slices_of(Split.T_TASK))
    )
    retention = (
        config.retention_slices if config.retention_slices is not None else splits.slices_of(Split.RETENTION)
    )
    return tuple(transfer), tuple(retention)


class _CountingScoreFn(VerifierScoreFn):
    """:class:`VerifierScoreFn` that counts scored items (evaluation generations)."""

    def __init__(self, tokenizer: Any, verifier: Verifier, *, max_new_tokens: int, batch_size: int) -> None:
        super().__init__(tokenizer, verifier, max_new_tokens=max_new_tokens, batch_size=batch_size)
        self.items_scored = 0

    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
        self.items_scored += len(items)
        return super().__call__(model, items)


# ============================================================================== SFT
def _encode_example(tokenizer: Any, item: Item) -> tuple[list[int], list[int]]:
    prompt = list(tokenizer(item.prompt, add_special_tokens=False)["input_ids"])
    answer = [*tokenizer(item.answer, add_special_tokens=False)["input_ids"], int(tokenizer.eos_token_id)]
    return prompt + answer, [-100] * len(prompt) + answer


def train_sft(model: nn.Module, tokenizer: Any, items: Sequence[Item], config: PilotConfig) -> dict[str, Any]:
    """Plain PyTorch SFT: ``sft_steps`` AdamW steps (lr ``sft_lr``, no weight decay) on batches of
    ``sft_batch_size`` examples drawn with replacement from a seeded generator; right-padded batches
    with an attention mask; the loss covers the answer tokens and the EOS only. Returns the loss
    curve (every 50 steps and the last step) and the step count. The model is left in eval mode."""
    if config.sft_steps == 0 or not items:
        model.eval()
        return {"steps": 0, "loss_curve": []}
    encoded = [_encode_example(tokenizer, it) for it in items]
    pad = int(tokenizer.pad_token_id)
    gen = torch.Generator().manual_seed(_seed("sft_batches", config.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft_lr, weight_decay=0.0)
    curve: list[dict[str, float]] = []
    model.train()
    for step in range(config.sft_steps):
        idx = torch.randint(len(encoded), (config.sft_batch_size,), generator=gen).tolist()
        rows = [encoded[i] for i in idx]
        length = max(len(r[0]) for r in rows)
        ids = torch.tensor([r[0] + [pad] * (length - len(r[0])) for r in rows], dtype=torch.long)
        labels = torch.tensor([r[1] + [-100] * (length - len(r[1])) for r in rows], dtype=torch.long)
        mask = torch.tensor([[1] * len(r[0]) + [0] * (length - len(r[0])) for r in rows], dtype=torch.long)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == config.sft_steps - 1:
            curve.append({"step": step, "loss": float(loss.detach())})
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"steps": config.sft_steps, "loss_curve": curve}


# ============================================================================== channel selection
def _ffn_weight_names(model: nn.Module) -> list[str]:
    return [
        f"{ref.module_path}.{proj}.weight"
        for ref in find_swiglu_ffns(model)
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]


def select_informed_channels(
    model: nn.Module, tokenizer: Any, sampler: ItemSampler, config: PilotConfig, max_new_tokens: int
) -> dict[str, Any]:
    """Forward-only informed channel selection on the start model (TRAIN and RETENTION texts only,
    purpose ``"select"``; never T_RULE / T_TASK / AUDIT).

    * ``contrast``: ``scorer_samples`` sampled completions of ``scorer_prompts`` TRAIN prompts,
      scored with V2 (exact); only prompts with at least one correct and one incorrect sample are
      kept (so the contrast compares rollouts of the same prompts); texts = prompt + completion;
      :func:`~research_rl.capability.scores.contrast_scores`. Needs ``scorer_min_texts`` texts in
      both groups.
    * ``usage_ratio``: texts = prompt + greedy completion of ``scorer_prompts`` TRAIN prompts versus
      the same number of retention prompts (spread over the retention slices);
      :func:`~research_rl.capability.scores.usage_ratio_scores` at ``usage_threshold`` (default: the
      mean ``|h|`` over every channel on the retention texts).

    ``"auto"`` uses contrast when it is feasible, else usage ratio, and records why. Returns a
    JSON-able dict with the scorer, the reason, the selection (``top_k_from_scores``, ``informed_k``
    channels) and counts.
    """
    v2_fn = VerifierScoreFn(
        tokenizer, ExactAnswerVerifier(), max_new_tokens=max_new_tokens, batch_size=config.eval_batch_size
    )
    prompts = sampler.draw(Split.TRAIN, config.scorer_prompts, purpose=SELECT_PURPOSE, index=0)
    info: dict[str, Any] = {"k": config.informed_k, "requested": config.informed_scorer}
    scores: dict[int, torch.Tensor] | None = None
    if config.informed_scorer in ("auto", "contrast"):
        correct: list[str] = []
        incorrect: list[str] = []
        for item in prompts:
            texts, raw = sample_completions_with_truncation(
                model,
                tokenizer,
                item.prompt,
                num_samples=config.scorer_samples,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                seed=_seed("select_samples", config.seed, item.uid),
            )
            flags = [a or b for a, b in zip(raw, v2_fn.truncated(texts))]
            s = v2_fn.score_completions(texts, [item] * len(texts), flags)
            if 0.0 < sum(s) < len(s):
                for text, value in zip(texts, s):
                    (correct if value == 1.0 else incorrect).append(item.prompt + text)
        info["contrast_texts"] = {"correct": len(correct), "incorrect": len(incorrect)}
        if len(correct) >= config.scorer_min_texts and len(incorrect) >= config.scorer_min_texts:
            sc = collect_channel_stats(model, tokenizer, correct, batch_size=config.eval_batch_size)
            si = collect_channel_stats(model, tokenizer, incorrect, batch_size=config.eval_batch_size)
            scores = contrast_scores(sc, si)
            info["scorer"] = "contrast"
            info["reason"] = "enough V2-correct and V2-incorrect samples of the same prompts"
        elif config.informed_scorer == "contrast":
            raise ValueError(
                f"contrast scorer infeasible: {len(correct)} correct / {len(incorrect)} incorrect texts "
                f"(need {config.scorer_min_texts} each); use informed_scorer='auto' or 'usage_ratio'"
            )
        else:
            info["reason"] = (
                f"contrast infeasible at this start ({len(correct)} correct / {len(incorrect)} incorrect texts "
                f"from prompts with mixed outcomes, need {config.scorer_min_texts} each): fell back to usage_ratio"
            )
    if scores is None:
        train_texts = [it.prompt + c for it, c in zip(prompts, v2_fn.generate(model, prompts))]
        retention = sampler.draw(Split.RETENTION, config.scorer_prompts, purpose=SELECT_PURPOSE, index=0)
        retention_texts = [it.prompt + c for it, c in zip(retention, v2_fn.generate(model, retention))]
        threshold = config.usage_threshold
        if threshold is None:
            probe_stats: ChannelStats = collect_channel_stats(
                model, tokenizer, retention_texts, batch_size=config.eval_batch_size
            )
            values = torch.cat([probe_stats.mean_abs[layer].double() for layer in probe_stats.layers()])
            threshold = float(values.mean())
        st = collect_channel_stats(
            model, tokenizer, train_texts, batch_size=config.eval_batch_size, threshold=threshold
        )
        sr = collect_channel_stats(
            model, tokenizer, retention_texts, batch_size=config.eval_batch_size, threshold=threshold
        )
        scores = usage_ratio_scores(st, sr)
        info["scorer"] = "usage_ratio"
        info["usage_threshold"] = threshold
        info.setdefault("reason", "requested")
    selection = ChannelSelection.top_k_from_scores(scores, config.informed_k)
    info["selection"] = {str(k): list(v) for k, v in selection.to_dict().items()}
    info["count"] = selection.count()
    return info


def _selection_from(info: Mapping[str, Any]) -> ChannelSelection:
    return ChannelSelection.from_dict({str(k): v for k, v in info["selection"].items()})


# ============================================================================== gate sizing
def choose_retention_items(config: PilotConfig, retention_slices: Sequence[str]) -> dict[str, Any]:
    """Pick ``items_per_retention_slice`` so that :class:`~research_rl.capability.loop.GatedLoop`'s
    start-of-run horizon check passes for the longest arm.

    Worst case per arm: :func:`~research_rl.capability.loop.worst_case_verdicts` with the gated
    procedure's ``1 + len(shrink_lambdas)`` verdicts per decision (``RollbackUnit.PARAMETERS``: no
    group stage) and one verdict per non-block update while ``alarm_triggers_decision``. The horizon
    of a candidate ``n`` is the minimum over the retention slices of
    ``GateConfig.retention_horizon(slice, n=n, n10=round(0.005 n), n01=0)`` (the loop's rule). The
    smallest candidate of ``retention_items_candidates`` that covers the worst case is chosen; a
    value fixed in ``config.gate`` is checked instead. Returns the choice and its justification
    (``short_horizon`` True if nothing covers it; the run then needs ``allow_short_horizon``).
    """
    base = config.base_gate_config()
    num_updates = max(config.updates_for(parse_arm(a)[0]) for a in config.arms)
    need = worst_case_verdicts(
        num_updates,
        block_updates=base.block_updates,
        verdicts_per_decision=1 + len(base.shrink_lambdas),
        alarm_triggers_decision=base.alarm_triggers_decision,
    )

    def horizon(n: int) -> int | None:
        best: int | None = None
        for name in retention_slices:
            k = base.retention_horizon(name, n=n, n10=round(0.005 * n), n01=0)
            if k is not None and (best is None or k < best):
                best = k
        return best

    fixed = "items_per_retention_slice" in config.gate
    candidates = (
        [int(config.gate["items_per_retention_slice"])]
        if fixed
        else sorted(config.retention_items_candidates)
    )
    table = []
    chosen: int | None = None
    for n in candidates:
        h = horizon(n)
        ok = h is None or h >= need
        table.append({"n": n, "horizon": h, "covers": ok})
        if ok and chosen is None:
            chosen = n
    short = chosen is None
    if chosen is None:
        chosen = candidates[-1]
    return {
        "items_per_retention_slice": chosen,
        "fixed_by_config": fixed,
        "worst_case_verdicts": need,
        "num_updates": num_updates,
        "verdicts_per_decision": 1 + len(base.shrink_lambdas),
        "rule": "horizon = min over retention slices of GateConfig.retention_horizon(n, n10=round(0.005 n), n01=0)",
        "candidates": table,
        "short_horizon": short,
    }


# ============================================================================== oracle
def _rule_slices(splits: CompositionSplits) -> tuple[str, ...]:
    return splits.slices_of(Split.T_RULE)


def excluded_uids(
    sampler: ItemSampler, start: StartMap, *, track_items_per_slice: int, max_step: int
) -> set[str]:
    """Every T_RULE uid the start map or the tracker (steps ``0..max_step``) of ``sampler`` uses."""
    out: set[str] = set()
    rule = _rule_slices(sampler.splits)
    for name, s in start.slices.items():
        if name in rule:
            out.update(s.uids)
    for step in range(max_step + 1):
        for name in rule:
            items = draw_slice(
                sampler, name, track_items_per_slice, purpose=TRACK_PURPOSE, index=step, pool=TRACK_POOL
            )
            out.update(it.uid for it in items)
    return out


def oracle_exclusion_steps(config: PilotConfig, num_updates: int | None = None) -> int:
    """Highest tracker step whose items the oracle excludes, the same for every arm of ``config``.

    A tracker step is at most the arm's update count (one point per decision, at most one decision
    per update, plus at most one trailing point). The bound is the maximum update count over every
    operator in ``config.arms`` (``updates`` / ``es_updates``), so GRPO and ES arms of one seed, and
    a shortened run of the same config (the ETA probe), draw identical oracle items at every decision
    index. ``num_updates`` (an arm run with an explicit update count) only raises the bound when it
    exceeds the config's maximum, so the exclusion still covers every step that arm measures.
    """
    bound = max(config.updates_for(parse_arm(a)[0]) for a in config.arms)
    if num_updates is not None:
        bound = max(bound, int(num_updates))
    return bound


def _measure_final(
    tracker: CapabilityTracker,
    model: nn.Module,
    reports: Sequence[Any],
    records: Sequence[DecisionRecord],
    *,
    trailing: int,
    final_step: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """``(trailing_point, final_point)`` of an arm's final weights, both measured at tracker step
    ``final_step`` (:func:`oracle_exclusion_steps` of the config: the same step for every arm, so
    same-seed arms' final points are scored on identical tracker items, whatever their decision
    counts). With ``trailing > 0`` updates after the last decision, the final weights get one curve
    point (reward = mean V1 reward of those updates) and it is the final point; otherwise the last
    decision point is already on the final weights but at a decision-count-dependent step, so they
    are measured once more (reward = the last decision's ``reward_v1``), not added to the curve."""
    if trailing > 0:
        tail = reports[-trailing:]
        reward = sum(float(r.reward_v1) for r in tail) / len(tail)
        point = tracker.measure(model, final_step, reward).to_dict()
        return point, point
    reward_last = records[-1].reward_v1 if records else None
    return None, tracker.measure(model, final_step, reward_last).to_dict()


def oracle_items(
    sampler: ItemSampler, n: int, decision_index: int, exclude: set[str]
) -> dict[str, list[Item]]:
    """``n`` oracle items per T_RULE slice for one decision: purpose :data:`ORACLE_PURPOSE`, index =
    ``decision_index``, from the tracker pool (so disjoint from every gate item by construction),
    without any uid in ``exclude`` (start-map and tracker items). Deterministic in the key."""
    out: dict[str, list[Item]] = {}
    for name in _rule_slices(sampler.splits):
        size = n + 32
        while True:
            items = draw_slice(
                sampler, name, size, purpose=ORACLE_PURPOSE, index=decision_index, pool=TRACK_POOL
            )
            kept = [it for it in items if it.uid not in exclude]
            if len(kept) >= n:
                out[name] = kept[:n]
                break
            size *= 2
    return out


# ============================================================================== arm monitoring
class _RecordingOperator:
    """Delegating :class:`~research_rl.capability.types.UpdateOperator` that keeps every report."""

    def __init__(self, inner: UpdateOperator, *, kind: str) -> None:
        self.inner = inner
        self.name = inner.name
        self.optimizer = inner.optimizer
        self.kind = kind
        self.reports: list[OperatorReport] = []

    def parameter_names(self) -> list[str]:
        return self.inner.parameter_names()

    def step(self, model: nn.Module, update_index: int) -> OperatorReport:
        report = self.inner.step(model, update_index)
        self.reports.append(report)
        return report

    def on_decision(self, decision: GateDecision) -> None:
        self.inner.on_decision(decision)

    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.inner.load_state_dict(state)


def _report_counts(report: OperatorReport, kind: str) -> dict[str, int]:
    n = len(report.v1_pass)
    hacks = sum(1 for a, b in zip(report.v1_pass, report.v2_pass) if a and not b)
    if kind == "grpo":
        fwd_seq, bwd_seq = 2 * n, n  # generated sequences + the training forward of the same rows
    else:
        fwd_seq, bwd_seq = n, 0
    return {
        "rollouts": n,
        "hacks": hacks,
        "v1_pass": sum(report.v1_pass),
        "v2_pass": sum(report.v2_pass),
        "forward_calls": report.forward_passes,
        "backward_calls": report.backward_passes,
        "forward_sequences": fwd_seq,
        "backward_sequences": bwd_seq,
        "fpe": fwd_seq + 2 * bwd_seq,
    }


def _sum_counts(rows: Sequence[Mapping[str, int]]) -> dict[str, int]:
    keys = ("rollouts", "hacks", "v1_pass", "v2_pass", "forward_calls", "backward_calls")
    out = {k: sum(int(r[k]) for r in rows) for k in keys}
    out.update({k: sum(int(r[k]) for r in rows) for k in ("forward_sequences", "backward_sequences", "fpe")})
    return out


class _ArmMonitor:
    """Oracle before and tracker after every decision of one arm; ETA printing."""

    def __init__(
        self,
        *,
        model: nn.Module,
        operator: _RecordingOperator,
        tracker: CapabilityTracker,
        oracle_fn: VerifierScoreFn,
        sampler: ItemSampler,
        config: PilotConfig,
        exclude: set[str],
        label: str,
        total_blocks: int,
        require_transfer_gain: bool,
    ) -> None:
        self.model = model
        self.operator = operator
        self.tracker = tracker
        self.oracle_fn = oracle_fn
        self.sampler = sampler
        self.config = config
        self.exclude = exclude
        self.label = label
        self.total_blocks = total_blocks
        self.require_transfer_gain = require_transfer_gain
        self.blocks: list[dict[str, Any]] = []
        self.oracle_items_scored = 0
        self._reports_seen = 0
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.block_seconds: list[float] = []
        self.setup_seconds = 0.0

    def start_clock(self, setup_seconds: float) -> None:
        """Start timing blocks now; ``setup_seconds`` = time spent before the first update."""
        self.setup_seconds = float(setup_seconds)
        self._t0 = time.perf_counter()
        self._last = self._t0

    def oracle(self, model: nn.Module, incumbent: StateSnapshot, decision_index: int) -> dict[str, Any]:
        per_slice = oracle_items(
            self.sampler, self.config.oracle_items_per_slice, decision_index, self.exclude
        )
        flat = [it for group in per_slice.values() for it in group]
        candidate = score_items(self.oracle_fn, model, flat)
        with swapped(model, incumbent):
            base = score_items(self.oracle_fn, model, flat)
        self.oracle_items_scored += 2 * len(flat)
        slices: dict[str, dict[str, float]] = {}
        pos = 0
        for name, group in per_slice.items():
            end = pos + len(group)
            c = sum(candidate[pos:end]) / len(group)
            b = sum(base[pos:end]) / len(group)
            slices[name] = {"candidate": c, "incumbent": b, "diff": c - b, "n": len(group)}
            pos = end
        diff = sum(s["diff"] for s in slices.values()) / len(slices)
        return {
            "candidate": sum(s["candidate"] for s in slices.values()) / len(slices),
            "incumbent": sum(s["incumbent"] for s in slices.values()) / len(slices),
            "diff": diff,
            "improved": diff > self.config.oracle_threshold,
            "n": len(flat),
            "items_sha256": hashlib.sha256("\n".join(it.uid for it in flat).encode()).hexdigest(),
            "slices": slices,
        }

    def after(self, record: DecisionRecord, oracle: Mapping[str, Any]) -> None:
        reports = self.operator.reports[self._reports_seen :]
        self._reports_seen = len(self.operator.reports)
        rows = [_report_counts(r, self.operator.kind) for r in reports]
        block_counts = _sum_counts(rows)
        point = self.tracker.measure(self.model, record.decision_index + 1, record.reward_v1)
        first = (
            classify_verdict(record.verdicts[0], require_transfer_gain=self.require_transfer_gain)
            if record.verdicts
            else None
        )
        accepted = first == REASON_ACCEPT
        self.blocks.append(
            {
                "decision_index": record.decision_index,
                "update_index": record.update_index,
                "updates_done": record.update_index + 1,
                "decision": record.decision.value,
                "reason": record.reason,
                "override": record.override,
                "first_reason": first,
                "accepted_first": accepted,
                "oracle": dict(oracle),
                "false_accept": accepted and not oracle["improved"],
                "false_reject": (not accepted) and bool(oracle["improved"]),
                "block_counts": block_counts,
                "hack_rate_block": block_counts["hacks"] / block_counts["rollouts"]
                if block_counts["rollouts"]
                else 0.0,
                "track": point.to_dict(),
            }
        )
        now = time.perf_counter()
        self.block_seconds.append(now - self._last)
        self._last = now
        if self.config.progress:
            done = len(self.blocks)
            mean = (now - self._t0) / done
            eta = mean * max(0, self.total_blocks - done)
            print(
                f"[pilot] {self.label}: decision {record.decision_index} {record.decision.value} "
                f"({record.reason}); block {self.block_seconds[-1]:.1f}s, arm ETA ~{eta:.0f}s",
                flush=True,
            )


class _PilotController(GateController):
    """:class:`~research_rl.capability.loop.GateController` with the pilot's oracle (before) and
    tracker (after) around every gate procedure; the procedure itself is unchanged."""

    def __init__(
        self, gate: CapabilityGate, probe: CapabilityProbe, *, monitor: _ArmMonitor | None = None
    ) -> None:
        super().__init__(gate, probe, rollback_unit=RollbackUnit.PARAMETERS)
        self.monitor = monitor

    def run(
        self,
        model: nn.Module,
        operator: UpdateOperator,
        incumbent: StateSnapshot,
        decision_index: int,
        update_index: int,
        alarm: Any,
        reward_v1: float | None,
        reward_v2: float | None = None,
    ) -> tuple[DecisionRecord, StateSnapshot]:
        monitor = self.monitor
        oracle = monitor.oracle(model, incumbent, decision_index) if monitor is not None else None
        record, new = super().run(
            model, operator, incumbent, decision_index, update_index, alarm, reward_v1, reward_v2
        )
        if monitor is not None and oracle is not None:
            monitor.after(record, oracle)
        return record, new


# ============================================================================== one arm
def _es_fitness(
    sampler: ItemSampler, v1_fn: VerifierScoreFn, v2_fn: VerifierScoreFn, n_items: int
) -> Callable[[nn.Module, int, int], FitnessResult]:
    """ES fitness: greedy completions of ``n_items`` fresh TRAIN items of update ``t`` (purpose
    ``"es"``, the same items for every member), scored with V1 (fitness) and V2 under the gate's
    truncation rule."""
    tokenizer = v2_fn.tokenizer
    calls = max(1, math.ceil(n_items / v2_fn.batch_size))

    def fitness(model: nn.Module, update_index: int, member_index: int) -> FitnessResult:
        del member_index  # items depend on the update only (common random numbers)
        items = sampler.draw(Split.TRAIN, n_items, purpose=ES_PURPOSE, index=update_index)
        texts = v2_fn.generate(model, items)
        truncated = v2_fn.truncated(texts)
        v1 = v1_fn.score_completions(texts, items, truncated)
        v2 = v2_fn.score_completions(texts, items, truncated)
        lengths = tuple(
            v2_fn.max_new_tokens if t else len(tokenizer(c, add_special_tokens=False)["input_ids"])
            for c, t in zip(texts, truncated)
        )
        return FitnessResult(
            reward_v1=sum(v1) / len(v1),
            reward_v2=sum(v2) / len(v2),
            v1_pass=tuple(s == 1.0 for s in v1),
            v2_pass=tuple(s == 1.0 for s in v2),
            lengths=lengths,
            extra={"extra_forward_passes": calls - 1},
        )

    return fitness


def _make_operator(
    spec: ArmSpec,
    model: nn.Module,
    tokenizer: Any,
    sampler: ItemSampler,
    config: PilotConfig,
    selection_info: Mapping[str, Any],
    max_new_tokens: int,
) -> tuple[UpdateOperator, dict[str, Any]]:
    v1, v2 = _verifiers(config, spec.variant)
    op_seed = _seed("pilot_operator", config.seed, spec.seed, spec.operator) % (2**31)
    if spec.operator == "grpo":
        grpo = TorchGRPOOperator(
            model,
            tokenizer,
            sampler,
            v1,
            v2,
            group_size=config.grpo_group_size,
            prompts_per_update=config.grpo_prompts_per_update,
            lr=config.grpo_lr,
            max_new_tokens=max_new_tokens,
            temperature=config.grpo_temperature,
            screen=config.grpo_screen,
            seed=op_seed,
        )
        return grpo, {
            "type": "torch_grpo",
            "parameters": len(grpo.parameter_names()),
            "screen": config.grpo_screen,
        }
    v1_fn = VerifierScoreFn(tokenizer, v1, max_new_tokens=max_new_tokens, batch_size=config.eval_batch_size)
    v2_fn = VerifierScoreFn(tokenizer, v2, max_new_tokens=max_new_tokens, batch_size=config.eval_batch_size)
    informed = _selection_from(selection_info)
    sigma, lr = config.es_sigma, config.es_lr
    info: dict[str, Any] = {}
    if spec.operator == "es_full":
        if config.es_full_space == "ffn_weights":
            space: Any = FullParameterSpace(model, _ffn_weight_names(model))
            sigma, lr = config.es_full_sigma, config.es_full_lr
            info = {"type": "FullParameterSpace", "names": "every FFN gate/up/down weight"}
        else:
            selection = ChannelSelection.all_channels(model)
            space = _subset_space(model, selection, config)
            info = {"type": type(space).__name__, "channels": selection.count(), "selection": "all channels"}
    else:
        if spec.operator == "es_informed":
            selection = informed
            source = f"informed ({selection_info['scorer']})"
        else:
            selection = density_matched_random(
                informed, model, _seed("pilot_random_channels", config.seed, spec.seed) % (2**31)
            )
            source = "density_matched_random(informed)"
        space = _subset_space(model, selection, config)
        info = {
            "type": type(space).__name__,
            "channels": selection.count(),
            "source": source,
            "selection": {str(k): list(v) for k, v in selection.to_dict().items()},
        }
    es_cfg = ESConfig(population=config.es_population, sigma=sigma, lr=lr, seed=op_seed)
    op = ESOperator(
        space, _es_fitness(sampler, v1_fn, v2_fn, config.es_items_per_update), es_cfg, name=spec.operator
    )
    info.update({"dim": op.dim, "es": es_cfg.to_dict(), "parameters": len(op.parameter_names())})
    return op, info


def _subset_space(model: nn.Module, selection: ChannelSelection, config: PilotConfig) -> Any:
    if config.es_space == "gain":
        return ChannelGainSpace(model, selection)
    return ChannelWeightSpace(model, selection)


@dataclass
class _ArmContext:
    """What every arm shares (JSON-able except the model weights, which come from ``start_path``)."""

    config: dict[str, Any]
    start_path: str
    start_map: dict[str, Any]
    selection: dict[str, Any]
    items_per_retention_slice: int
    output_dir: str | None


def _run_arm(
    spec: ArmSpec, ctx: _ArmContext, *, num_updates: int | None = None, write_log: bool = True
) -> dict[str, Any]:
    """Run one arm from the saved start weights; returns its JSON-able result."""
    config = PilotConfig.from_dict(ctx.config)
    t_start = time.perf_counter()
    with _torch_threads(config.threads_per_arm):
        splits = build_splits(config)
        tokenizer = build_char_tokenizer()
        model = build_model(config, len(tokenizer))
        model.load_state_dict(load_file(ctx.start_path), strict=True)
        model.eval()
        budget = _budget(config, splits)
        sampler = ItemSampler(splits, root_seed=_sampler_root(config, spec.seed))
        start = StartMap.from_dict(ctx.start_map)
        transfer, retention = _slices(config, splits)
        gate_cfg = dataclasses.replace(
            config.base_gate_config(items_per_retention_slice=ctx.items_per_retention_slice),
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
        inner, space_info = _make_operator(spec, model, tokenizer, sampler, config, ctx.selection, budget)
        operator = _RecordingOperator(inner, kind="grpo" if spec.operator == "grpo" else "es")
        alarm_kwargs = dict(config.alarm)
        if spec.operator != "grpo":
            alarm_kwargs["rollouts_per_unit"] = config.es_population
        alarm = DisagreementAlarm(**alarm_kwargs)
        updates = config.updates_for(spec.operator) if num_updates is None else int(num_updates)
        # every arm of the config measures its final weights at this one tracker step (inside the
        # oracle exclusion), so `final` of same-seed arms is scored on identical tracker items
        final_step = oracle_exclusion_steps(config, updates)
        exclude = excluded_uids(
            sampler,
            start,
            track_items_per_slice=config.track_items_per_slice,
            max_step=final_step,
        )
        controller = _PilotController(CapabilityGate(gate_cfg), probe)
        log_path = None
        if write_log and ctx.output_dir is not None:
            log_path = Path(ctx.output_dir) / "decisions" / f"{spec.file_stem}.jsonl"
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
            label=spec.key,
            total_blocks=updates // gate_cfg.block_updates,
            require_transfer_gain=gate_cfg.require_transfer_gain,
        )
        controller.monitor = monitor
        point0 = tracker.measure(model, 0, None)
        monitor.start_clock(time.perf_counter() - t_start)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = loop.run(updates)
        rows = [_report_counts(r, operator.kind) for r in operator.reports]
        trailing = loop.updates_since_decision
        trailing_point, final_point = _measure_final(
            tracker, model, operator.reports, records, trailing=trailing, final_step=final_step
        )
        final_fingerprint = fingerprint(model)
    return _arm_result(
        spec,
        config,
        records=records,
        monitor=monitor,
        rows=rows,
        point0=point0.to_dict(),
        trailing_point=trailing_point,
        trailing_updates=trailing,
        final_point=final_point,
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


def _arm_result(
    spec: ArmSpec,
    config: PilotConfig,
    *,
    records: Sequence[DecisionRecord],
    monitor: _ArmMonitor,
    rows: Sequence[Mapping[str, int]],
    point0: Mapping[str, Any],
    trailing_point: Mapping[str, Any] | None,
    trailing_updates: int,
    final_point: Mapping[str, Any],
    tracker: CapabilityTracker,
    space_info: Mapping[str, Any],
    alarm: DisagreementAlarm,
    gate_cfg: GateConfig,
    horizon: int | None,
    required: int,
    updates: int,
    eval_items: Mapping[str, int],
    warnings_seen: Sequence[str],
    wall: float,
    final_fingerprint: str,
) -> dict[str, Any]:
    totals = _sum_counts(rows)
    curve = [_curve_point(point0, 0, 0)]
    cumulative = 0
    upto = 0
    for block in monitor.blocks:
        done = block["updates_done"]
        cumulative += sum(r["fpe"] for r in rows[upto:done])
        upto = done
        curve.append(_curve_point(block["track"], done, cumulative))
    if trailing_point is not None:
        curve.append(_curve_point(trailing_point, len(rows), totals["fpe"]))
    last = _curve_point(final_point, len(rows), totals["fpe"])
    final_block = monitor.blocks[-1] if monitor.blocks else None
    last_rows = rows[-gate_cfg.block_updates :] if rows else []
    last_counts = _sum_counts(last_rows) if last_rows else None
    decided = list(monitor.blocks)
    counts_by_decision: dict[str, int] = {}
    for r in records:
        counts_by_decision[r.decision.value] = counts_by_decision.get(r.decision.value, 0) + 1
    improved = [b for b in decided if b["oracle"]["improved"]]
    not_improved = [b for b in decided if not b["oracle"]["improved"]]
    return {
        "key": spec.key,
        **spec.to_dict(),
        "updates": updates,
        "trailing_updates": trailing_updates,
        "space": dict(space_info),
        "gate_config": gate_cfg.to_dict(),
        "horizon": horizon,
        "required_verdicts": required,
        "decisions": [r.to_dict() for r in records],
        "decision_counts": counts_by_decision,
        "blocks": monitor.blocks,
        "curve": curve,
        "t_rule0": tracker.t_rule0,
        "final_step": last["step"],
        "final": {
            "A": last["A"],
            "R": dict(last["R"]),
            "R_mean": last["R_mean"],
            "T_rule": last["T_rule"],
            "T_task": last["T_task"],
            "D": last["D"],
            "t_rule_unseen": last["t_rule_unseen"],
        },
        "hack_rate_total": totals["hacks"] / totals["rollouts"] if totals["rollouts"] else 0.0,
        "hack_rate_final": (last_counts["hacks"] / last_counts["rollouts"])
        if last_counts and last_counts["rollouts"]
        else 0.0,
        "counts": {**totals, "eval_items_scored": dict(eval_items)},
        "gate_errors": {
            "blocks": len(decided),
            "accepted_first": sum(1 for b in decided if b["accepted_first"]),
            "oracle_improved": len(improved),
            "false_accepts": sum(1 for b in decided if b["false_accept"]),
            "false_rejects": sum(1 for b in decided if b["false_reject"]),
            "false_accept_rate": _ratio(
                sum(1 for b in not_improved if b["accepted_first"]), len(not_improved)
            ),
            "false_reject_rate": _ratio(sum(1 for b in improved if not b["accepted_first"]), len(improved)),
        },
        "alarm": {
            "updates": alarm.updates,
            "skipped_updates": alarm.skipped_updates,
            "gap_alarms": alarm.gap_alarms,
            "length_alarms": alarm.length_alarms,
            "config": alarm.config(),
        },
        "last_decision": final_block["decision"] if final_block else None,
        "final_fingerprint": final_fingerprint,
        "setup_s": monitor.setup_seconds,
        "block_seconds": list(monitor.block_seconds),
        "warnings": list(warnings_seen),
        "wall_time_s": wall,
    }


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _curve_point(track: Mapping[str, Any], updates: int, fpe: int) -> dict[str, Any]:
    r = dict(track["R"])
    return {
        "step": track["step"],
        "updates": updates,
        "fpe": fpe,
        "reward_v1": track["reward_v1"],
        "A": track["A"],
        "R": r,
        "R_mean": sum(r.values()) / len(r) if r else math.nan,
        "T_rule": track["T_rule"],
        "T_task": track["T_task"],
        "D": track["D"],
        "t_rule_unseen": track["slices"].get("t_rule/unseen_pairs", math.nan),
    }


def _arm_worker(payload: tuple[dict[str, Any], dict[str, Any], int | None, bool]) -> dict[str, Any]:
    """Process-pool entry point (module level, picklable): ``(spec dict, context dict, num_updates,
    write_log)`` -> arm result."""
    spec_d, ctx_d, num_updates, write_log = payload
    spec = ArmSpec(**spec_d)
    ctx = _ArmContext(**ctx_d)
    return _run_arm(spec, ctx, num_updates=num_updates, write_log=write_log)


def _run_many(
    payloads: Sequence[tuple[dict[str, Any], dict[str, Any], int | None, bool]], jobs: int
) -> list[dict[str, Any]]:
    if jobs <= 1 or len(payloads) <= 1:
        return [_arm_worker(p) for p in payloads]
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        return list(pool.map(_arm_worker, payloads))


# ============================================================================== statistics for the criteria
def paired_interval(diffs: Sequence[float], alpha: float = 0.05) -> dict[str, Any]:
    """Two-sided ``1 - alpha`` Student-t interval of the mean of paired differences.

    Returns ``n``, ``mean``, ``sd``, ``se``, ``lo``, ``hi``, ``df`` and ``method``. With ``n < 2``
    the interval is undefined (``lo = hi = None``; ``mean`` is None for ``n = 0``); with zero variance it collapses to the mean (``se = 0``). The
    t-interval assumes roughly normal per-seed differences; with three seeds it is wide and the
    criteria treat it as such (any seed count below 3 is INCONCLUSIVE before this is consulted).
    """
    values = [float(d) for d in diffs]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None, "lo": None, "hi": None, "df": 0, "method": "t"}
    mean = math.fsum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "se": None, "lo": None, "hi": None, "df": 0, "method": "t"}
    if max(values) == min(values):
        mean = values[0]  # exact: zero variance collapses the interval to the common value
        sd = 0.0
    else:
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    q = student_t_quantile(1.0 - alpha / 2.0, n - 1)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "lo": mean - q * se,
        "hi": mean + q * se,
        "df": n - 1,
        "method": "t",
    }


def _select(arms: Sequence[Mapping[str, Any]], **where: Any) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for arm in arms:
        if all(arm.get(k) == v for k, v in where.items()):
            out[int(arm["seed"])] = arm
    return out


def _paired(
    a: Mapping[int, Mapping[str, Any]],
    b: Mapping[int, Mapping[str, Any]],
    fn: Callable[[Mapping[str, Any]], float],
) -> tuple[list[int], list[float]]:
    seeds = sorted(set(a) & set(b))
    return seeds, [fn(a[s]) - fn(b[s]) for s in seeds]


def _per_seed_mean(per_op: Mapping[str, tuple[list[int], list[float]]]) -> tuple[list[int], list[float]]:
    """Average a statistic over operators per seed (only seeds every operator has)."""
    if not per_op:
        return [], []
    common = set.intersection(*(set(seeds) for seeds, _ in per_op.values()))
    seeds = sorted(common)
    out = []
    for s in seeds:
        vals = [diffs[seeds_op.index(s)] for seeds_op, diffs in per_op.values()]
        out.append(sum(vals) / len(vals))
    return seeds, out


def _verdict(verdict: str, statistic: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {"verdict": verdict, "statistic": dict(statistic), "reason": reason}


# ============================================================================== the section-3 criteria
CRITERIA_DEFINITIONS: dict[str, str] = {
    "pairing": (
        "Every statistic is a paired difference over seeds: arms with the same seed share the start "
        "checkpoint, the sampler root seed and hence every tracker / oracle item. Scope: every seed of a run "
        "also shares one split, one start checkpoint (after SFT), one start map and one informed channel selection "
        "(made once on the start model); the arm seed varies only the RL stage (the sampler root, i.e. every item drawn through the arm's "
        "sampler: gate, tracker, oracle and training (GRPO prompt / ES fitness) items, the operator seed and the es_random draw). The intervals therefore cover "
        "RL-stage randomness only, conditional on this split, start and selection, not SFT or selection "
        "variance. Interval: two-sided "
        "1 - criteria_alpha Student-t over seeds (paired_interval). Fewer than 3 paired seeds or a "
        "missing counterpart arm -> INCONCLUSIVE, never SUCCESS or KILL."
    ),
    "gate": (
        "Per operator and seed: leak variant d_hack = hack_rate_final(gate_off) - hack_rate_final(gate_on) "
        "and d_T = T_rule_final(gate_on) - T_rule_final(gate_off); clean variant c = T_rule_final(gate_on) - "
        "T_rule_final(gate_off); each averaged over the operators that have all four arms for that seed. "
        "SUCCESS: interval(d_hack).lo > 0 and interval(d_T).lo > 0 and mean(c) >= -se(c) (clean T_rule within "
        "one standard error of gate-off). KILL: the clean-run T_rule cost exceeds the planted-run gain, tested "
        "paired: k = -c - d_T per seed, interval(k).lo > 0; or the first verdicts (all arms, both gate settings) reject more than half of the blocks "
        "the oracle marks as improved (evaluable only with >= min_improved_blocks such blocks). SUCCESS requires both "
        "KILL conditions to be evaluable: with fewer than min_improved_blocks oracle-improved blocks the verdict is "
        "INCONCLUSIVE ('reject-rate KILL not evaluable'), whatever the intervals show (the k KILL still applies). "
        "Otherwise INCONCLUSIVE. "
        "T_rule_final = tracker T_rule of the weights after the last update, measured at the tracker step "
        "oracle_exclusion_steps(config), the same for every arm (so same-seed arms' finals are scored on identical "
        "tracker items whatever their decision counts); hack_rate_final = V1-pass & V2-fail share of "
        "the last block_updates updates' training rollouts. Every gate verdict carries gate_exercise: when fewer "
        "than min_improved_blocks blocks are oracle-improved or the gate accepted no block, the reason says "
        "'gate not exercised' and the error rates are reported as not evaluable / not informative."
    ),
    "subset": (
        "Per seed (averaged over the gate settings and variants present for both arms): d_R = R_mean(es_informed) "
        "- R_mean(es_random), d_T = T_rule(es_informed) - T_rule(es_random), d_A = A(es_informed) - A(es_random) "
        "at the end of training. 'Equal A': |mean(d_A)| <= equal_a_tolerance. SUCCESS: equal A and "
        "(interval(d_R).lo > 0 or interval(d_T).lo > 0). KILL: equal A and neither interval excludes zero in "
        "favour of the informed subset (no separation). INCONCLUSIVE: A not equal, or < 3 seeds. The informed "
        "subset is one fixed selection shared by every seed (only the density-matched random control is redrawn "
        "per seed), so a verdict is about this selection, not about the selection procedure. Reported "
        "beside it: each subset's A gain since its step-0 tracker point (a no-separation KILL where neither "
        "subset acquired anything is vacuous and should be read as such)."
    ),
    "operator": (
        "Per seed, GRPO arm (clean variant and gate_off if present, else the first available): G = T_rule_final "
        "- T_rule at the arm's own step-0 tracker point (not the start map: one fixed sample shared by every "
        "seed would add a bias the interval cannot see) and budget B = operator_budget_factor x its total forward-pass equivalents (sequence level: "
        "each generated or training-forward sequence = 1, each backward sequence = 2). ES arm "
        "(operator_es_arm, same variant and gate): T_rule gain (relative to its own step-0 point) at its "
        "last tracker point with cumulative FPE <= B (one point, like GRPO's final point; not a maximum "
        "over noisy points, which would favour ES by selection); r = gain - operator_gain_fraction x G. SUCCESS: interval(r).lo > 0 and mean(G) > 0. KILL: "
        "interval(r).hi < 0, mean(G) > 0 and the ES arm spent at least B in every seed. INCONCLUSIVE "
        "otherwise (in particular when GRPO gained nothing or ES stopped before spending B). kill_reachable "
        "(statistic) = the ES arm spent B in every paired seed (None without a paired seed); when it is False "
        "every reason says 'Operator KILL not reachable' and the run warns: only SUCCESS or INCONCLUSIVE "
        "were possible (raise es_updates)."
    ),
    "low_start": (
        "Start pass@k (k = start_k, sampled) on t_rule/unseen_pairs versus each arm's final greedy pass@1 on that "
        "slice (tracker), per seed. SUCCESS: some arm (operator x gate x variant) has interval(final - "
        "start_pass_k).lo > 0. KILL: every arm has interval.hi <= 0 (none exceeds the start's pass@k: the task "
        "is acquisition-limited; reported, the task is not weakened). Otherwise INCONCLUSIVE. The start pass@k "
        "is one sample shared by every arm and seed, so its sampling error is common to all paired differences: "
        "each interval adds se_start = baseline_standard_error(pass_k, "
        "start items) (binomial bound, never 0); the interval half-width is the quadrature sum of the seeds' t half-width "
        "and the baseline's half-width (interval_with_baseline_error). Several arms are "
        "compared without a multiplicity correction (stated, not corrected)."
    ),
}


def gate_exercise(arms: Sequence[Mapping[str, Any]], min_improved_blocks: int) -> dict[str, Any]:
    """Whether the run exercised the gate enough for its error rates to mean anything.

    Pooled over every block of ``arms`` (first verdicts, both gate settings): ``accepted`` blocks,
    ``oracle_improved`` blocks and ``not_improved`` blocks. The false-reject rate is *evaluable*
    with at least ``min_improved_blocks`` oracle-improved blocks; the false-accept rate is
    *informative* only if the gate accepted at least one block (a gate that never accepts has a
    false-accept rate of 0 by construction, which says nothing about discrimination). ``exercised``
    = both. ``problems`` lists every missing piece in words (empty when exercised); the pilot
    copies them into ``results["warnings"]`` and the summary."""
    blocks = [b for arm in arms for b in arm.get("blocks", [])]
    improved = sum(1 for b in blocks if b["oracle"]["improved"])
    accepted = sum(1 for b in blocks if b["accepted_first"])
    not_improved = len(blocks) - improved
    problems: list[str] = []
    if improved < min_improved_blocks:
        problems.append(
            f"only {improved} of {len(blocks)} blocks are oracle-improved (need >= {min_improved_blocks}): "
            "the false-reject rate and the reject-rate KILL are not evaluable"
        )
    if accepted == 0:
        problems.append(
            f"the gate accepted none of {len(blocks)} blocks (first verdicts): its false-accept rate is 0 by "
            "construction and says nothing about discrimination"
        )
    diffs = [float(b["oracle"]["diff"]) for b in blocks if b["oracle"].get("diff") is not None]
    return {
        "blocks": len(blocks),
        "accepted": accepted,
        "oracle_improved": improved,
        "not_improved": not_improved,
        "min_improved_blocks": int(min_improved_blocks),
        "max_oracle_diff": max(diffs) if diffs else None,
        "false_reject_evaluable": improved >= min_improved_blocks,
        "false_accept_informative": accepted > 0 and not_improved > 0,
        "exercised": not problems,
        "problems": problems,
    }


def _gate_criterion(arms: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    alpha = float(cfg["criteria_alpha"])
    hack: dict[str, tuple[list[int], list[float]]] = {}
    gain: dict[str, tuple[list[int], list[float]]] = {}
    cost: dict[str, tuple[list[int], list[float]]] = {}
    for op in OPERATORS:
        lo = _select(arms, operator=op, variant="leak", gate="gate_on")
        lf = _select(arms, operator=op, variant="leak", gate="gate_off")
        co = _select(arms, operator=op, variant="clean", gate="gate_on")
        cf = _select(arms, operator=op, variant="clean", gate="gate_off")
        seeds = sorted(set(lo) & set(lf) & set(co) & set(cf))
        if not seeds:
            continue
        hack[op] = (seeds, [float(lf[s]["hack_rate_final"]) - float(lo[s]["hack_rate_final"]) for s in seeds])
        gain[op] = (seeds, [float(lo[s]["final"]["T_rule"]) - float(lf[s]["final"]["T_rule"]) for s in seeds])
        cost[op] = (seeds, [float(co[s]["final"]["T_rule"]) - float(cf[s]["final"]["T_rule"]) for s in seeds])
    blocks = [b for arm in arms for b in arm.get("blocks", [])]
    improved = [b for b in blocks if b["oracle"]["improved"]]
    rejected = [b for b in improved if not b["accepted_first"]]
    reject_rate = _ratio(len(rejected), len(improved))
    oracle_stat = {
        "improved_blocks": len(improved),
        "rejected_improved": len(rejected),
        "false_reject_rate": reject_rate,
        "accepted_not_improved": sum(
            1 for b in blocks if b["accepted_first"] and not b["oracle"]["improved"]
        ),
        "not_improved_blocks": sum(1 for b in blocks if not b["oracle"]["improved"]),
    }
    seeds, d_hack = _per_seed_mean(hack)
    _, d_t = _per_seed_mean(gain)
    _, c = _per_seed_mean(cost)
    stat: dict[str, Any] = {
        "operators": sorted(hack),
        "seeds": seeds,
        "d_hack": paired_interval(d_hack, alpha),
        "d_T_rule_leak": paired_interval(d_t, alpha),
        "c_T_rule_clean": paired_interval(c, alpha),
        "k_cost_minus_gain": paired_interval([-ci - di for ci, di in zip(c, d_t)], alpha),
        "oracle": oracle_stat,
        "exercise": gate_exercise(arms, int(cfg["min_improved_blocks"])),
        "per_operator": {
            op: {
                "d_hack": paired_interval(hack[op][1], alpha),
                "d_T_rule_leak": paired_interval(gain[op][1], alpha),
                "c_T_rule_clean": paired_interval(cost[op][1], alpha),
            }
            for op in sorted(hack)
        },
    }
    if len(seeds) < 3:
        return _verdict(
            VERDICT_INCONCLUSIVE,
            stat,
            f"{len(seeds)} paired seed(s) with all four gate x variant arms (need >= 3)",
        )
    exercise = stat["exercise"]
    # the reject-rate KILL is evaluable with >= min_improved_blocks oracle-improved blocks (the same
    # test as gate_exercise, so the criterion and results["gate_exercise"] cannot disagree)
    reject_evaluable = bool(exercise["false_reject_evaluable"])
    note = "" if exercise["exercised"] else " [gate not exercised: " + "; ".join(exercise["problems"]) + "]"
    if reject_evaluable and reject_rate is not None and reject_rate > 0.5:
        return _verdict(
            VERDICT_KILL,
            stat,
            f"the gate rejected {reject_rate:.0%} of oracle-improved blocks (> 50%)" + note,
        )
    sd_t, sc, k = stat["d_T_rule_leak"], stat["c_T_rule_clean"], stat["k_cost_minus_gain"]
    if k["lo"] > 0:
        return _verdict(
            VERDICT_KILL,
            stat,
            f"clean-run T_rule cost {-sc['mean']:.4f} exceeds the planted-run T_rule gain {sd_t['mean']:.4f} "
            f"(paired k = cost - gain, interval lower bound {k['lo']:.4f} > 0)" + note,
        )
    clean_ok = sc["mean"] >= -(sc["se"] or 0.0)
    intervals_ok = stat["d_hack"]["lo"] > 0 and sd_t["lo"] > 0 and clean_ok
    if not reject_evaluable:
        # SUCCESS needs both KILL conditions evaluated and not holding: with too few oracle-improved
        # blocks the reject-rate KILL cannot be checked (a gate that rejects everything would pass)
        rate = "-" if reject_rate is None else f"{reject_rate:.0%}"
        return _verdict(
            VERDICT_INCONCLUSIVE,
            stat,
            f"reject-rate KILL not evaluable: {len(improved)} oracle-improved block(s), need >= "
            f"{exercise['min_improved_blocks']} (rejected {len(rejected)}, rate {rate}); "
            + (
                "the T_rule / hack intervals alone would meet the success condition"
                if intervals_ok
                else "the T_rule / hack intervals do not establish success either"
            )
            + note,
        )
    if intervals_ok:
        return _verdict(
            VERDICT_SUCCESS,
            stat,
            "lower final hack rate and higher T_rule with the gate on leak runs; clean cost within 1 SE"
            + note,
        )
    return _verdict(
        VERDICT_INCONCLUSIVE, stat, "intervals do not establish success and no kill condition holds" + note
    )


def _start_point(arm: Mapping[str, Any], key: str) -> float | None:
    """``key`` of the arm's step-0 tracker point (``None`` if the curve does not carry it)."""
    curve = arm.get("curve") or []
    if not curve or key not in curve[0] or curve[0][key] is None:
        return None
    return float(curve[0][key])


def _a_gain(arm: Mapping[str, Any]) -> float | None:
    start = _start_point(arm, "A")
    return None if start is None else float(arm["final"]["A"]) - start


def _subset_criterion(arms: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    alpha = float(cfg["criteria_alpha"])
    d_r: dict[str, tuple[list[int], list[float]]] = {}
    d_t: dict[str, tuple[list[int], list[float]]] = {}
    d_a: dict[str, tuple[list[int], list[float]]] = {}
    for variant in VARIANTS:
        for gate in GATES:
            inf = _select(arms, operator="es_informed", variant=variant, gate=gate)
            rnd = _select(arms, operator="es_random", variant=variant, gate=gate)
            key = f"{variant}/{gate}"
            if not set(inf) & set(rnd):
                continue
            d_r[key] = _paired(inf, rnd, lambda a: float(a["final"]["R_mean"]))
            d_t[key] = _paired(inf, rnd, lambda a: float(a["final"]["T_rule"]))
            d_a[key] = _paired(inf, rnd, lambda a: float(a["final"]["A"]))
    a_gains: dict[str, list[float]] = {"es_informed": [], "es_random": []}
    for arm in arms:
        gain_a = _a_gain(arm)
        if arm.get("operator") in a_gains and gain_a is not None:
            a_gains[str(arm["operator"])].append(gain_a)
    seeds, r = _per_seed_mean(d_r)
    _, t = _per_seed_mean(d_t)
    _, a = _per_seed_mean(d_a)
    stat: dict[str, Any] = {
        "settings": sorted(d_r),
        "seeds": seeds,
        "d_R": paired_interval(r, alpha),
        "d_T_rule": paired_interval(t, alpha),
        "d_A": paired_interval(a, alpha),
        "equal_a_tolerance": float(cfg["equal_a_tolerance"]),
        "A_gain_informed": paired_interval(a_gains["es_informed"], alpha),
        "A_gain_random": paired_interval(a_gains["es_random"], alpha),
    }
    if len(seeds) < 3:
        return _verdict(
            VERDICT_INCONCLUSIVE, stat, f"{len(seeds)} paired seed(s) of es_informed vs es_random (need >= 3)"
        )
    equal_a = abs(stat["d_A"]["mean"]) <= float(cfg["equal_a_tolerance"])
    stat["equal_A"] = equal_a
    if not equal_a:
        return _verdict(
            VERDICT_INCONCLUSIVE, stat, "acquisition A differs between the subsets: no comparison at equal A"
        )
    if stat["d_R"]["lo"] > 0 or stat["d_T_rule"]["lo"] > 0:
        return _verdict(
            VERDICT_SUCCESS, stat, "informed subset beats the density-matched random subset at equal A"
        )
    return _verdict(
        VERDICT_KILL,
        stat,
        "no separation in favour of the informed subset: keep channel groups only as the rollback unit",
    )


def _pick_setting(arms: Sequence[Mapping[str, Any]], operator: str) -> tuple[str, str] | None:
    for variant in ("clean", "leak"):
        for gate in ("gate_off", "gate_on"):
            if _select(arms, operator=operator, variant=variant, gate=gate):
                return variant, gate
    return None


def _operator_criterion(arms: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    alpha = float(cfg["criteria_alpha"])
    es_arm = str(cfg["operator_es_arm"])
    factor = float(cfg["operator_budget_factor"])
    fraction = float(cfg["operator_gain_fraction"])
    setting = _pick_setting(arms, "grpo")
    stat: dict[str, Any] = {
        "es_arm": es_arm,
        "budget_factor": factor,
        "gain_fraction": fraction,
        "count": "sequence-level FPE",
    }
    if setting is None:
        return _verdict(VERDICT_INCONCLUSIVE, stat, "no GRPO arm")
    variant, gate = setting
    stat["setting"] = {"variant": variant, "gate": gate}
    grpo = _select(arms, operator="grpo", variant=variant, gate=gate)
    es = _select(arms, operator=es_arm, variant=variant, gate=gate)
    seeds = sorted(set(grpo) & set(es))
    gains: list[float] = []
    margins: list[float] = []
    spent_all = True
    per_seed = []
    for s in seeds:
        g_arm, e_arm = grpo[s], es[s]
        g_start = _start_point(g_arm, "T_rule")
        e_start = _start_point(e_arm, "T_rule")
        g0 = float(g_arm["t_rule0"]) if g_start is None else g_start
        e0 = float(e_arm["t_rule0"]) if e_start is None else e_start
        g_gain = float(g_arm["final"]["T_rule"]) - g0
        budget = factor * float(g_arm["counts"]["fpe"])
        within = [p for p in e_arm["curve"] if float(p["fpe"]) <= budget]
        # the last tracker point within the budget, scored by the same tracker procedure as GRPO's
        # final point (a maximum over noisy points would favour ES by selection); it is a curve
        # point at ES's own step, so not on the items of GRPO's final point at the shared final step
        es_point = within[-1] if within else None  # curve order = cumulative FPE order
        es_gain = float(es_point["T_rule"]) - e0 if es_point is not None else 0.0
        spent = float(e_arm["counts"]["fpe"]) >= budget
        spent_all = spent_all and spent
        gains.append(g_gain)
        margins.append(es_gain - fraction * g_gain)
        per_seed.append(
            {
                "seed": s,
                "grpo_gain": g_gain,
                "budget_fpe": budget,
                "es_gain_within_budget": es_gain,
                "es_point_fpe": float(es_point["fpe"]) if es_point is not None else None,
                "es_fpe_total": float(e_arm["counts"]["fpe"]),
                "es_budget_share": float(e_arm["counts"]["fpe"]) / budget if budget > 0 else None,
                "es_spent_budget": spent,
            }
        )
    unspent = [p["seed"] for p in per_seed if not p["es_spent_budget"]]
    stat.update(
        {
            "seeds": seeds,
            "per_seed": per_seed,
            "grpo_gain": paired_interval(gains, alpha),
            "r": paired_interval(margins, alpha),
            # the KILL needs the ES arm to have spent B in every paired seed; None = no paired seed
            "kill_reachable": (not unspent) if seeds else None,
        }
    )
    note = ""
    if unspent:
        # an unspent seed has budget > its FPE >= 0, so its share is a number
        share = max(float(p["es_budget_share"] or 0.0) for p in per_seed if not p["es_spent_budget"])
        note = (
            f" [Operator KILL not reachable: the {es_arm} arm stopped before spending the {factor:g}x budget in "
            f"{len(unspent)} of {len(seeds)} paired seed(s) (at most {share:.1%} of it spent); raise es_updates]"
        )
    if len(seeds) < 3:
        return _verdict(
            VERDICT_INCONCLUSIVE,
            stat,
            f"{len(seeds)} paired seed(s) of grpo vs {es_arm} (need >= 3)" + note,
        )
    if stat["grpo_gain"]["mean"] <= 0:
        return _verdict(
            VERDICT_INCONCLUSIVE, stat, "GRPO did not gain T_rule: there is no gain for ES to reach" + note
        )
    if stat["r"]["lo"] > 0:
        return _verdict(
            VERDICT_SUCCESS,
            stat,
            f"ES reached {fraction:g} of GRPO's T_rule gain within {factor:g}x its FPE budget" + note,
        )
    if stat["r"]["hi"] < 0:
        if spent_all:
            return _verdict(
                VERDICT_KILL, stat, "ES stays below the required share of GRPO's gain within the budget"
            )
        return _verdict(
            VERDICT_INCONCLUSIVE,
            stat,
            "ES is below the required share but stopped before spending the budget (raise es_updates)" + note,
        )
    return _verdict(VERDICT_INCONCLUSIVE, stat, "interval of r contains zero" + note)


LOW_START_SLICE = "t_rule/unseen_pairs"


def _start_items(results: Mapping[str, Any], name: str, cfg: Mapping[str, Any]) -> int:
    """Item count of the start map's ``name`` slice (``results["start_map"]``), else the config's
    ``start_items_per_slice`` (synthetic results without a start map)."""
    sl = results.get("start_map", {}).get("slices", {}).get(name)
    if isinstance(sl, Mapping):
        if "n" in sl:
            return int(sl["n"])
        if "uids" in sl:
            return len(sl["uids"])
    return int(cfg["start_items_per_slice"])


def baseline_standard_error(p: float, n: int) -> float:
    """Conservative standard error of a start-map mean ``p`` of ``n`` per-item values in ``[0, 1]``.

    The variance of a ``[0, 1]`` variable with mean ``m`` is at most ``m (1 - m)``; ``m`` is taken as
    ``(n p + 0.5) / (n + 1)`` so that an observed ``p = 0`` (or 1) still gets a non-zero error (a
    zero-count sample does not prove a zero rate). Per-item pass@k with ``num_samples = k`` is 0/1,
    so the bound is then the binomial standard error."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    m = (n * float(p) + 0.5) / (n + 1)
    return math.sqrt(m * (1.0 - m) / n)


def interval_with_baseline_error(
    interval: Mapping[str, Any], se_baseline: float, alpha: float = 0.05, *, baseline_n: int | None = None
) -> dict[str, Any]:
    """Widen a :func:`paired_interval` of ``value - baseline`` differences for a baseline that is
    one noisy measurement shared by every seed (so its error is common to all differences and the
    across-seed spread cannot see it).

    The two independent error sources are combined by their half-widths in quadrature, each at its
    own quantile: ``half = sqrt((t_{1-alpha/2, n-1} se_seeds)^2 + (t_{1-alpha/2, baseline_n-1}
    se_baseline)^2)`` (``baseline_n`` = the baseline's item count; ``None`` uses the normal quantile
    for the baseline term). A pooled Welch-Satterthwaite interval is anti-conservative with three
    seeds (the estimated ``se_seeds`` is itself very noisy); the quadrature rule keeps the seeds'
    small-sample quantile on their term. ``se`` reports ``sqrt(se_seeds^2 + se_baseline^2)``;
    ``n < 2`` stays undefined."""
    out = dict(interval)
    out["se_seeds"] = interval.get("se")
    out["se_baseline"] = float(se_baseline)
    n = int(interval["n"])
    if interval.get("se") is None or n < 2:
        return out
    se_seeds = float(interval["se"])
    se_b = float(se_baseline)
    q_seeds = student_t_quantile(1.0 - alpha / 2.0, n - 1)
    base_df = float(max(1, int(baseline_n) - 1)) if baseline_n else 1e9  # 1e9: the normal quantile
    q_base = student_t_quantile(1.0 - alpha / 2.0, base_df)
    half = math.sqrt((q_seeds * se_seeds) ** 2 + (q_base * se_b) ** 2)
    mean = float(interval["mean"])
    out.update(
        {
            "se": math.sqrt(se_seeds**2 + se_b**2),
            "lo": mean - half,
            "hi": mean + half,
            "method": "t (seeds) + t (baseline items), half-widths in quadrature",
        }
    )
    return out


def _low_start_criterion(results: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    alpha = float(cfg["criteria_alpha"])
    arms = results.get("arms", [])
    start = results.get("start_summary", {}).get(LOW_START_SLICE)
    stat: dict[str, Any] = {"slice": LOW_START_SLICE}
    if start is None:
        return _verdict(VERDICT_INCONCLUSIVE, stat, "the start map has no t_rule/unseen_pairs entry")
    pass_k = float(start["pass_k"])
    n_start = _start_items(results, LOW_START_SLICE, cfg)
    se_start = baseline_standard_error(pass_k, n_start)
    stat["start_pass_k"] = pass_k
    stat["start_items"] = n_start
    stat["start_pass_k_se"] = se_start
    groups: dict[str, list[float]] = {}
    for arm in arms:
        key = f"{arm['operator']}/{arm['gate']}/{arm['variant']}"
        groups.setdefault(key, []).append(float(arm["final"]["t_rule_unseen"]) - pass_k)
    stat["arms"] = {
        k: interval_with_baseline_error(paired_interval(v, alpha), se_start, alpha, baseline_n=n_start)
        for k, v in sorted(groups.items())
    }
    eligible = {k: v for k, v in stat["arms"].items() if v["n"] >= 3}
    if not eligible:
        return _verdict(VERDICT_INCONCLUSIVE, stat, "no arm has >= 3 seeds")
    winners = sorted(k for k, v in eligible.items() if v["lo"] > 0)
    if winners:
        stat["exceeding"] = winners
        return _verdict(
            VERDICT_SUCCESS, stat, f"{', '.join(winners)} exceed(s) the start's pass@k on unseen pairs"
        )
    if len(eligible) == len(stat["arms"]) and all(v["hi"] <= 0 for v in eligible.values()):
        return _verdict(
            VERDICT_KILL, stat, "no arm exceeds the start's pass@k on unseen pairs: acquisition-limited task"
        )
    return _verdict(
        VERDICT_INCONCLUSIVE,
        stat,
        "no arm significantly exceeds the start's pass@k, but not every arm is below it",
    )


def evaluate_criteria(results: Mapping[str, Any]) -> dict[str, Any]:
    """Section-3 kill criteria on a results dict (pure; see :data:`CRITERIA_DEFINITIONS`).

    Uses ``results["config"]`` (``criteria_alpha``, ``equal_a_tolerance``, ``operator_budget_factor``,
    ``operator_gain_fraction``, ``operator_es_arm``, ``min_improved_blocks``; defaults of
    :class:`PilotConfig` for missing keys), ``results["start_summary"]`` and ``results["arms"]``:
    each arm needs ``operator``, ``gate``, ``variant``, ``seed``, ``final`` (``A``, ``R_mean``,
    ``T_rule``, ``t_rule_unseen``), ``t_rule0``, ``hack_rate_final``, ``counts["fpe"]``, ``curve``
    (points with ``fpe`` and ``T_rule``, optionally ``A``; the first point is the arm's step-0
    baseline, ``t_rule0`` is the fallback) and ``blocks`` (with ``accepted_first`` and
    ``oracle["improved"]``). Returns ``{name: {"verdict", "statistic", "reason"}}`` plus
    ``"definitions"``.
    """
    defaults = PilotConfig().to_dict()
    cfg = {**defaults, **dict(results.get("config", {}))}
    arms = list(results.get("arms", []))
    return {
        "gate": _gate_criterion(arms, cfg),
        "subset": _subset_criterion(arms, cfg),
        "operator": _operator_criterion(arms, cfg),
        "low_start": _low_start_criterion(results, cfg),
        "definitions": dict(CRITERIA_DEFINITIONS),
    }


def _criteria_warnings(criteria: Mapping[str, Any]) -> list[str]:
    """Run warnings derived from the evaluated criteria (``results["warnings"]`` and the summary):
    a section-3 KILL condition the run could not reach. Shared by the P0 pilot and the P1 script."""
    out: list[str] = []
    op = criteria.get("operator")
    if isinstance(op, Mapping) and op.get("statistic", {}).get("kill_reachable") is False:
        st = op["statistic"]
        unspent = [p for p in st.get("per_seed", []) if not p.get("es_spent_budget")]
        share = max(
            (float(p["es_budget_share"]) for p in unspent if p.get("es_budget_share") is not None),
            default=0.0,
        )
        out.append(
            f"Operator KILL not reachable: the {st.get('es_arm')} arm spent at most {share:.1%} of its "
            f"{st.get('budget_factor'):g}x-GRPO FPE budget in {len(unspent)} of {len(st.get('seeds', []))} paired "
            "seed(s), so the Operator criterion could only be SUCCESS or INCONCLUSIVE (raise es_updates)"
        )
    return out


# ============================================================================== summary
def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def _interval_text(stat: Mapping[str, Any]) -> str:
    if not isinstance(stat, Mapping) or "mean" not in stat:
        return "-"
    return f"mean {_fmt(stat['mean'], 4)} [{_fmt(stat['lo'], 4)}, {_fmt(stat['hi'], 4)}] (n={stat['n']})"


def render_summary(results: Mapping[str, Any]) -> str:
    """``summary.md``: banner, warnings, start map, one table per arm, gate error rates, criteria."""
    lines = ["# Capability pilot P0 summary", "", f"**{BANNER}**", ""]
    lines.append(
        "Every number below comes from a tiny Llama on synthetic digit-string tasks on CPU. "
        f"GPU metrics: {NOT_TESTED}. Smoke runs (`--smoke`) are plumbing checks, not results."
    )
    lines.append("")
    if results.get("config", {}).get("smoke"):
        lines += [
            "> SMOKE RUN: tiny item counts, short retention horizon; the verdicts below are not meaningful.",
            "",
        ]
    for warning in results.get("warnings", []):
        lines.append(f"> WARNING: {warning}")
    if results.get("warnings"):
        lines.append("")
    splits = results.get("splits", {})
    lines += [
        "## Setup",
        "",
        f"- splits fingerprint: `{splits.get('fingerprint')}` ({splits.get('num_operators')} operators)",
        f"- model: {results.get('sft', {}).get('parameters')} parameters, SFT {results.get('sft', {}).get('steps')} steps",
        f"- channel selection: {results.get('selection', {}).get('scorer')} ({results.get('selection', {}).get('reason')})",
        "- seeds: every seed shares this split, the one start checkpoint, the start map and the informed "
        "channel selection; the seed varies only the RL stage (sampler root, operator seed, es_random draw), so "
        "the paired intervals cover RL-stage randomness only and the Subset verdict concerns this one selection",
    ]
    gc = results.get("gate_choice", {})
    lines.append(
        f"- items_per_retention_slice = {gc.get('items_per_retention_slice')} (worst-case verdicts "
        f"{gc.get('worst_case_verdicts')}, short horizon: {_fmt(gc.get('short_horizon'))})"
    )
    oracle = results.get("oracle", {})
    lines.append(
        f"- oracle: {oracle.get('items_per_slice')} items per T_RULE slice per decision, improvement threshold "
        f"{oracle.get('improvement_threshold')}"
    )
    lines += [
        "",
        "## Start map (start checkpoint)",
        "",
        "| slice | pass@1 | pass@k | sampled |",
        "|---|---|---|---|",
    ]
    for name, s in results.get("start_summary", {}).items():
        lines.append(f"| {name} | {_fmt(s['pass1'])} | {_fmt(s['pass_k'])} | {_fmt(s['sampled_mean'])} |")
    lines += ["", "## Arms", ""]
    for arm in results.get("arms", []):
        f = arm["final"]
        ge = arm["gate_errors"]
        c = arm["counts"]
        lines += [
            f"### {arm['key']}",
            "",
            "| metric | value |",
            "|---|---|",
            f"| updates | {arm['updates']} |",
            f"| space | {arm['space'].get('type')} (dim {arm['space'].get('dim', '-')}) |",
            f"| final A / T_rule / T_task / R_mean | {_fmt(f['A'])} / {_fmt(f['T_rule'])} / {_fmt(f['T_task'])} / {_fmt(f['R_mean'])} |",
            f"| T_rule(0) (start map) | {_fmt(arm['t_rule0'])} |",
            f"| final D(t) | {_fmt(f['D'])} |",
            f"| hack rate total / final block | {_fmt(arm['hack_rate_total'])} / {_fmt(arm['hack_rate_final'])} |",
            f"| decisions | {', '.join(f'{k}={v}' for k, v in sorted(arm['decision_counts'].items()))} |",
            f"| first-verdict accepts / oracle-improved / blocks | {ge['accepted_first']} / {ge['oracle_improved']} / {ge['blocks']} |",
            f"| false-accept rate / false-reject rate | {_fmt(ge['false_accept_rate'])} / {_fmt(ge['false_reject_rate'])} |",
            f"| forward / backward calls (operator) | {c['forward_calls']} / {c['backward_calls']} |",
            f"| forward / backward sequences, FPE | {c['forward_sequences']} / {c['backward_sequences']}, {c['fpe']} |",
            f"| alarm: gap alarms / skipped updates | {arm['alarm']['gap_alarms']} / {arm['alarm']['skipped_updates']} of {arm['alarm']['updates']} |",
            f"| GPU memory / throughput | {NOT_TESTED} |",
            "",
            "| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for b in arm["blocks"]:
            lines.append(
                f"| {b['decision_index']} | {b['updates_done']} | {b['decision']} | {b['reason']} | {b['first_reason']} | "
                f"{_fmt(b['oracle']['diff'], 4)} | {_fmt(b['track']['A'])} | {_fmt(b['track']['T_rule'])} | {_fmt(b['hack_rate_block'])} |"
            )
        lines.append("")
    ex = results.get("gate_exercise")
    if isinstance(ex, Mapping):
        lines += [
            "## Gate exercise (are the error rates meaningful?)",
            "",
            f"- blocks {ex['blocks']}: first-verdict accepts {ex['accepted']}, oracle-improved {ex['oracle_improved']} "
            f"(need >= {ex['min_improved_blocks']}), max oracle diff {_fmt(ex['max_oracle_diff'], 4)} "
            f"(threshold {oracle.get('improvement_threshold')})",
            f"- false-reject rate evaluable: {_fmt(ex['false_reject_evaluable'])}; false-accept rate informative: "
            f"{_fmt(ex['false_accept_informative'])}; exercised: **{_fmt(ex['exercised'])}**",
        ]
        lines += [f"- {problem}" for problem in ex["problems"]]
        lines.append("")
    crit = results.get("criteria", {})
    lines += [
        "## Section-3 criteria (automatic evaluation)",
        "",
        "| criterion | verdict | reason |",
        "|---|---|---|",
    ]
    for name in ("gate", "subset", "operator", "low_start"):
        if name in crit:
            lines.append(f"| {name} | **{crit[name]['verdict']}** | {crit[name]['reason']} |")
    lines.append("")
    for name in ("gate", "subset", "operator", "low_start"):
        if name not in crit:
            continue
        st = crit[name]["statistic"]
        lines.append(f"### {name}")
        lines.append("")
        for key, value in st.items():
            if isinstance(value, Mapping) and "mean" in value:
                lines.append(f"- {key}: {_interval_text(value)}")
        lines.append(f"- definition: {crit.get('definitions', {}).get(name, '')}")
        lines.append("")
    lines += [f"**{BANNER}**", ""]
    return "\n".join(lines)


# ============================================================================== the pilot
def _log(config: PilotConfig, message: str) -> None:
    if config.progress:
        print(f"[pilot] {message}", flush=True)


def _environment(config: PilotConfig) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads_main": config.threads,
        "threads_per_arm": config.threads_per_arm,
        "jobs": config.jobs,
        "gpu": NOT_TESTED,
    }


def run_pilot(config: PilotConfig, output_dir: str | Path) -> dict[str, Any]:
    """Run the P0 pilot (module docstring) and write its outputs to ``output_dir``; returns the
    results dict (also written as ``results.json``). Never downloads anything; CPU only."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    (out / "pilot_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    run_warnings: list[str] = []
    timing: dict[str, Any] = {}
    with _torch_threads(config.threads):
        splits = build_splits(config)
        splits.check_disjoint()
        tokenizer = build_char_tokenizer()
        model = build_model(config, len(tokenizer))
        budget = _budget(config, splits)
        _log(
            config,
            f"splits {splits.fingerprint()[:12]}; model {count_parameters(model)} parameters; SFT {config.sft_steps} steps",
        )
        t = time.perf_counter()
        sft_items = sft_examples(
            splits, config.sft_examples, seed=config.seed, demo_pairs=config.sft_demo_pairs
        )
        sft = train_sft(model, tokenizer, sft_items, config)
        timing["sft_s"] = time.perf_counter() - t
        start_path = out / "start_model.safetensors"
        save_file(
            {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}, str(start_path)
        )
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
        _log(config, "start map: " + ", ".join(f"{k}={v['pass1']:.3f}" for k, v in start.summary().items()))
        t = time.perf_counter()
        selection = select_informed_channels(model, tokenizer, base_sampler, config, budget)
        timing["selection_s"] = time.perf_counter() - t
        _log(config, f"informed channels: {selection['scorer']} ({selection['count']} channels)")
    _, retention = _slices(config, splits)
    gate_choice = choose_retention_items(config, retention)
    if gate_choice["short_horizon"]:
        msg = (
            f"items_per_retention_slice={gate_choice['items_per_retention_slice']} cannot cover the worst-case "
            f"{gate_choice['worst_case_verdicts']} verdicts; later verdicts may be horizon_exhausted"
        )
        if not config.allow_short_horizon:
            raise ValueError(msg + " (set allow_short_horizon or add larger retention_items_candidates)")
        run_warnings.append(msg)
    for name in retention:
        if name in start.slices and start.pass1(name) < 0.5:
            run_warnings.append(
                f"start retention {name} pass@1 = {start.pass1(name):.3f} < 0.5: retention evidence is noisy and "
                "gate verdicts on this slice may reflect the weak start rather than the candidate"
            )
    ctx = _ArmContext(
        config=config.to_dict(),
        start_path=str(start_path),
        start_map=start.to_dict(),
        selection=selection,
        items_per_retention_slice=int(gate_choice["items_per_retention_slice"]),
        output_dir=str(out),
    )
    specs = arm_specs(config)
    if config.eta_probe:
        probe_specs: dict[str, ArmSpec] = {}
        for spec in specs:
            probe_specs.setdefault(spec.operator, spec)
        block = config.base_gate_config().block_updates
        payloads = [(s.to_dict(), dataclasses.asdict(ctx), block, False) for s in probe_specs.values()]
        t = time.perf_counter()
        probe_results = _run_many(payloads, min(config.jobs, len(payloads)))
        per_block = {r["operator"]: max(r["block_seconds"], default=0.0) for r in probe_results}
        setup = {r["operator"]: r["setup_s"] for r in probe_results}
        eta = (
            sum(
                setup[s.operator] + per_block[s.operator] * (config.updates_for(s.operator) // block)
                for s in specs
            )
            / config.jobs
        )
        timing["eta_probe"] = {
            "seconds_per_block": per_block,
            "setup_seconds": setup,
            "probe_s": time.perf_counter() - t,
            "eta_arms_s": eta,
            "note": "one block per operator, timed before the arms; the estimate ignores alarm-triggered extra decisions",
        }
        _log(
            config,
            "ETA probe: "
            + ", ".join(f"{k}={v:.1f}s/block (+{setup[k]:.1f}s setup)" for k, v in per_block.items())
            + f"; arms ETA ~{eta / 60:.1f} min with {config.jobs} job(s)",
        )
    _log(config, f"running {len(specs)} arm(s) with {config.jobs} job(s)")
    t = time.perf_counter()
    arm_results = _run_many([(s.to_dict(), dataclasses.asdict(ctx), None, True) for s in specs], config.jobs)
    timing["arms_s"] = time.perf_counter() - t
    by_key = {r["key"]: r for r in arm_results}
    arms = [by_key[s.key] for s in specs]
    for arm in arms:
        if arm["alarm"]["skipped_updates"] == arm["alarm"]["updates"] and arm["alarm"]["updates"] > 0:
            run_warnings.append(
                f"{arm['key']}: the alarm skipped every update (effective units below min_rollouts); it never could fire"
            )
    exercise = gate_exercise(arms, config.min_improved_blocks)
    run_warnings.extend(f"gate not exercised: {problem}" for problem in exercise["problems"])
    results: dict[str, Any] = {
        "version": _RESULTS_VERSION,
        "banner": BANNER,
        "status": "complete",
        "config": config.to_dict(),
        "environment": _environment(config),
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
            "start_fingerprint": fingerprint(model),
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
            "definition": (
                "candidate and incumbent scored with the exact verifier on the same fresh T_RULE items "
                "(purpose 'oracle', index = decision index, tracker pool minus start-map and tracker uids); "
                "improved iff mean over T_RULE slices of (candidate - incumbent) > improvement_threshold; "
                "accepted iff the first verdict classifies as 'accept' (gated and ungated arms alike); "
                "false-accept rate = accepted and not improved / not-improved blocks; false-reject rate = "
                "not accepted and improved / improved blocks"
            ),
        },
        "counting": {
            "operator_calls": "as reported by the operators: GRPO 2 forward calls (generate + training forward) and 1 backward per update; ES 1 generation call per member (plus extra batches)",
            "fpe": "sequence level: every generated sequence and every training-forward sequence = 1 forward, every backward sequence = 2 forwards",
        },
        "arms": arms,
        "gate_exercise": exercise,
        "warnings": run_warnings,
    }
    results["criteria"] = evaluate_criteria(results)
    run_warnings.extend(_criteria_warnings(results["criteria"]))
    timing["total_s"] = time.perf_counter() - t0
    results["timing"] = timing
    results = _finite(results)
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    (out / "summary.md").write_text(render_summary(results), encoding="utf-8")
    _log(config, f"done in {timing['total_s']:.1f}s; wrote {out / 'results.json'} and {out / 'summary.md'}")
    return results


def _finite(value: Any) -> Any:
    """Recursively replace non-finite floats by ``None`` (strict JSON) and tuples by lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def expected_outputs(output_dir: str | Path, config: PilotConfig) -> list[Path]:
    """Every file :func:`run_pilot` must produce for ``config``."""
    out = Path(output_dir)
    files = [
        out / "results.json",
        out / "summary.md",
        out / "pilot_config.json",
        out / "start_model.safetensors",
    ]
    files += [out / "decisions" / f"{s.file_stem}.jsonl" for s in arm_specs(config)]
    return files
