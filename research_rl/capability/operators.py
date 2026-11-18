"""Backprop update operators for the gated loop (``docs/research_rl/CAPABILITY.md`` 1.9 / 1.16).

:class:`TorchGRPOOperator` is a minimal on-policy GRPO in plain PyTorch, the backprop control arm
of the pilot. It implements :class:`~research_rl.capability.types.UpdateOperator`; one
:meth:`~TorchGRPOOperator.step` (update ``t``) does exactly this:

1. **Prompts.** ``prompts_per_update`` TRAIN items ``sampler.draw(Split.TRAIN, P,
   purpose=TRAIN_PURPOSE, index=t)`` (fresh per update, a pure function of ``t``); every item's
   answer must satisfy the truncation contract ``len(answer) + 1 <= max_new_tokens``
   (``ValueError``), and construction checks ``max_new_tokens >= operators.max_len + 1``
   (:func:`~research_rl.capability.probe.check_truncation_budget`).
2. **Sampling.** ``group_size`` completions per prompt (rows prompt-major: row ``i`` belongs to
   prompt ``i // group_size``), one batched ``model.generate`` call: prompts encoded with
   ``tokenizer(prompt, add_special_tokens=False)`` and **left-padded** with the pad id (EOS if
   none) plus an attention mask, exactly as
   :func:`~research_rl.evaluation.capability_retention.greedy_generate` builds its batches; every
   ``GenerationConfig`` on the model is neutralised for the call; pure ancestral sampling
   (``do_sample=True``, ``temperature``, ``top_k=0``, ``top_p=1``, one beam); model in eval mode,
   every submodule's training flag restored afterwards. The call runs inside
   ``torch.random.fork_rng`` with the CPU generator (and, for a CUDA model, that device's
   generator; NOT_TESTED) seeded with :func:`sample_seed` ``(seed, t)``, so the global RNG is
   untouched and the samples of update ``t`` do not depend on call history.
3. **Completions and truncation.** Each generated row is cut at its first EOS: the *action
   tokens* are the tokens up to and including that EOS (all ``max_new_tokens`` tokens when there
   is none), the completion text is the decoded tokens before it (``skip_special_tokens=True``, as
   in ``greedy_generate``). A completion is *truncated* iff its row has no EOS (it filled the
   budget) **or** ``VerifierScoreFn.truncated`` flags its text (the gate's re-encoding rule; both
   agree unless sampled special tokens other than EOS were dropped by decoding, the case the
   EOS flag adds).
4. **Scoring.** V1 and V2 through :class:`~research_rl.capability.tasks.VerifierScoreFn`
   ``.score_completions(texts, items, truncated)`` (never raw ``verify()``), so a truncated
   completion scores 0 under both verifiers, as in the gate. ``reward_v1`` / ``reward_v2`` are
   the means over all rows; ``v1_pass`` / ``v2_pass`` are ``score == 1``.
5. **Advantages** (:func:`group_advantages`): per group of ``group_size`` rows,
   ``A_i = (r_i - mean) / (std + ADVANTAGE_EPS)`` on the V1 rewards, in float64, ``std`` the
   population standard deviation (ddof 0), ``ADVANTAGE_EPS = 1e-6``; a group with ``std == 0``
   (all rewards equal, e.g. all 0) gets advantage exactly 0. With ``screen=True`` rows that pass
   V1 and fail V2 (:func:`~research_rl.capability.alarm.screen_mask`) then get advantage 0
   (excluded from the update, not penalised; the group statistics still include them, as TRL's
   screen zeroes advantages after normalisation); ``report.screened`` counts them (0 when
   ``screen=False``).
6. **Loss.** One forward of ``[left-padded prompt | action tokens, right-padded]`` with the
   attention mask (prompt pads and the pads after each row's actions are 0) and ``position_ids =
   cumsum(mask) - 1`` (the positions generation used). Token-level policy gradient (on-policy,
   so the importance ratio is 1 and there is no clipping or KL term)::

       loss = - sum_{i, j} A_i * log pi(a_ij | context) * m_ij  /  sum_{i, j} m_ij

   where ``log pi`` is ``log_softmax(logits / temperature)`` in float32 (the distribution the
   samples came from), ``m_ij`` is 1 exactly on action tokens (prompt and padding masked), and the
   normaliser is the total number of action tokens of the batch (every row, also rows with
   advantage 0). The model's training mode is left as the caller set it for this forward (so
   dropout, if the model has any and is in train mode, is active); the forward runs inside
   ``torch.random.fork_rng`` with the CPU generator (and a CUDA model's device generator;
   NOT_TESTED) seeded with :func:`loss_seed` ``(seed, t)``, so the dropout masks of update ``t``
   are a function of ``(seed, t)`` and the global RNG is untouched (the backward reuses the masks
   saved by the forward and draws nothing).
7. **Update.** ``model.zero_grad(set_to_none=True)``, ``loss.backward()``, exactly one
   ``optimizer.step()``, then ``model.zero_grad(set_to_none=True)`` again (no gradient is kept
   between updates, also for parameters outside the optimizer). The step happens even when every
   advantage is 0 (all rollouts screened or every group with zero variance), and it is **not** a
   no-op in general: the gradient is then exactly 0, which leaves the parameters unchanged only
   for an optimizer without state and without weight decay (e.g. the default AdamW before its
   first update with signal); once AdamW holds non-zero moments, the zero-gradient step still
   moves every parameter with a non-zero first moment (``m`` decays, ``m_hat / (sqrt(v_hat) +
   eps)`` stays non-zero) and advances the step counter.

Optimizer: ``optimizer=None`` builds ``torch.optim.AdamW(trainable parameters, lr=lr,
weight_decay=0.0)``; ``lr`` is used only for that default. Any ``torch.optim.Optimizer`` over
parameters of ``model`` works, including a
:class:`~research_rl.optim.optimizer_wrapper.HookedOptimizer` (e.g. with a
:class:`~research_rl.optim.update_interface.MaskRule` from
:func:`~research_rl.models.ffn_channels.channel_mask` to train a channel subset: masked elements
stay bit-identical). ``parameter_names()`` are the model's names of the optimizer's parameters
(the rollback scope of the gate).

Report: ``forward_passes = 2`` per update (one batched generation call + one training forward;
the same "calls" convention as :class:`~research_rl.capability.es.ESOperator`, which counts
generation calls), ``backward_passes = 1``; ``completion_lengths`` = tokens before the first EOS
(``max_new_tokens`` for truncated rows); ``extra["alarm_units"]`` = the prompt index of every
rollout (rollouts of one prompt are correlated; the alarm counts prompts as independent units);
``extra`` also holds the advantages, the loss, the item uids and the sample seed.

State: :meth:`~TorchGRPOOperator.state_dict` holds the configuration and counters only. The
optimizer state is **not** part of it (the :class:`~research_rl.capability.loop.GatedLoop`
checkpoints it with the current weights); neither sampling nor the training forward's dropout
needs RNG state (both are seeded per update index), so one update is a pure function of
``(seed, update_index)``, the weights and the optimizer state, whatever the global RNG holds.

Verified on CPU with tiny models only (``tests/research_rl/test_capability_operators.py``); GPU
behaviour is NOT_TESTED.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationConfig

from research_rl.capability.alarm import UNITS_KEY, screen_mask
from research_rl.capability.probe import check_truncation_budget
from research_rl.capability.tasks import ItemSampler, VerifierScoreFn
from research_rl.capability.types import GateDecision, Item, OperatorReport, Split
from research_rl.evaluation.capability_retention import _neutralise_generation_configs
from research_rl.rewards.verifier_interface import Verifier

__all__ = [
    "ADVANTAGE_EPS",
    "TRAIN_PURPOSE",
    "Rollouts",
    "TorchGRPOOperator",
    "group_advantages",
    "loss_seed",
    "sample_seed",
]

TRAIN_PURPOSE = "grpo"
"""``ItemSampler`` purpose of the GRPO training prompts (index = update index)."""
ADVANTAGE_EPS = 1e-6
"""Added to the group standard deviation in :func:`group_advantages`."""
_STATE_VERSION = 1


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def sample_seed(seed: int, update_index: int) -> int:
    """Sampling seed of update ``update_index``: the first 8 bytes (little endian, top bit
    cleared) of ``SHA-256("research_rl.capability.grpo:{seed}:{update_index}")``."""
    digest = hashlib.sha256(f"research_rl.capability.grpo:{seed}:{update_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def loss_seed(seed: int, update_index: int) -> int:
    """Seed of the training forward of update ``update_index`` (its dropout masks): the first 8
    bytes (little endian, top bit cleared) of ``SHA-256("research_rl.capability.grpo.loss:{seed}:
    {update_index}")``; a different domain from :func:`sample_seed`, so the two streams differ."""
    digest = hashlib.sha256(f"research_rl.capability.grpo.loss:{seed}:{update_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def group_advantages(
    rewards: Sequence[float], group_size: int, keep: Sequence[bool] | None = None
) -> list[float]:
    """Group-normalised advantages (see the module docstring, step 5).

    ``rewards`` holds consecutive groups of ``group_size`` finite values. Per group, in float64:
    ``A_i = (r_i - mean) / (std + ADVANTAGE_EPS)`` with the population std; exactly 0 for every
    member when ``std == 0``. Then ``A_i = 0`` wherever ``keep[i]`` is False (``keep`` has one
    bool per reward; ``None`` keeps all). ``ValueError`` for a length that is not a multiple of
    ``group_size``, ``group_size < 1``, non-finite rewards or a ``keep`` of the wrong length.
    """
    if not _is_int(group_size) or group_size < 1:
        raise ValueError(f"group_size must be an int >= 1, got {group_size!r}")
    values = [float(r) for r in rewards]
    if len(values) % group_size:
        raise ValueError(f"{len(values)} rewards cannot be grouped by group_size={group_size}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("rewards must be finite")
    if keep is not None and len(keep) != len(values):
        raise ValueError(f"keep has {len(keep)} entries for {len(values)} rewards")
    out: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        mean = math.fsum(group) / group_size
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in group) / group_size)
        if std == 0.0:
            out.extend([0.0] * group_size)
        else:
            out.extend((v - mean) / (std + ADVANTAGE_EPS) for v in group)
    if keep is not None:
        out = [a if bool(k) else 0.0 for a, k in zip(out, keep)]
    return out


@dataclass(frozen=True)
class Rollouts:
    """The sampled and scored rollouts of one update (rows prompt-major).

    ``prompt_ids[i]`` are the prompt tokens of row ``i``, ``actions[i]`` its action tokens (up to
    and including the first EOS; all ``max_new_tokens`` tokens when ``no_eos[i]``), ``texts[i]``
    the decoded completion before the EOS, ``truncated[i]`` the truncation flag, ``v1`` / ``v2``
    the 0/1 scores and ``units[i]`` the prompt index. ``items`` are the update's prompts (one per
    group) and ``seed`` the sampling seed.
    """

    update_index: int
    seed: int
    items: tuple[Item, ...]
    prompt_ids: tuple[tuple[int, ...], ...]
    actions: tuple[tuple[int, ...], ...]
    no_eos: tuple[bool, ...]
    texts: tuple[str, ...]
    truncated: tuple[bool, ...]
    v1: tuple[float, ...]
    v2: tuple[float, ...]
    units: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.actions)
        for name in ("prompt_ids", "no_eos", "texts", "truncated", "v1", "v2", "units"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Rollouts.{name} has {len(getattr(self, name))} rows, actions has {n}")
        if any(len(a) == 0 for a in self.actions):
            raise ValueError("every row needs at least one action token")

    @property
    def lengths(self) -> tuple[int, ...]:
        """Completion tokens before the first EOS, per row (all action tokens without EOS)."""
        return tuple(len(a) - (0 if free else 1) for a, free in zip(self.actions, self.no_eos))


def _accepts_position_ids(model: nn.Module) -> bool:
    try:
        return "position_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


class TorchGRPOOperator:
    """Minimal on-policy GRPO :class:`~research_rl.capability.types.UpdateOperator` (module
    docstring for the exact semantics).

    Args:
        model: the policy (a causal LM with ``generate``); :meth:`step` must receive this object.
        tokenizer: its tokenizer (needs an EOS id; pad id or EOS for padding).
        sampler: TRAIN prompt source.
        v1: the training-reward verifier (may be leaky). v2: the strict verifier (screening, the
            alarm feed and ``reward_v2``).
        group_size: completions per prompt (>= 1; 1 gives advantage 0 everywhere).
        prompts_per_update: prompts per update (>= 1).
        lr: learning rate of the default AdamW (finite, > 0).
        max_new_tokens: sampling budget (>= ``operators.max_len + 1``).
        temperature: sampling temperature (finite, > 0); also divides the logits of the loss.
        optimizer: optional optimizer over parameters of ``model``.
        screen: zero the advantage of V1-pass / V2-fail rollouts.
        seed: root of the per-update sampling seeds.
        name: operator name (``"grpo"``).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        sampler: ItemSampler,
        v1: Verifier,
        v2: Verifier,
        *,
        group_size: int,
        prompts_per_update: int,
        lr: float,
        max_new_tokens: int,
        temperature: float,
        optimizer: torch.optim.Optimizer | None = None,
        screen: bool = True,
        seed: int,
        name: str = "grpo",
    ) -> None:
        if not isinstance(sampler, ItemSampler):
            raise TypeError(f"sampler must be an ItemSampler, got {type(sampler).__name__}")
        for label, verifier in (("v1", v1), ("v2", v2)):
            if not callable(getattr(verifier, "verify", None)):
                raise TypeError(f"{label} must be a verifier with verify(), got {type(verifier).__name__}")
        for label, value in (
            ("group_size", group_size),
            ("prompts_per_update", prompts_per_update),
            ("max_new_tokens", max_new_tokens),
        ):
            if not _is_int(value) or value < 1:
                raise ValueError(f"{label} must be an int >= 1, got {value!r}")
        for label, number in (("lr", lr), ("temperature", temperature)):
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not (math.isfinite(number) and number > 0)
            ):
                raise ValueError(f"{label} must be a finite number > 0, got {number!r}")
        if not isinstance(screen, bool):
            raise ValueError(f"screen must be a bool, got {screen!r}")
        if not _is_int(seed):
            raise ValueError(f"seed must be an int, got {seed!r}")
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs an eos_token_id (completions end at EOS)")
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.v1 = v1
        self.v2 = v2
        self.group_size = int(group_size)
        self.prompts_per_update = int(prompts_per_update)
        self.lr = float(lr)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.screen = screen
        self.seed = int(seed)
        self.name = name
        self.v1_score = VerifierScoreFn(tokenizer, v1, max_new_tokens=self.max_new_tokens, batch_size=1)
        self.v2_score = VerifierScoreFn(tokenizer, v2, max_new_tokens=self.max_new_tokens, batch_size=1)
        check_truncation_budget(sampler, self.v2_score)
        if optimizer is None:
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                raise ValueError("the model has no trainable parameters")
            optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=0.0)
        self.optimizer: torch.optim.Optimizer | None = optimizer
        names = {id(p): n for n, p in model.named_parameters()}
        selected: set[int] = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) not in names:
                    raise ValueError("the optimizer holds a parameter that is not a parameter of the model")
                selected.add(id(p))
        self._names = [n for n, p in model.named_parameters() if id(p) in selected]
        self._pass_positions = _accepts_position_ids(model)
        self.steps_done = 0
        self.last_update_index: int | None = None
        self.forward_passes_total = 0
        self.backward_passes_total = 0

    # ---------------------------------------------------------------- protocol
    def parameter_names(self) -> list[str]:
        """Model names of the optimizer's parameters, in ``named_parameters`` order."""
        return list(self._names)

    def on_decision(self, decision: GateDecision) -> None:
        """No-op (GRPO keeps no weight copy of its own; the controller restores weights and
        optimizer state itself). Validates the type."""
        if not isinstance(decision, GateDecision):
            raise TypeError(f"decision must be a GateDecision, got {decision!r}")

    def config(self) -> dict[str, Any]:
        """The hyper-parameters as a JSON-able dict (checked by :meth:`load_state_dict`)."""
        return {
            "name": self.name,
            "group_size": self.group_size,
            "prompts_per_update": self.prompts_per_update,
            "lr": self.lr,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "screen": self.screen,
            "seed": self.seed,
            "parameter_names": list(self._names),
        }

    def state_dict(self) -> dict[str, Any]:
        """Configuration and counters (JSON-able). Optimizer state is not included (see the
        module docstring)."""
        return {
            "version": _STATE_VERSION,
            "config": self.config(),
            "steps_done": self.steps_done,
            "last_update_index": self.last_update_index,
            "forward_passes_total": self.forward_passes_total,
            "backward_passes_total": self.backward_passes_total,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the counters after checking that the configuration matches exactly
        (``ValueError`` otherwise; nothing changes then)."""
        if state.get("version") != _STATE_VERSION:
            raise ValueError(f"unsupported GRPO state version {state.get('version')!r}")
        if dict(state.get("config", {})) != self.config():
            raise ValueError(
                f"GRPO config mismatch: state has {state.get('config')}, operator has {self.config()}"
            )
        counters = {}
        for key in ("steps_done", "forward_passes_total", "backward_passes_total"):
            value = state.get(key)
            if not _is_int(value) or value < 0:
                raise ValueError(f"invalid {key} {value!r}")
            counters[key] = int(value)
        last = state.get("last_update_index")
        if last is not None and (not _is_int(last) or last < 0):
            raise ValueError(f"invalid last_update_index {last!r}")
        self.steps_done = counters["steps_done"]
        self.forward_passes_total = counters["forward_passes_total"]
        self.backward_passes_total = counters["backward_passes_total"]
        self.last_update_index = last

    # ---------------------------------------------------------------- rollouts
    def _check_model(self, model: nn.Module) -> None:
        if model is not self.model:
            raise ValueError("step() / rollouts() got a model that is not the operator's model")

    def items(self, update_index: int) -> list[Item]:
        """The update's prompts: ``sampler.draw(Split.TRAIN, prompts_per_update,
        purpose=TRAIN_PURPOSE, index=update_index)`` (truncation contract checked)."""
        items = self.sampler.draw(
            Split.TRAIN, self.prompts_per_update, purpose=TRAIN_PURPOSE, index=update_index
        )
        long = [it.uid for it in items if len(it.answer) + 1 > self.max_new_tokens]
        if long:
            raise ValueError(
                f"answers too long for max_new_tokens={self.max_new_tokens} (need len(answer) + 1 <= "
                f"max_new_tokens): {long[:3]}"
            )
        return items

    def _devices(self, model: nn.Module) -> tuple[torch.device, list[int]]:
        device = next(model.parameters()).device
        if device.type == "cuda":
            return device, [device.index if device.index is not None else torch.cuda.current_device()]
        return device, []

    @torch.no_grad()
    def _generate(self, model: nn.Module, prompts: Sequence[Sequence[int]], seed: int) -> list[list[int]]:
        """Raw sampled continuation ids per row (each at most ``max_new_tokens`` long; after a
        row's first EOS the remaining columns are padding). See the module docstring, step 2.
        Subclasses may override it (tests script completions this way)."""
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos
        device, devices = self._devices(model)
        width = max(len(p) for p in prompts)
        input_ids = torch.full((len(prompts), width), int(pad), dtype=torch.long)
        attention = torch.zeros(len(prompts), width, dtype=torch.long)
        for row, ids in enumerate(prompts):
            input_ids[row, width - len(ids) :] = torch.tensor(list(ids), dtype=torch.long)
            attention[row, width - len(ids) :] = 1
        config = GenerationConfig(
            do_sample=True,
            temperature=self.temperature,
            top_k=0,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=eos,
            pad_token_id=int(pad),
        )
        modes = [(module, module.training) for module in model.modules()]
        saved = _neutralise_generation_configs(model)
        try:
            with torch.random.fork_rng(devices=devices):
                model.eval()
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(seed)
                generated = model.generate(  # type: ignore[operator]
                    input_ids=input_ids.to(device),
                    attention_mask=attention.to(device),
                    generation_config=config,
                )
        finally:
            for module, was_training in modes:
                module.training = was_training
            for owner, original in saved:
                setattr(owner, "generation_config", original)  # noqa: B010
        rows: list[list[int]] = generated[:, width:].tolist()
        return rows

    def rollouts(self, model: nn.Module, update_index: int) -> Rollouts:
        """Sample and score the rollouts of update ``update_index`` (steps 1-4 of the module
        docstring) without changing any weight. Deterministic in ``(seed, update_index)`` and the
        current weights; the global RNG state is unchanged."""
        self._check_model(model)
        if not _is_int(update_index) or update_index < 0:
            raise ValueError(f"update_index must be an int >= 0, got {update_index!r}")
        items = self.items(update_index)
        prompt_rows: list[tuple[int, ...]] = []
        row_items: list[Item] = []
        units: list[int] = []
        for i, item in enumerate(items):
            ids = tuple(int(t) for t in self.tokenizer(item.prompt, add_special_tokens=False)["input_ids"])
            if not ids:
                raise ValueError(f"item {item.uid!r} has an empty prompt")
            for _ in range(self.group_size):
                prompt_rows.append(ids)
                row_items.append(item)
                units.append(i)
        seed = sample_seed(self.seed, update_index)
        raw = self._generate(model, prompt_rows, seed)
        if len(raw) != len(prompt_rows):
            raise ValueError(f"generation returned {len(raw)} rows for {len(prompt_rows)} prompts")
        eos = int(self.tokenizer.eos_token_id)
        actions: list[tuple[int, ...]] = []
        texts: list[str] = []
        no_eos: list[bool] = []
        for row in raw:
            row = [int(t) for t in row]
            if eos in row:
                cut = row.index(eos)
                actions.append(tuple(row[: cut + 1]))
                body = row[:cut]
                no_eos.append(False)
            else:
                if len(row) != self.max_new_tokens:
                    raise ValueError(
                        f"a generated row without EOS has {len(row)} tokens, expected max_new_tokens="
                        f"{self.max_new_tokens}"
                    )
                actions.append(tuple(row))
                body = row
                no_eos.append(True)
            texts.append(self.tokenizer.decode(body, skip_special_tokens=True))
        truncated = [a or b for a, b in zip(no_eos, self.v2_score.truncated(texts))]
        v1 = self.v1_score.score_completions(texts, row_items, truncated)
        v2 = self.v2_score.score_completions(texts, row_items, truncated)
        out = Rollouts(
            update_index=update_index,
            seed=seed,
            items=tuple(items),
            prompt_ids=tuple(prompt_rows),
            actions=tuple(actions),
            no_eos=tuple(no_eos),
            texts=tuple(texts),
            truncated=tuple(truncated),
            v1=tuple(v1),
            v2=tuple(v2),
            units=tuple(units),
        )
        return out

    # ---------------------------------------------------------------- update
    def advantages(self, rollouts: Rollouts) -> tuple[list[float], list[bool]]:
        """``(advantages, keep)`` of the rollouts (step 5; ``keep`` all True when ``screen`` is
        off)."""
        v1_pass = [s == 1.0 for s in rollouts.v1]
        v2_pass = [s == 1.0 for s in rollouts.v2]
        keep = screen_mask(v1_pass, v2_pass) if self.screen else [True] * len(v1_pass)
        return group_advantages(rollouts.v1, self.group_size, keep if self.screen else None), keep

    def policy_loss(self, model: nn.Module, rollouts: Rollouts, advantages: Sequence[float]) -> torch.Tensor:
        """The token-level policy-gradient loss of step 6 (a scalar with a graph). The forward runs
        under ``fork_rng`` seeded with :func:`loss_seed` ``(seed, rollouts.update_index)``."""
        eos = self.tokenizer.eos_token_id
        pad = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos)
        device = next(model.parameters()).device
        rows = len(rollouts.actions)
        width = max(len(p) for p in rollouts.prompt_ids)
        span = max(len(a) for a in rollouts.actions)
        input_ids = torch.full((rows, width + span), pad, dtype=torch.long)
        mask = torch.zeros(rows, width + span, dtype=torch.long)
        for r, (prompt, action) in enumerate(zip(rollouts.prompt_ids, rollouts.actions)):
            input_ids[r, width - len(prompt) : width] = torch.tensor(prompt, dtype=torch.long)
            mask[r, width - len(prompt) : width] = 1
            input_ids[r, width : width + len(action)] = torch.tensor(action, dtype=torch.long)
            mask[r, width : width + len(action)] = 1
        kwargs: dict[str, Any] = {
            "input_ids": input_ids.to(device),
            "attention_mask": mask.to(device),
            "use_cache": False,
        }
        if self._pass_positions:
            kwargs["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0).to(device)
        _, devices = self._devices(model)
        seed = loss_seed(self.seed, rollouts.update_index)
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            out = model(**kwargs)
        logits = out.logits[:, width - 1 : width + span - 1, :].float() / self.temperature
        targets = input_ids[:, width:].to(device)
        logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        action_mask = mask[:, width:].to(device=device, dtype=logp.dtype)
        adv = torch.tensor(list(advantages), dtype=logp.dtype, device=device)
        return -(adv[:, None] * logp * action_mask).sum() / action_mask.sum()

    def step(self, model: nn.Module, update_index: int) -> OperatorReport:
        """One GRPO update in place (module docstring); returns its report."""
        rollouts = self.rollouts(model, update_index)
        adv, keep = self.advantages(rollouts)
        assert self.optimizer is not None
        model.zero_grad(set_to_none=True)
        loss = self.policy_loss(model, rollouts, adv)
        loss.backward()
        self.optimizer.step()
        model.zero_grad(set_to_none=True)
        v1_pass = tuple(s == 1.0 for s in rollouts.v1)
        v2_pass = tuple(s == 1.0 for s in rollouts.v2)
        screened = sum(1 for k in keep if not k) if self.screen else 0
        self.steps_done += 1
        self.last_update_index = update_index
        self.forward_passes_total += 2
        self.backward_passes_total += 1
        n = len(rollouts.v1)
        return OperatorReport(
            update_index=update_index,
            reward_v1=math.fsum(rollouts.v1) / n,
            reward_v2=math.fsum(rollouts.v2) / n,
            v1_pass=v1_pass,
            v2_pass=v2_pass,
            completion_lengths=rollouts.lengths,
            screened=screened,
            forward_passes=2,
            backward_passes=1,
            extra={
                "operator": self.name,
                UNITS_KEY: list(rollouts.units),
                "advantages": list(adv),
                "loss": float(loss.detach()),
                "uids": [it.uid for it in rollouts.items],
                "sample_seed": rollouts.seed,
                "truncated": sum(1 for t in rollouts.truncated if t),
                "group_size": self.group_size,
            },
        )
