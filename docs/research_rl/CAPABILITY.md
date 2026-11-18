# Capability-gated training (`research_rl.capability`)

Status: **in development**. Everything in this package is verified on CPU with tiny models only.
Nothing here has run on a GPU or on a real 7-8B model: every GPU / large-model statement is
**NOT_TESTED**. Results produced by the tiny-model pilot say nothing about 7B models.

## 0. Paradigm and the defaults this build assumes

Training reward is not the measure of learning. A system should start from low capability,
acquire a skill from experience, keep it, and use it on unseen tasks and unseen rule
combinations. Reward optimisation serves that; it does not replace measuring it.

The user has not yet confirmed three readings; the build takes these defaults and makes each one
a configuration switch:

| Question | Default in this build | Switch |
|---|---|---|
| 置信度回溯 / 獎勵幻覺 | Update-level gate on **capability**: a block of updates is accepted only with statistical confidence that held-out transfer rose and no retention slice fell by more than a margin. "Reward hallucination" = training reward rising while that capability does not (divergence D(t)). | `GateConfig` (`require_transfer_gain`, margins, alpha); the gate can be disabled (`gate.enabled=False`) to get the ungated control arm. |
| "不依賴反向傳播" | Applies to the **weight write**: the primary update operator is forward-only (evolution strategies over a selected FFN-channel subset). Channel selection is forward-only too. A backprop GRPO arm runs behind the *same* gate as the control. | `operator` = `es` / `grpo`; `scores` = forward-only scorers only (a Fisher scorer is not built). |
| Testbed | Synthetic **operator composition** with invented operator names, held-out pairs and deeper nesting; a deterministic verifier; a planted verifier leak to know the ground truth. | `tasks` config; the gate and the TRL callback work with any `ScoreFn` and item source. |

Out of scope for v1: rollout-level backtracking (the model backing up inside a generation),
abstention outcomes, LLM judges, Fisher-style (one-backward) selectors.

## 1. Package layout and public API (contract for implementers)

Shared data types and protocols live in `research_rl/capability/types.py` (already written; do
not change them without updating every module). Each module below lists its public API; names
and signatures are binding. Every module has a test file `tests/research_rl/test_capability_<module>.py`.

### 1.1 `tasks.py` — operator-composition task generators and splits

- Atomic operators on digit strings (length 3-7 by default), each with an invented lowercase
  name of 3 letters. At least 10 length-preserving operators, so compositions are well defined,
  e.g. reverse, rotate-left-1, rotate-right-1, +1 mod 10 per digit, -1 mod 10 per digit,
  nine's complement, swap adjacent pairs, sort ascending, sort descending, mirror-first-half,
  replace each digit by (digit * 3) mod 10. No two operators may be identical functions on the
  sampled input domain (checked at construction).
- `OperatorSet(seed: int = 0, num_operators: int = 10)`: deterministic name assignment
  (invented names must not be common English words and must not collide). `apply(rule, x) -> str`.
- `CompositionSplits.build(operators, *, seed, train_fraction=0.5, audit_fraction=0.2, depth3_count=...)`:
  partitions all ordered pairs (f, g), f != g, into TRAIN pairs, T_RULE pairs (unseen pairs) and
  AUDIT pairs, **disjoint at the rule level**. T_RULE depth-3 rules `f(g(h))` are built so that
  both contiguous sub-pairs (f, g) and (g, h) are TRAIN pairs (deeper nesting of trained rules)
  while the triple itself is never trained; AUDIT depth-3 rules contain at least one AUDIT pair.
  Invariant: no AUDIT pair is a contiguous sub-chain of any non-AUDIT rule, and no rule appears
  in two splits. Exposes `rules(split) -> tuple[tuple[str,...], ...]`,
  `slices() -> dict[slice_name, Split]` and `check_disjoint() -> None` (raises on any violation).
- Slices: `train/pairs`, `t_rule/unseen_pairs`, `t_rule/depth3`, `t_task/<family>` (at least one
  held-out family that is not a composition, e.g. "which position holds the maximum digit" or
  "count digits > 4"; it must be learnable but never trained), `retention/atomic` (single
  operators), `retention/copy` (echo the input), `audit/pairs`, `audit/depth3`.
- Prompt format (characters must be in `research_rl.models.tiny.DEFAULT_CHARS`):
  `"zib(mok(4721))="` → answer `"1274"`; atomic: `"zib(4721)="`; families define their own
  short prompt with `=` as the last character. Answers are digit strings, no spaces.
- `ItemSampler(splits, *, root_seed)`: `draw(split, n, *, purpose: str, index: int, slice: str | None = None) -> list[Item]`.
  Deterministic in `(root_seed, split, slice, purpose, index)`; items for different
  `(purpose, index)` are drawn from independent streams (fresh items per decision). Uids encode
  rule + input so a test can check freshness. **AUDIT**: `draw(Split.AUDIT, ...)` raises
  `AuditAccessError` unless the sampler was built with `allow_audit=True` (only the final
  report path constructs one).
- `sft_examples(splits, n, *, seed, demo_pairs: int = 0) -> list[Item]`: supervised data for
  building a low-start pilot model: atomic operators, the copy task and the held-out family's
  *atomic prerequisites only*; `demo_pairs` > 0 adds a few composed examples drawn **only from
  TRAIN pairs** (never T_RULE / AUDIT / T_TASK).
- Verifiers (implement `research_rl.rewards.verifier_interface.BaseVerifier`):
  `ExactAnswerVerifier` (V2: the completion, stripped of surrounding whitespace and an optional
  trailing terminator, must equal the answer) and `SuffixLeakVerifier(k=2)` (planted V1 leak:
  passes when the last `k` answer characters match the last `k` characters of the completion).
- `make_score_fn(tokenizer, verifier, *, max_new_tokens, batch_size) -> ScoreFn` using
  `research_rl.evaluation.capability_retention.greedy_generate`.

### 1.2 `stats.py` — paired statistics (pure functions, no torch dependency needed)

- `sign_test_p_value(n01, n10) -> float`: exact one-sided binomial p-value of H0 "candidate not
  better" (P[X >= n01], X ~ Bin(n01+n10, 1/2)); 1.0 when there are no discordant pairs.
- `paired_gain_lower_bound(paired: PairedScores | Sequence[PairedScores], alpha) -> SliceTest`:
  binary scores → exact sign test decides `passed` (p <= alpha); `bound` = Clopper-Pearson-based
  conditional bound `(2 p_L - 1) * m / n` (document that the bound is conditional on the number
  of discordant pairs `m`; the decision is exact). Non-binary scores → paired bootstrap with a
  fixed seed **and** a documented coverage check in tests (or a t-bound); method name recorded.
- `noninferiority_upper_bound(paired, margin, alpha) -> SliceTest`: H0 "drop >= margin" for
  binary paired data via **Tango's score test** (restricted MLE) for the paired difference;
  `passed` iff H0 rejected at alpha; `bound` = the score-interval upper limit of the drop.
- `required_paired_n(delta, discordant_rate, alpha=0.05, power=0.8) -> int`
  (Connor 1987: n = (z_a sqrt(q) + z_b sqrt(q - delta^2))^2 / delta^2, ceil).
- `AlphaLedger(total_alpha, schedule)` with schedules `"per_decision"` (alpha every time; the
  guarantee is per decision only), `"spending"` (alpha_k = total * 6 / (pi^2 k^2), k = 1, 2, ...;
  run-level family-wise guarantee for accepts), `next_alpha() -> float`, `spent`, `state_dict()` /
  `load_state_dict()`. Retention tests in one decision share that decision's alpha: each slice is
  tested at alpha (intersection-union needs no split).
- **Calibration tests are mandatory**: by simulation (fixed seeds, >= 4000 replications, fast
  enough for CI), the false-accept rate of `paired_gain_lower_bound` at true gain 0 and the
  false-pass rate of `noninferiority_upper_bound` at true drop = margin must not exceed alpha by
  more than the Monte-Carlo tolerance, for n in {50, 200} and discordance rates {0.1, 0.3}.

### 1.3 `gate.py` — the pure verdict

- `GateConfig` (dataclass, JSON-able): `enabled=True`, `alpha=0.05`, `alpha_schedule="spending"`,
  `retention_margin=0.02` (absolute pass-rate points), per-slice overrides
  `retention_margins: dict[str, float]`, `require_transfer_gain=True`, `hold_limit=3`
  (consecutive HOLDs before a forced ROLLBACK), `shrink_lambdas=(0.5, 0.25)`,
  `max_group_rollbacks=4`, `rollback_unit=RollbackUnit.PARAMETERS`, `block_updates=K`,
  `items_per_transfer_slice`, `items_per_retention_slice`, `alarm_triggers_decision=True`.
- `CapabilityGate(config, ledger: AlphaLedger)`: `verdict(evidence: GateEvidence) -> GateVerdict`
  consumes one alpha from the ledger per call (the shrink / group re-tests of one procedure are
  separate tests and each consumes alpha — documented), pools transfer slices, tests every
  retention slice; `state_dict()` / `load_state_dict()`.

### 1.4 `alarm.py` — tier-1 verifier-disagreement alarm

- `DisagreementAlarm(window=..., threshold=..., min_rollouts=...)`: `update(report: OperatorReport) -> AlarmState`;
  statistic g(t) = pass_V1 - pass_V2 on the training rollouts; change-point test against the
  step-0 level (one-sided CUSUM on the per-update gap, documented false-alarm behaviour, tested on
  simulated clean and leaking streams); also tracks mean completion length and flags collapse
  (length below a fraction of its baseline). `state_dict` / `load_state_dict`.
- `AlarmState(alarm: bool, gap: float, baseline_gap: float, cusum: float, length_ratio: float, reasons: tuple[str, ...])`.
- `screen_mask(v1_pass, v2_pass) -> list[bool]`: True = keep; rollouts where V1 passes and V2
  fails are screened out of the advantage (they are not penalised).
- `static_leak_audit(completions, references, v1, v2) -> LeakAudit` (share passing V1 but failing V2).
- `strict_math_verifier(**overrides) -> MathVerifier`: the strict V2 configuration for math
  (`extractors=("boxed",)`, `strict_boxed=True`, `strict_trailing=True`, `strip_percent=False`).

### 1.5 `weights.py` — exact snapshots, swaps, interpolation, group rollback

- `StateSnapshot` (dataclass): `params: dict[str, Tensor]` (CPU clones, original dtype, by
  `named_parameters` name), `optimizer_state: dict | None` (deep copy of `optimizer.state_dict()`,
  tensors cloned), `masters: dict[str, Tensor] | None` (fp32 masters of a `HookedOptimizer`),
  `channel_state: ChannelState | None`, `fingerprint: str`.
- `capture(model, names, *, optimizer=None, channel_layers=None) -> StateSnapshot`.
- `restore(model, snapshot, *, optimizer=None, restore_optimizer=True)`: params, optimizer
  state and HookedOptimizer masters **bit-identical** to the snapshot (tested with AdamW and
  with a HookedOptimizer that keeps fp32 masters of bf16 params).
- `swapped(model, snapshot)` (context manager): temporarily loads the snapshot's params (for
  evaluating the incumbent) and puts the current values back bit-identically on exit (also on
  exceptions); never touches optimizer state.
- `interpolate(model, incumbent, candidate, lam, *, optimizer=None)`: writes
  `incumbent + lam * (candidate - incumbent)` computed in float32 (float64 for float64 params)
  and cast to the parameter dtype; updates HookedOptimizer masters consistently.
- `channel_groups_by_damage(incumbent, candidate_model, activation_scores, groups) -> list[ChannelGroup]`:
  orders channel groups by (displacement x activation on the failing slice), largest first.
- `revert_groups(model, incumbent, groups, *, optimizer=None)`: reverts the channels of the
  groups to the incumbent with `research_rl.models.ffn_channels.rollback_channels`, then fixes
  masters; returns the `RollbackReport`s.
- `ChannelGroup(name: str, selection: ChannelSelection)` is defined in `types.py`.

### 1.6 `scores.py` — forward-only channel scoring and grouping

- `collect_channel_stats(model, tokenizer, texts, *, layers=None, batch_size=8, threshold=0.0) -> ChannelStats`:
  forward pre-hooks on every SwiGLU `down_proj` (its input is the intermediate activation
  `act(gate(x)) * up(x)`), masked by the attention mask; per layer and channel: count, mean |h|,
  mean h^2, activation rate (|h| > threshold). Hooks are always removed (also on exceptions);
  the model's training mode is restored; no gradients.
- Scorers returning `dict[int, Tensor]` usable with `ChannelSelection.top_k_from_scores`:
  `wanda_scores(model, stats)` (||W_down[:, c]||_2 * sqrt(mean h_c^2)), `contrast_scores(stats_correct, stats_incorrect)`
  (difference of mean |h| between V2-correct and V2-incorrect rollouts of the same prompts),
  `usage_ratio_scores(stats_train, stats_retention, eps)` (activation-rate ratio, high = used on
  the task and rarely on retention), `random_scores(model, seed)` (density-matched random control),
  `layer_prior(scores, weights: dict[int, float])`.
- `group_channels(scores, *, bands=4, layers=None) -> list[ChannelGroup]`: groups by layer x
  score quantile band; every channel in exactly one group; deterministic names `"L{layer}/b{band}"`.
- `density_matched_random(selection, model, seed) -> ChannelSelection`: same per-layer counts.

### 1.7 `spaces.py` and `es.py` — the forward-only operator

- Spaces implementing `types.ParameterSpace`:
  `ChannelGainSpace(model, selection, *, up=False)` (P1: one multiplicative gain per selected
  channel on its `down_proj` column, optionally also on its `up_proj` row; `write(v)` sets
  `W = base * (1 + v)` from the stored base), `ChannelWeightSpace(model, selection)` (P2: every
  weight owned by the selected channels, `W = base + v`), `FullParameterSpace(model, names)`.
  All writes under `torch.no_grad()` with `copy_`; elements outside the space never change
  (tested with `verify_modification_scope`).
- `ESConfig`: `population` (even with antithetic pairs), `sigma`, `lr`, `antithetic=True`,
  `fitness_shaping="zscore" | "rank"`, `consolidate=False` (CoRP-style agreement re-weighting),
  `max_step_norm=None`, `seed`.
- `ESOperator(space, fitness: Callable[[nn.Module, int, int], FitnessResult], config)` implements
  `types.UpdateOperator` (`optimizer = None`). One `step` = one ES update: perturbation `i` of
  update `t` is generated from a `torch.Generator` seeded by `(seed, t, i)` (common random
  numbers: every member is evaluated on the same items, fixed by the caller-provided
  `fitness(model, update_index, member_index)` which must draw its items from `update_index`
  only); the update is the shaped-fitness-weighted sum; the space's base is restored after each
  member; the accepted direction is written and the space re-based. `on_decision(ROLLBACK)` and
  `on_decision(SHRINK/GROUP_ROLLBACK)` re-base the space from the (restored) model, never by
  subtracting regenerated noise. `FitnessResult(reward_v1, reward_v2, v1_pass, v2_pass, lengths)`.
  The report counts forward passes (generation calls) and `backward_passes = 0`.

### 1.8 `probe.py` and `tracker.py` — evidence and capability curves

- `CapabilityProbe(sampler, score_fn, *, transfer_slices, retention_slices, items_per_slice)`:
  `evidence(model, incumbent: StateSnapshot, decision_index, *, attempt: int = 0, alarm=False) -> GateEvidence`
  draws fresh items for `(decision_index, attempt)`, scores the candidate (current weights) and
  the incumbent (via `weights.swapped`) on the same items.
- `StartMap`: pass@1 (greedy) and pass@k (sampled, k configurable, default 16 on CPU) of the
  start checkpoint per slice.
- `CapabilityTracker(sampler, score_fn, start)` → `measure(model, step, reward_v1) -> CapabilityPoint`
  with A (acquisition on fresh TRAIN items), R per retention slice, T_task, T_rule, and
  D(t) = [reward(t) - reward(0)] - [T_rule(t) - T_rule(0)]; `to_dict`.

### 1.9 `loop.py` and `operators.py` — the gated loop (framework-agnostic)

- `GateController(gate, probe, *, rollback_unit, channel_groups=None)`:
  `run(model, operator, incumbent, decision_index, update_index, alarm, reward_v1) -> tuple[DecisionRecord, StateSnapshot]`
  implementing: ACCEPT if transfer bound > 0 and every retention slice passes and no alarm; HOLD
  if retention passes but transfer is not confirmed (counts toward `hold_limit`); otherwise try
  SHRINK for each lambda (fresh items each time), then (only for `RollbackUnit.CHANNELS`) group
  rollback in damage order up to `max_group_rollbacks`, then ROLLBACK (params + optimizer state
  restored to the incumbent). A confirmed alarm (alarm and retention or transfer fails) goes to
  ROLLBACK directly. Returns the record and the new incumbent.
- `GatedLoop(model, operator, controller, alarm, *, config, recorder=None, log_path=None)`:
  `run(num_updates)`; a decision every `block_updates` updates and immediately on an alarm
  (if `alarm_triggers_decision`); writes one JSON line per `DecisionRecord`; with
  `gate.enabled=False` it never rolls back but still measures and logs (ungated arm); the
  operator receives only the decision. `state_dict()` / `load_state_dict()` cover incumbent,
  decision index, ledger, sampler position, alarm state, hold counter, operator state.
- `operators.TorchGRPOOperator(model, tokenizer, sampler, v1, v2, *, group_size, prompts_per_update, lr, max_new_tokens, temperature, optimizer=None, screen=True, seed)`:
  minimal on-policy GRPO (sampled completions, group-normalised V1 rewards, screened rollouts
  get advantage 0, one optimizer step per update) used by the pilot's backprop control arm.
  Works with a plain `torch.optim.AdamW` or a `HookedOptimizer` (e.g. with a `MaskRule` for a
  channel subset).

### 1.10 `trl_gate.py` — the gate inside the existing TRL GRPO path

- `CapabilityGateCallback(probe, gate, *, block_updates, rollback_unit, channel_groups=None, log_path=None)`:
  a `transformers.TrainerCallback` that is also a `StatefulComponent` (register it in the
  trainer's `research_components` so it travels with checkpoints). `on_train_begin` captures the
  incumbent (trainable params + optimizer state + HookedOptimizer masters) and **refuses**
  (ValueError) a `block_updates` whose micro-batch count is not a multiple of
  `steps_per_generation * num_iterations` (so a decision never lands inside TRL's buffered
  generation batch); `on_step_end` at block boundaries runs the `GateController` on the
  trainer's model and optimizer. Must verify against TRL 1.18 source that after a rollback the
  next step regenerates from the restored weights (no stale buffered rollouts or old log-probs);
  if TRL keeps a buffer across the boundary, invalidate it explicitly and test it.
- `DisagreementScreen(v1, v2, alarm, tokenizer)`: an `research_rl.trainers.rl_common.AdvantageShaper`
  that decodes the batch's completions, applies V2, zeroes the advantage of V1-pass/V2-fail rows
  (`signal` False for them) and feeds the alarm; stateful (alarm state) for resume.
- Integration test: `ResearchGRPOTrainer` (via the existing GRPO adapter) on the tiny model with
  the callback; a forced-reject probe must restore params and optimizer state bit-identically;
  resume from a checkpoint restores the gate state.

### 1.11 `search.py` — subset search with capability as the objective

- `SubsetSearch(groups, make_operator, probe, *, rounds, burst_updates, survivors_fraction=0.5, repeats_final=2, controls=("random", "whole_layer", "unrestricted"))`:
  successive halving; every candidate subset starts from the same snapshot, gets a short burst
  with its own operator, is scored on fresh items by the transfer gain point estimate and the
  worst retention drop (objective = gain - penalty * max(0, worst_drop - margin)); finalists are
  re-evaluated `repeats_final` times on new items and ranked by the mean (denoised); controls
  are always evaluated with the same budget. The model is restored to the start snapshot after
  the search. Returns `SearchReport` (JSON-able, includes every evaluation and the controls).

### 1.12 `edits.py` — declarative edits and the version graph (shared with the UI)

- `EditOp` in {`ablate`, `revert`, `scale`, `interpolate`, `apply_delta`}; `EditRecord(op, selection: dict[int, list[int]], params: dict, note, created)`.
  `scale` multiplies the channels' `down_proj` columns (and optionally `up_proj` rows) by
  `params["factor"]`; `revert` and `interpolate` use a named reference version; `apply_delta`
  uses a stored delta id. No free-form code, ever.
- `VersionStore(root)`: versions are immutable records (id, parent id, edits, gate verdict or
  `override` reason, model fingerprint, created) persisted as JSON; the channel state of each
  version is stored with safetensors (never pickle). `stage(edit)`, `evaluate(staged, gate, probe)`,
  `commit(staged, *, override: str | None)` (refused if the gate did not pass and no override
  reason is given), `restore(version_id)`, `history()`; scope verification
  (`verify_modification_scope`) is run and stored for every commit.

### 1.13 `research_rl/ui/` — local search + edit tool

Local HTTP server (Python stdlib `http.server`, no new dependency) next to the GPU plus a
static browser front end shipped as package data. Binds to 127.0.0.1 by default; every API
request needs the session token (constant-time compare), Host and Origin are checked, request
bodies are capped, static files are served only from the package's static directory
(path containment). One model worker thread executes queued jobs; the HTTP threads never touch
weights. Edits are declarative `EditRecord`s; commits only after the gate passes or with a
recorded override. The AUDIT split is not reachable through the API. Weights load from
safetensors only. Features: channel search (rank by stored scores, filter by layer/group),
inspect (vocabulary projection of the `down_proj` column, activation histograms per split,
change across versions), stage edit, evaluate (gate verdict with reward beside capability),
commit / discard, version history and restore, training view (reads a run's decision log).

### 1.14 Pilot

- `examples/capability_pilot_tiny.py`: P0 on CPU. Builds a tiny Llama from scratch, SFT on
  atomic skills (low start: depth-2 pass@1 low, pass@k small but measured), then arms
  {GRPO, ES full, ES informed subset, ES random subset} x {gate off, gate on} x seeds, on a
  planted-leak and a clean variant. Writes `results.json` and `summary.md` with A, T_rule, R,
  D(t), hack rate, forward/backward pass counts and the gate's false-accept / false-reject
  rates against the oracle. `--smoke` runs in under a minute for CI.
- `scripts/gpu/capability_p1.py`: the same pipeline for a local 7-8B checkpoint path on one GPU.
  NOT_TESTED. Never downloads anything (`HF_HUB_OFFLINE=1` respected; a path is required).
- Kill criteria are fixed in section 3 **before** any pilot run.

### 1.15 As-built amendments (wave 1, after adversarial review)

Where this section and §1.1–1.7 differ, this section and the code win.

- **Split disjointness is functional, not only nominal** (`tasks.py`). Two rules are *similar*
  when they are equal on every probe input of some input length, or agree on at least
  `max_agreement` (default `DEFAULT_MAX_AGREEMENT = 0.25`) of probe inputs. Similar pairs share a
  split class; a class containing a rule similar to the identity or an atomic operator goes to
  TRAIN; depth-3 rules similar to any pair, atomic, identity or earlier depth-3 rule are
  rejected. `check_disjoint` enforces this. Consequence: with the default `depth3_count=16`,
  `num_operators` must be >= 9 (10-12 never failed over seeds 0-199).
- **Truncation reaches the verifier**: `VerifierScoreFn` marks a completion truncated when it
  uses the whole `max_new_tokens` budget, and both verifiers score it 0. Contract for callers:
  `max_new_tokens >= longest answer + 1` (compositions: `max_len + 1`).
- **Non-binary scores** (`stats.py`): no bootstrap and no t-bound (the t-bound was shown to be
  anti-conservative). The same tests run on fractional discordant mass
  (`gain = sum max(c-i,0)`, `loss = sum max(i-c,0)`): methods `sign_test_fractional` and
  `tango_score_fractional`, calibrated by simulation in the tests.
- **Retention sample size** (`gate.py`): default `items_per_retention_slice = 2048`; at 512 a
  clean slice stops being passable after a handful of spending-schedule verdicts.
  `GateConfig.retention_horizon(n=, n10=, n01=)` returns the last verdict index at which a slice
  can still pass; the pilot must choose `n` and the decision count so the horizon covers the run
  and record that choice. `verdict` rejects wrong splits and uid overlap before spending alpha.
- **Alarm with correlated rollouts** (`alarm.py`): `min_rollouts` is compared with the Kish
  effective size. ES callers construct the alarm with `rollouts_per_unit=population` or put
  per-rollout unit ids in `report.extra["alarm_units"]`.
- **ES default step** (`es.py`): `ESConfig.lr = 0.001` (step / perturbation ratio ~1).
- **Interpolation endpoints** (`weights.py`): `lam` 0 and 1 are exact copies; `restore` checks
  optimizer-state shapes; `revert_groups` validates the incumbent before writing.

### 1.16 Decision semantics pinned for wave 2 (loop, trl_gate, edits)

- **Alarm with both tests passing** → HOLD (not promoted), reason `"alarm_hold"`; counts toward
  `hold_limit`. Alarm with any failing test → ROLLBACK directly (reason `"confirmed_alarm"`).
- **SHRINK** (tried only when retention failed): for each lambda, `interpolate` to
  `incumbent + lam (candidate - incumbent)` and re-test on fresh items. It *passes* when
  retention passes. The shrunk weights stay as the current weights; the incumbent is promoted to
  them only if transfer also passes and there is no alarm (record `promoted: true/false` in the
  reason). Optimizer state is kept from the candidate (Adam moments on interpolated weights;
  documented, not restored). **GROUP_ROLLBACK**: the same pass/promotion rule after cumulatively
  reverting groups in damage order. If nothing passes → ROLLBACK (params, optimizer state,
  masters restored to the incumbent).
- **Gate disabled** (`enabled=False`, the ungated arm): the full procedure's *first* verdict is
  still computed on the same fresh items; the decision applied is ACCEPT-with-no-rollback, and
  the record stores `override="gate_disabled"` and `reason="would=<decision the gate would have
  taken on the first verdict>"`, so false-accept / false-reject rates against the oracle can be
  computed. The disabled arm uses a separate ledger so it never changes the gated arm's alpha.
- **Alpha / horizon budgeting is in verdicts, not decisions.** One decision uses up to
  `1 + len(shrink_lambdas) + max_group_rollbacks` verdicts. `GatedLoop` checks at start that
  `GateConfig.retention_horizon(n=items_per_retention_slice, n10=round(0.005 n), n01=0)` covers
  the worst-case verdict count of the planned run, and raises (or warns with
  `allow_short_horizon=True`) otherwise. A verdict where retention is unpassable at the current
  alpha even with zero observed drop gets reason `"horizon_exhausted"`, never a plain rollback
  reason.
- **Training-time scoring uses the same truncation rule as the gate**: operators and the TRL
  screen score sampled completions through `VerifierScoreFn.truncated` + `score_completions`
  (never raw `verify()`), so V1/V2 divergence is not manufactured by different truncation.
- **Alarm units**: GRPO operators put the prompt index into `report.extra["alarm_units"]`; ES
  callers build the alarm with `rollouts_per_unit=population` (or set units); the TRL screen uses
  the prompt group (`num_generations`). An end-to-end test must show a clean stream through
  `GatedLoop` does not alarm.
- **edits.py** evaluates with its own ledger and never spends a training run's alpha.

## 2. Engineering rules for this package

- Reuse existing pieces: `ChannelSelection`, `extract_channels`, `rollback_channels`,
  `verify_modification_scope` (`models/ffn_channels.py`), `find_swiglu_ffns`, `snapshot`,
  `fingerprint` (`models/model_interface.py`), `HookedOptimizer`, `MaskRule`,
  `ParameterMask`, `greedy_generate`, `RetentionSuite`, `pass_at_k`, `BaseVerifier`,
  `MathVerifier`, `ExperimentRecorder`, `AdvantageShaper`.
- Bit-exactness is asserted only for copies, swaps and restores, never for trained floats.
- Deterministic: every random draw comes from an explicit seed.
- `ruff format`, `ruff check` (0.17.5) and `mypy` (2.4.1) clean; Python >= 3.10 syntax.

## 3. Pilot kill criteria (fixed before any run)

See section 4.5 of the feasibility report. Restated as the pilot's pass/fail rules:

| Question | Success | Kill |
|---|---|---|
| Gate | On planted-leak runs: lower final hack rate and higher T_rule with the gate than without, paired-over-seeds interval excluding zero. On clean runs: T_rule within one standard error of gate-off. | Clean-run T_rule cost exceeds the planted-run T_rule gain, or the gate rejects more than half of the blocks the oracle marks as improved. |
| Subset | Informed subset beats the density-matched random subset on R or T_rule (paired interval excluding zero) at equal A. | No separation across seeds: drop the search, keep channel groups only as the rollback unit. |
| Operator | ES reaches at least half of GRPO's T_rule gain within ten times its forward-pass-equivalent budget (one backward counted as two forwards). | Below that: keep GRPO behind the gate; do not spend the GPU run on ES. |
| Low start | At least one arm exceeds the start checkpoint's pass@k on unseen pairs. | None does: the task is acquisition-limited; report it instead of weakening the task. |

The oracle for the gate's error rates is the exact verifier on a large fresh AUDIT-free
evaluation (T_rule) measured before and after each block.
