# Capability-gated training (`research_rl.capability`)

Status: **implemented (waves 1-3), CPU-verified only**. Everything in this package is verified on
CPU with tiny models only (section 4 has the as-built module map and a tested quickstart).
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
- `OperatorSet(seed: int = 0, num_operators: int = 10, *, min_len=3, max_len=7)`: deterministic name assignment
  (invented names must not be common English words and must not collide). `apply(rule, x) -> str`.
- `CompositionSplits.build(operators, *, seed, train_fraction=0.5, audit_fraction=0.2, depth3_count=16,
  audit_depth3_count=None, task_families=("argmax", "countbig"), max_agreement=0.25)`:
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
  operators), `retention/copy` (echo the input), `retention/prereq` (as built: the atomic
  prerequisites of the held-out families), `audit/pairs`, `audit/depth3`. As built the held-out
  families are `t_task/argmax` and `t_task/countbig`.
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
  `research_rl.evaluation.capability_retention.greedy_generate` (as built it returns a
  `VerifierScoreFn`, which also exposes `generate`, `truncated` and `score_completions`).

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
- `AlphaLedger(total_alpha=0.05, schedule="spending")` with schedules `"per_decision"` (alpha every time; the
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
  `max_group_rollbacks=4`, `rollback_unit=RollbackUnit.PARAMETERS`, `block_updates=8`,
  `items_per_transfer_slice=256`, `items_per_retention_slice=2048`, `alarm_triggers_decision=True`;
  as built also `retention_horizon(slice_name=None, *, n=None, n10=0, n01=0)` and `make_ledger()`.
- `CapabilityGate(config, ledger: AlphaLedger | None = None)` (default `config.make_ledger()`; an
  explicit ledger is authoritative): `verdict(evidence: GateEvidence) -> GateVerdict`
  consumes one alpha from the ledger per call (the shrink / group re-tests of one procedure are
  separate tests and each consumes alpha — documented), pools transfer slices, tests every
  retention slice; `state_dict()` / `load_state_dict()`.

### 1.4 `alarm.py` — tier-1 verifier-disagreement alarm

- `DisagreementAlarm(window=5, threshold=0.3, min_rollouts=32, *, drift=0.1, baseline_updates=3,
  baseline_gap=None, baseline_length=None, length_fraction=0.5, length_collapse_alarm=True,
  reset_on_alarm=True, rollouts_per_unit=1)`: `update(report: OperatorReport) -> AlarmState`;
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
  `layer_prior(scores, weights: dict[int, float])`. As built: `usage_ratio_scores(..., eps=1e-3)` and
  `random_scores(model, seed, *, layers=None)`.
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

- `CapabilityProbe(sampler, score_fn, *, transfer_slices, retention_slices, items_per_transfer_slice,
  items_per_retention_slice)` (as built: two sizes; at least one retention slice is required):
  `evidence(model, incumbent: StateSnapshot, decision_index, *, attempt: int = 0, alarm=False) -> GateEvidence`
  draws fresh items for `(decision_index, attempt)`, scores the candidate (current weights) and
  the incumbent (via `weights.swapped`) on the same items.
- `StartMap`: pass@1 (greedy) and pass@k (sampled, k configurable, default 16 on CPU) of the
  start checkpoint per slice. As built it lives in `probe.py` (`StartMap.measure(...)`) and
  `tracker.py` re-exports it.
- `CapabilityTracker(sampler, score_fn, start, *, items_per_slice)` → `measure(model, step, reward_v1) -> CapabilityPoint`
  with A (acquisition on fresh TRAIN items), R per retention slice, T_task, T_rule, and
  D(t) = [reward(t) - reward(0)] - [T_rule(t) - T_rule(0)]; `to_dict`.

### 1.9 `loop.py` and `operators.py` — the gated loop (framework-agnostic)

- `GateController(gate, probe, *, rollback_unit, channel_groups=None, damage_texts_fn=None, tokenizer=None,
  damage_batch_size=8)`:
  `run(model, operator, incumbent, decision_index, update_index, alarm, reward_v1, reward_v2=None) -> tuple[DecisionRecord, StateSnapshot]`
  implementing: ACCEPT if transfer bound > 0 and every retention slice passes and no alarm; HOLD
  if retention passes but transfer is not confirmed (counts toward `hold_limit`); otherwise try
  SHRINK for each lambda (fresh items each time), then (only for `RollbackUnit.CHANNELS`) group
  rollback in damage order up to `max_group_rollbacks`, then ROLLBACK (params + optimizer state
  restored to the incumbent). A confirmed alarm (alarm and retention or transfer fails) goes to
  ROLLBACK directly. Returns the record and the new incumbent.
- `GatedLoop(model, operator, controller, alarm, *, config, recorder=None, log_path=None, allow_short_horizon=False)`:
  `run(num_updates)`; a decision every `block_updates` updates and immediately on an alarm
  (if `alarm_triggers_decision`); writes one JSON line per `DecisionRecord`; with
  `gate.enabled=False` it never rolls back but still measures and logs (ungated arm); the
  operator receives only the decision. `state_dict()` / `load_state_dict()` cover incumbent,
  decision index, ledger, sampler position, alarm state, hold counter, operator state.
- `operators.TorchGRPOOperator(model, tokenizer, sampler, v1, v2, *, group_size, prompts_per_update, lr, max_new_tokens, temperature, optimizer=None, screen=True, seed, name="grpo")`:
  minimal on-policy GRPO (sampled completions, group-normalised V1 rewards, screened rollouts
  get advantage 0, one optimizer step per update) used by the pilot's backprop control arm.
  Works with a plain `torch.optim.AdamW` or a `HookedOptimizer` (e.g. with a `MaskRule` for a
  channel subset).

### 1.10 `trl_gate.py` — the gate inside the existing TRL GRPO path

- `CapabilityGateCallback(probe, gate, *, block_updates, rollback_unit, channel_groups=None, log_path=None, alarm=None)`:
  a `transformers.TrainerCallback` that is also a `StatefulComponent` (register it in the
  trainer's `research_components` so it travels with checkpoints). `on_train_begin` captures the
  incumbent (trainable params + optimizer state + HookedOptimizer masters) and **refuses**
  (ValueError) a `block_updates` whose micro-batch count (as built: `block_updates *
  gradient_accumulation_steps`) is not a multiple of `steps_per_generation * num_iterations` (so a
  block decision never lands inside TRL's buffered generation batch; an off-boundary alarm
  decision that changes the weights sets `trainer._buffered_inputs = None`); `on_step_end` at block boundaries runs the `GateController` on the
  trainer's model and optimizer. Must verify against TRL 1.18 source that after a rollback the
  next step regenerates from the restored weights (no stale buffered rollouts or old log-probs);
  if TRL keeps a buffer across the boundary, invalidate it explicitly and test it.
- `DisagreementScreen(v1, v2, alarm, tokenizer, *, max_new_tokens, answers=None, answer_column="answer")`: an `research_rl.trainers.rl_common.AdvantageShaper`
  that decodes the batch's completions, applies V2, zeroes the advantage of V1-pass/V2-fail rows
  (`signal` False for them) and feeds the alarm; stateful (alarm state) for resume.
- Integration test: `ResearchGRPOTrainer` (via the existing GRPO adapter) on the tiny model with
  the callback; a forced-reject probe must restore params and optimizer state bit-identically;
  resume from a checkpoint restores the gate state.

### 1.11 `search.py` — subset search with capability as the objective

- `SubsetSearch(groups, make_operator, probe, *, rounds, burst_updates, survivors_fraction=0.5, repeats_final=2, controls=("random", "whole_layer", "unrestricted"), penalty=1.0, margin=0.02, seed=0, items_per_transfer_slice=None, items_per_retention_slice=None, index=0, pool="track")`,
  `run(model, *, update_offset=0) -> SearchReport`:
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
- `VersionStore(root, model, *, channel_layers=None)` (as built the store is bound to one live model;
  `init_from_model(note="")` records the root version): versions are immutable records (id, parent id, edits, gate verdict or
  `override` reason, model fingerprint, created) persisted as JSON; the channel state of each
  version is stored with safetensors (never pickle). `stage(edit)`, `discard(staged)`,
  `evaluate(staged, gate_config, probe)` (the store's own persisted ledger, and gate items from
  the store's own stream: purpose `edit/<gate_nonce>/gate/0`, nonce drawn when the store is created),
  `commit(staged, *, override: str | None = None, note="")` (refused if the gate did not pass and no override
  reason is given), `restore(version_id)`, `history()`, plus `add_delta`, `delta_between`, `diff`; scope verification
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

- `examples/capability_pilot_tiny.py`: P0 on CPU (as built a thin CLI over
  `research_rl.capability.pilot.run_pilot`; the full default config is 48 arms = 4 operators x 2 gates
  x 2 variants x 3 seeds). Builds a tiny Llama from scratch, SFT on
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
  computed. The disabled arm uses a separate ledger so it never changes the gated arm's alpha
  (it spends one alpha per decision, the gated arm one per verdict, so after the gated arm's first
  re-test the two arms' first verdicts share items but not alpha).
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
- **edits.py** evaluates with its own ledger and never spends a training run's alpha, on gate
  items from a per-store stream (purpose `edit/<gate_nonce>/gate/0`, see 4.3).

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

## 4. As built (wave 3): module map, quickstart, limits

This section describes the code as it exists. Where sections 1.1-1.14 and the code differ, the
code, its docstrings and sections 1.15/1.16 win. Section 3 (the kill criteria) is unchanged.

### 4.1 Module map

| Module | What it does | Main public names | Tests |
|---|---|---|---|
| `types.py` | shared data types and protocols (no behaviour) | `Split`, `Item`, `ScoreFn`, `PairedScores`, `GateEvidence`, `GateVerdict`, `GateDecision`, `DecisionRecord`, `OperatorReport`, `UpdateOperator`, `ParameterSpace`, `RollbackUnit`, `ChannelGroup` | through every other test |
| `tasks.py` | operator-composition generators, functional split disjointness, sampler, verifiers, score function | `OperatorSet`, `CompositionSplits`, `ItemSampler`, `sft_examples`, `ExactAnswerVerifier`, `SuffixLeakVerifier`, `VerifierScoreFn`, `make_score_fn`, `AuditAccessError` | `test_capability_tasks.py` |
| `stats.py` | paired tests (exact sign test, Tango score test, fractional variants), sample sizes, alpha ledger | `paired_gain_lower_bound`, `noninferiority_upper_bound`, `required_paired_n`, `retention_power_floor_n`, `AlphaLedger` | `test_capability_stats.py` (calibration by simulation) |
| `gate.py` | the pure verdict on one `GateEvidence` | `GateConfig`, `CapabilityGate` | `test_capability_gate.py` |
| `alarm.py` | tier-1 V1/V2 disagreement alarm (CUSUM + length collapse), screening, static leak audit | `DisagreementAlarm`, `AlarmState`, `screen_mask`, `static_leak_audit`, `strict_math_verifier` | `test_capability_alarm.py` |
| `weights.py` | exact snapshots, swaps, interpolation, channel-group damage order and revert | `StateSnapshot`, `capture`, `restore`, `swapped`, `interpolate`, `channel_groups_by_damage`, `revert_groups` | `test_capability_weights.py` |
| `scores.py` | forward-only channel statistics, scorers and grouping | `collect_channel_stats`, `wanda_scores`, `contrast_scores`, `usage_ratio_scores`, `random_scores`, `group_channels`, `density_matched_random` | `test_capability_scores.py` |
| `spaces.py` | parameter spaces for forward-only operators | `ChannelGainSpace`, `ChannelWeightSpace`, `FullParameterSpace` | `test_capability_spaces.py` |
| `es.py` | evolution-strategies update operator (no backward pass) | `ESConfig`, `ESOperator`, `FitnessResult`, `shape_fitness` | `test_capability_es.py` |
| `probe.py` | gate evidence on fresh items, start map | `CapabilityProbe`, `StartMap`, `SliceStart`, `score_items` | `test_capability_probe.py` |
| `tracker.py` | capability curves A, R, T_task, T_rule and D(t) | `CapabilityTracker`, `CapabilityPoint` | `test_capability_tracker.py` |
| `loop.py` | the gate procedure and the gated loop (decision log, resume, horizon check) | `GateController`, `GatedLoop`, `classify_verdict`, `worst_case_verdicts`, `REASON_*` | `test_capability_loop.py` |
| `operators.py` | minimal on-policy GRPO (the backprop control arm) | `TorchGRPOOperator`, `group_advantages` | `test_capability_operators.py` |
| `trl_gate.py` | the gate inside TRL's GRPO path (lazy export, imports `transformers`) | `CapabilityGateCallback`, `DisagreementScreen` | `test_capability_trl_gate.py` |
| `search.py` | successive-halving subset search with capability as the objective | `SubsetSearch`, `SearchReport`, `STATEMENT` | `test_capability_search.py` |
| `edits.py` | declarative channel edits and the immutable, gate-checked version graph | `EditOp`, `EditRecord`, `VersionStore`, `Version`, `StagedEdit` | `test_capability_edits.py` |
| `pilot.py` | the P0 pilot pipeline and the section-3 criteria evaluation | `PilotConfig`, `smoke_config`, `run_pilot`, `evaluate_criteria` | `test_capability_pilot.py` |
| `calibration.py` | the section-5 exploratory positive control of the gate (candidates of known quality, one first verdict each, oracle, the section-5 table); see 4.4 | `CalibrationConfig`, `calibration_smoke_config`, `run_calibration`, `evaluate_rules`, `build_seed_plan` | `test_capability_calibration.py` |
| `calibration_alarm.py` | the section-5.1 follow-up: `hack_replay` / `good_hack` candidates, tier-2 first verdict, tier-1 alarm stream (3 incumbent + 16 candidate reports) and null stream, combined decision, oracle, the 5.1 table; see 4.5 | `CalibrationAlarmConfig`, `calibration51_smoke_config`, `run_calibration_51`, `evaluate_rules_51`, `build_seed_plan_51`, `alarm_stream`, `combined_decision` | `test_capability_calibration51.py` |
| `research_rl/ui/` | local channel search + edit tool (stdlib HTTP server, static front end) | `python -m research_rl.ui` | `test_ui.py` |
| `research_rl/models/safe_checkpoint.py` | safetensors-only allowlist check of a local model directory, shared by UI `--model-dir` and the P1 script | `check_safetensors_dir`, `UnsafeCheckpointError` | `test_ui.py`, `test_capability_p1_script.py` |
| `examples/capability_pilot_tiny.py` | CLI of the P0 pilot | `--smoke`, `--output-dir`, `--arms`, `--jobs` | `test_capability_pilot.py` |
| `examples/capability_calibration_tiny.py` | CLI of the section-5 positive control | `--smoke`, `--output-dir`, `--seeds`, `--start`, `--jobs` | `test_capability_calibration.py` |
| `examples/capability_calibration51_tiny.py` | CLI of the section-5.1 alarm follow-up | `--smoke`, `--output-dir`, `--seeds`, `--start`, `--jobs` | `test_capability_calibration51.py` |
| `scripts/gpu/capability_p1.py` | P1 on a local 7-8B checkpoint, one GPU (**NOT_TESTED**; only `--dry-run` on CPU is tested) | | `test_capability_p1_script.py` |

Imports: `research_rl/capability/__init__.py` re-exports every module's `__all__` (and the shared
types), so `from research_rl.capability import GatedLoop, ESOperator` works. The `trl_gate` names
(`CapabilityGateCallback`, `DisagreementScreen`, `GATE_COMPONENT`, `base_optimizer`,
`generate_every`) are resolved lazily on first attribute access: `import research_rl.capability`
does not import `trl_gate`, `research_rl.trainers` or TRL. The intra-package imports form a DAG
(checked by `tests/research_rl/test_capability_exports.py`, which also imports every module as the
first module in a fresh interpreter). The full API reference is in [API.md](API.md), with a
section for every `research_rl.capability` and `research_rl.ui` module; the same test checks that
each section lists the module's public names with their current signatures.

### 4.2 Quickstart

Every block below is executed verbatim by `tests/research_rl/test_capability_exports.py` (the
blocks marked `quickstart:` are extracted from this file; the shell blocks are run with
`--port 0` / `--store <tmp>` and a temporary `--output-dir` substituted). They run on CPU with
tiny, randomly initialised models built in-process: they show the plumbing, and **every number
they print is a smoke value, not a result**.

**Splits, sampler and score function.**

<!-- quickstart:splits -->
```python
from research_rl.capability import (
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    Split,
    make_score_fn,
)
from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama

tokenizer = build_char_tokenizer()
splits = CompositionSplits.build(OperatorSet(seed=0, num_operators=10), seed=0)
splits.check_disjoint()  # raises SplitIntegrityError on any rule-level or functional overlap
print({name: split.value for name, split in splits.slices().items()})

sampler = ItemSampler(splits, root_seed=1)  # drawing from Split.AUDIT raises AuditAccessError
items = sampler.draw(Split.T_RULE, 4, purpose="demo", index=0, slice="t_rule/unseen_pairs")
print(items[0].prompt, "->", items[0].answer, items[0].rule)

# strict verifier V2; max_new_tokens >= longest answer + 1 (a truncated completion scores 0)
score_fn = make_score_fn(tokenizer, ExactAnswerVerifier(), max_new_tokens=8, batch_size=32)
model = build_tiny_llama(len(tokenizer), seed=0)  # random weights: expect all zeros
print(score_fn(model, items))
```

**One `GatedLoop` run with `ESOperator`** (continues the block above). The fitness is the mean
log-probability of the answer, because a greedy pass rate is flat on random weights and ES would
take a zero step; V1 (a planted suffix leak) and V2 (exact) grade the greedy rollouts for the
alarm. The gate uses `per_decision` alpha and small slices to stay fast; see 4.3 before choosing
real settings. The alarm keeps its calibrated defaults: 4 members x 32 shared items give
`n_eff = 32`, the smallest size inside its false-alarm scope (4.3, "Alarm scope").

<!-- quickstart:loop -->
```python
import tempfile
from pathlib import Path

import torch

from research_rl.capability import (
    CapabilityGate,
    CapabilityProbe,
    ChannelGainSpace,
    DisagreementAlarm,
    ESConfig,
    ESOperator,
    FitnessResult,
    GateConfig,
    GateController,
    GatedLoop,
    RollbackUnit,
    SuffixLeakVerifier,
    VerifierScoreFn,
)
from research_rl.models.ffn_channels import ChannelSelection

config = GateConfig(
    alpha_schedule="per_decision",  # the guarantee then holds per verdict only (see 4.3)
    retention_margin=0.2,
    hold_limit=2,
    block_updates=2,
    items_per_transfer_slice=16,
    items_per_retention_slice=64,
)
v1 = VerifierScoreFn(tokenizer, SuffixLeakVerifier(k=2), max_new_tokens=8, batch_size=32)
v2 = VerifierScoreFn(tokenizer, ExactAnswerVerifier(), max_new_tokens=8, batch_size=32)


def answer_logprob(m, batch):
    """Mean log-probability of the answer characters after the prompt (teacher forcing)."""
    total = 0.0
    for it in batch:
        p = tokenizer(it.prompt, add_special_tokens=False)["input_ids"]
        a = tokenizer(it.answer, add_special_tokens=False)["input_ids"]
        logp = torch.log_softmax(m(input_ids=torch.tensor([p + a])).logits[0].float(), -1)
        total += float(logp[torch.arange(len(p) - 1, len(p) + len(a) - 1), torch.tensor(a)].mean())
    return total / len(batch)


def fitness(m, update_index, member_index):
    # items depend on update_index only: every member is scored on the same 32 items
    batch = sampler.draw(Split.TRAIN, 32, purpose="es", index=update_index)
    texts = v1.generate(m, batch)
    truncated = v1.truncated(texts)
    s1 = v1.score_completions(texts, batch, truncated)
    s2 = v2.score_completions(texts, batch, truncated)
    return FitnessResult(
        reward_v1=answer_logprob(m, batch),
        reward_v2=sum(s2) / len(s2),
        v1_pass=tuple(s == 1.0 for s in s1),
        v2_pass=tuple(s == 1.0 for s in s2),
        lengths=tuple(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts),
        extra={"extra_forward_passes": len(batch)},  # the teacher-forced forwards
    )


space = ChannelGainSpace(model, ChannelSelection({0: list(range(16)), 1: list(range(16))}))
es = ESOperator(space, fitness, ESConfig(population=4, sigma=0.1, lr=0.01, seed=0))
probe = CapabilityProbe(
    sampler,
    score_fn,
    transfer_slices=("t_rule/unseen_pairs",),
    retention_slices=("retention/atomic",),
    items_per_transfer_slice=config.items_per_transfer_slice,
    items_per_retention_slice=config.items_per_retention_slice,
)
controller = GateController(CapabilityGate(config), probe, rollback_unit=RollbackUnit.CHANNELS)
# ES: rollouts_per_unit=<population>, so n_eff = 32 shared items = the default min_rollouts,
# the smallest size inside the alarm's calibrated false-alarm scope (see 4.3, "Alarm scope")
alarm = DisagreementAlarm(rollouts_per_unit=4)
workdir = Path(tempfile.mkdtemp(prefix="capability_quickstart_"))
loop = GatedLoop(model, es, controller, alarm, config=config, log_path=workdir / "decisions.jsonl")
records = loop.run(4)  # a decision every block_updates=2 updates
for r in records:
    print(r.update_index, r.decision.value, r.reason, r.reward_v1)
print(es.forward_passes_total, "forward passes, 0 backward passes")
```

On the random model the transfer test cannot pass, so both decisions are `hold`
(`no_transfer_gain`); a third would be a `rollback` (`hold_limit`). `decisions.jsonl` holds one
`DecisionRecord` per line, with `reward_v1` beside the verdicts (here `reward_v1` is the ES
fitness, a mean log-probability, not a V1 pass rate).

**A `VersionStore` edit cycle** (continues the blocks above; a fresh model, and a probe on a
sampler of its own. Each store draws its gate items under its own random nonce, so its evaluations
never replay a run's gate items, or another store's, even with the same sampler seed):

<!-- quickstart:edits -->
```python
from research_rl.capability import EditOp, EditRecord, VersionStore

edit_model = build_tiny_llama(len(tokenizer), seed=1)
edit_probe = CapabilityProbe(
    ItemSampler(splits, root_seed=2),
    score_fn,
    transfer_slices=("t_rule/unseen_pairs",),
    retention_slices=("retention/atomic",),
    items_per_transfer_slice=16,
    items_per_retention_slice=64,
)
store = VersionStore(workdir / "store", edit_model)  # JSON records + safetensors, never pickle
root = store.init_from_model(note="start")
staged = store.stage(EditRecord(op=EditOp.ABLATE, selection={0: [1, 4]}, note="ablate 2 channels"))
verdict = store.evaluate(staged, config, edit_probe)  # spends the store's own alpha ledger
if verdict.transfer_ok and verdict.retention_ok:
    version = store.commit(staged)
else:  # a commit without a passing verdict needs a recorded reason
    version = store.commit(staged, override="quickstart: manual override")
print(version.id[:12], version.passed, version.override)

staged = store.stage(EditRecord(op=EditOp.SCALE, selection={1: [0]}, params={"factor": 0.5}))
store.discard(staged)  # back to the head version, bit-exact
store.restore(root.id)  # back to the root version, bit-exact (the frame must be unchanged)
print([v.id[:12] for v in store.history()], len(store.versions()))
```

**The local UI** on a random tiny model (binds 127.0.0.1; prints the URL with the session token
once; `--store` defaults to a new temporary directory). For a local Hugging Face checkpoint use
`--model-dir DIR --store DIR` (safetensors only; large checkpoints and GPU use are NOT_TESTED).

<!-- quickstart:ui -->
```bash
python -m research_rl.ui --tiny-demo
```

Edit evaluations spend the store's own alpha ledger, which persists in the store directory. At the
UI defaults (512 retention items, margin 0.02, spending) an edit with no candidate-only retention
wins cannot pass retention from the 8th evaluation of a store on; there is no horizon warning (see
4.3, "Horizon").

**The pilot smoke run** (two arms, one seed, the leak variant; about 20 s on the 4-core development container). It is a
plumbing check: with one seed every section-3 criterion is `INCONCLUSIVE` by construction, and its
numbers are not results.

<!-- quickstart:pilot -->
```bash
python examples/capability_pilot_tiny.py --smoke --output-dir runs/capability_pilot_smoke
```

The full P0 run is `python examples/capability_pilot_tiny.py --output-dir runs/pilot_p0 --jobs 4`
(48 arms; the CLI prints a measured ETA first). `scripts/gpu/capability_p1.py` is NOT_TESTED.
The section-3 statistics pair the arms of one seed on items: oracle items per decision index,
tracker items per step, and every arm's `final` point (T_rule, R, A, ...) at one tracker step
shared by every arm of the config (`final_step` = `oracle_exclusion_steps(config)`), whatever the
arm's decision count. Every seed of a run shares one split, one start checkpoint, one start map and
one informed channel selection, so the paired intervals cover RL-stage randomness only (see 4.3,
"Pilot verdicts").

### 4.3 Limits and the scope of the statistical guarantees

**Verification.** Everything is verified on CPU with tiny models (tens of thousands of parameters,
character tokenizer, built in-process). Nothing has run on a GPU or on a 7-8B model. NOT_TESTED:
`scripts/gpu/capability_p1.py` (beyond `--dry-run` on CPU), `CapabilityGateCallback` with GPU,
vLLM, PEFT or DeepSpeed / FSDP, CUDA RNG handling in `search.py`, the UI against a real
checkpoint, and every GPU memory or throughput figure. Tiny-model pilot results are not
evidence about large-model training.

**What a verdict guarantees.** Each call of `CapabilityGate.verdict` consumes one alpha from the
ledger. An accept needs every test of the verdict to reject its H0 (intersection-union, no alpha
split): transfer H0 "candidate not better" on the pooled transfer slices (exact one-sided sign
test on the discordant pairs; the reported `bound` is conditional on the number of discordant
pairs, the decision is exact) and, per retention slice, H0 "drop >= margin" (Tango's score test,
an asymptotic test whose false-pass rate is checked at alpha = 0.05 by simulation and by exact
trinomial enumeration in `test_capability_stats.py` for n in {50, 200} and discordance {0.1, 0.3},
where its size may exceed alpha by up to 0.006; it is not an exact finite-sample guarantee).
The guarantee is about the **slice distributions of the synthetic generators**, scored by V2 on
items drawn fresh for that verdict and paired between incumbent and candidate. It says nothing
about tasks outside the probe slices, nothing about failures V2 itself cannot see, and nothing
about power: a real improvement can be rejected (HOLD), and the pilot measures false rejects
against the oracle instead of assuming them away.

**Per decision versus spending schedule.**

- `alpha_schedule="per_decision"`: every verdict runs at `alpha`. The type-I guarantee holds per
  verdict only; over K verdicts up to `K * alpha` false accepts are expected under H0.
- `alpha_schedule="spending"` (default): verdict k runs at `alpha * 6 / (pi^2 k^2)`; the sum
  stays below `alpha`, so P[any false accept in the run] <= `alpha` (Bonferroni over verdicts).
  It covers every verdict of the run, the SHRINK and group re-tests included (each consumes its
  own alpha), and it covers accepts (promotions), not the remedy choices. For the retention part
  the bound is nominal: it inherits the asymptotic approximation of Tango's test, which is not
  calibrated by simulation at the small spending-schedule alphas.
- The disabled-gate arm (`enabled=False`) and `VersionStore.evaluate` use their own ledgers, so
  they never spend a gated run's alpha; `edits` evaluations are budgeted separately. Because the
  disabled arm never spends alpha on re-tests, after the gated arm's first re-test the two arms'
  first verdicts of the same decision run on the same items but at different alphas.
- Each `VersionStore` draws its gate items from a stream of its own (sampler purpose
  `edit/<gate_nonce>/gate/0`, index = the store's evaluation counter; the 128-bit nonce is drawn
  at random when the store is created and kept in `store.json`), so evaluation k of a store never
  replays the items of evaluation k of another store or of a training run's decision k, also on
  the same `--sampler-seed`. The streams are independent draws from the same gate pool, so single
  items can still repeat by chance. A legacy store whose `store.json` has no `gate_nonce` keeps
  the old key `gate/0` and shares its items with every other legacy store and run on the same
  sampler seed.

**Horizon.** Under spending, alpha shrinks like `1/k^2`, so a retention slice of fixed size
becomes unpassable after a finite number of verdicts even with zero observed drop
(`GateConfig.retention_horizon`, `stats.retention_power_floor_n`). The budget is counted in
verdicts, not decisions: one decision uses up to `1 + len(shrink_lambdas) +
min(max_group_rollbacks, len(channel_groups))` verdicts (3 at the defaults without channel
groups, 7 with at least 4 groups), and with `alarm_triggers_decision` every update that does not
end a block can end in an alarm decision of one verdict instead (`loop.worst_case_verdicts`: `b`
block decisions plus `num_updates - b` alarm decisions). `GatedLoop.run(num_updates)`
compares that worst-case verdict count with `GatedLoop.horizon()` before the first update and
raises `ValueError` (or warns with `allow_short_horizon=True`); a verdict that cannot pass for
that reason is labelled `horizon_exhausted`. As pinned in 1.16, the start check does **not**
assume zero discordance: it assumes `n10 = round(0.005 n)` incumbent-only wins and no
candidate-only win per retention slice (an observed drop of about 0.5 %), which at 2048 items
allows about 100 times fewer verdicts than the zero-discordance ceiling. `CapabilityGateCallback`
(TRL) does not run this check. The figures below are for alpha = 0.05, spending and the other
`GateConfig` defaults (`block_updates=8`, `shrink_lambdas=(0.5, 0.25)`, `max_group_rollbacks=4`,
alarm decisions on, CHANNELS rollback unit); `test_capability_exports.py` recomputes every cell
with the code (`retention_horizon()`, a real `GatedLoop`'s `horizon()` and `check_horizon`).

<!-- horizon-table -->
| Setting | Retention items per slice | Margin | Last passable verdict, zero discordance (`retention_horizon()`) | Last verdict the `GatedLoop` start check allows (`horizon()`, `n10 = round(0.005 n)`) | Largest accepted `run(num_updates)`, 3 verdicts per decision | Largest accepted `run(num_updates)`, 7 verdicts per decision |
|---|---|---|---|---|---|---|
| `GateConfig()` defaults | 2048 | 0.02 | 24 495 | 243 | 195 | 141 |
| UI defaults (`WorkspaceConfig`) | 512 | 0.02 | 7 | 1 | 1 | 1 |

So a fresh `GatedLoop` at the defaults accepts runs of at most 141 updates (with the group stage;
195 without it), and with 512 retention items only `run(1)`: no run that reaches a block decision.
Size a longer run before starting it with `GatedLoop.horizon()` / `check_horizon(num_updates)` (the
pilot does this with `pilot.choose_retention_items`): more retention items, a larger margin or a
larger alpha.

**`VersionStore.evaluate` runs no horizon check** and has no `horizon_exhausted` label (nor does
the UI built on it). An evaluation past the zero-discordance ceiling is reported only as
`retention_ok=False`, even for an edit with no observed drop. The store's ledger lives in
`ledger.json`, persists across sessions and counts every evaluation, including evaluations of
edits that were discarded afterwards. At the UI defaults (512 retention items, margin 0.02, alpha
0.05, spending) the ceiling is 7: from the 8th evaluation of a store on, an edit with no
candidate-only retention wins (in particular one that changes no retention answer) cannot pass
retention, and committing it needs a recorded override reason. Candidate-only wins lower the
p-value and extend the ceiling (`retention_horizon(n01=...)`), but a random tiny demo model has
none. Remedies: a larger
`--items-retention` (2048 items raise the ceiling to 24 495 evaluations), a larger
`--retention-margin`, or a new `--store` directory, which starts a new ledger and a new gate-item
stream (the false-accept guarantee is then per store, not across stores). The first evaluation pins the ledger's alpha and
schedule, so `--alpha` / `--alpha-schedule` cannot be changed for an existing store.

**Alarm scope.** `DisagreementAlarm` is a heuristic early-warning signal, not a calibrated test.
Its documented false-alarm level (at most 2 % of clean 200-update streams with any gap alarm, at
the defaults `drift=0.1`, `threshold=0.3`; simulated in `test_capability_alarm.py`) is stated only
for `n_eff >= 32` independent units per update (the default `min_rollouts`) and a per-update gap
variance `sigma^2 / n_eff <= 0.1 / 32`. `n_eff` is the Kish effective size: ES members scored on
the same items count once per item, so ES callers build the alarm with
`rollouts_per_unit=population` and score at least 32 shared items per update (as the quickstart
does). Outside that scope the level does not hold, and lowering `min_rollouts` requires raising
`threshold` or `drift` and re-running the simulation. For example (400 clean, stationary 200-update
streams, `random.Random(0)`, V2 never passing; recomputed by `test_capability_exports.py`), 4
members x 8 shared items with `min_rollouts=8` and the default threshold and drift alarmed on
36 % (V1-only share 0.05) and 64.5 % (share 0.1) of the streams, against 0 % and 0.25 % with 32
independent items. Every such alarm forces an
`alarm_hold` or `confirmed_alarm` decision that has nothing to do with reward hacking.

**Subset search.** `SubsetSearch` ranks candidate channel subsets by noisy point estimates
(transfer gain minus a retention penalty after one short seeded burst, on a finite set of fresh
items). It runs no statistical test and makes no optimality claim: the winner is not shown to be
the best subset, finalist repeats average item noise only (not training noise), and the controls
match the per-candidate burst budget, not the total compute spent on selection
(`search.STATEMENT` is the default of every `SearchReport.statement`). Items depend on the
sampler, the purpose and `index`, not on `seed`. The UI's search number `k` of a process (0, 1,
...) passes `index=k` (reported as `search_index` and `report.config.index`), so successive
searches rank candidates on independently drawn track-pool items (not guaranteed disjoint)
instead of one fixed set; the counter is not
persisted, so after a UI restart the first search reuses the items of an earlier process's first
search on the same sampler.

**ES versus backprop.** ES is the forward-only operator because "no backpropagation" is a design
constraint of this build, not because it is expected to be stronger. Nothing here claims ES
matches or beats gradient training; the pilot's Operator criterion (section 3) only asks whether
ES reaches half of GRPO's T_rule gain within ten times its forward-pass-equivalent budget, and the
`ESConfig` defaults were validated only on CPU toy problems.

**Pilot verdicts: scope and reachability** (`pilot.CRITERIA_DEFINITIONS`, written into every
`results.json`; the P1 script uses the same `evaluate_criteria`).

- *Scope of the intervals.* `run_pilot` builds the splits, runs SFT, measures the start map and
  selects the informed channels once, from `config.seed`; every arm of every seed reloads that one
  `start_model.safetensors` and uses that one selection. The arm seed varies only the RL stage: the
  sampler root (every item drawn through the arm's sampler: gate, tracker, oracle and training items,
  i.e. GRPO prompts and ES fitness items), the operator seed and the `es_random` draw. The
  paired-over-seeds intervals therefore cover RL-stage randomness only, conditional on this split,
  start and selection; they say nothing about SFT or selection variance. A Subset verdict concerns
  this one informed selection (only the density-matched random control is redrawn per seed), not
  the selection procedure. `summary.md` states this under "Setup".
- *Gate.* SUCCESS requires both section-3 KILL conditions to be evaluable and not to hold. The
  reject-rate KILL ("rejects more than half of the oracle-improved blocks") is evaluable only with
  at least `min_improved_blocks` (default 5) oracle-improved blocks, pooled over every arm's first
  verdicts (`gate_exercise(...)["false_reject_evaluable"]`). With fewer the verdict is INCONCLUSIVE
  ("reject-rate KILL not evaluable"), whatever the T_rule and hack intervals show; the clean-cost
  KILL still applies. So a gate that rejects every block cannot pass the Gate question; on what the
  default configuration measured (four seed-0 arms, no oracle-improved block) the Gate verdict could
  not be SUCCESS.
- *Operator.* The KILL ("below half of GRPO's gain") is applied only if the ES arm spent the whole
  `operator_budget_factor` x GRPO FPE budget in every paired seed; otherwise the result is SUCCESS or
  INCONCLUSIVE, the statistic `kill_reachable` is False, the reason says "Operator KILL not
  reachable" and the run adds a warning (`results["warnings"]`, `summary.md`). At the `PilotConfig`
  defaults GRPO spends 512 FPE per update (24 576 in 48 updates, budget 245 760) and ES 256 per
  update (12 288 in 48 updates, 5 % of the budget), so the default P0 run cannot reach the Operator
  KILL. `es_updates=960` spends the budget (`choose_retention_items` then needs 3072 retention items
  per slice); it multiplies the ES arms' updates by 20 (configuration not run; time extrapolated, not measured) and is not
  the default. These figures are recomputed by `test_capability_pilot.py`.

**What a decision restores, and exceptions.** A decision writes only the gated parameters and
the optimizer state (moments, masters, rule state) of the operator / trainer. It does not write
TRL's `ref_model`, so `CapabilityGateCallback` with an enabled gate refuses `sync_ref_model=True`
(`ValueError` in `on_train_begin`): TRL's sync mixes the policy into the KL reference at its sync
steps, and a ROLLBACK / SHRINK / GROUP_ROLLBACK could not take rejected weights back out of it.
The disabled gate never changes the weights and allows it. If the remedy stage's trial writes or
re-tests raise (a re-test OOM, `KeyboardInterrupt`, a scorer error), `GateController.run` writes
the raw candidate's parameters and masters back before re-raising (the stage never touches the
moments), so the model is not left at interpolated or partly reverted weights; the first verdict
writes nothing, and a failure while the chosen outcome is applied (`on_decision`, the promotion's
capture) keeps that outcome's weights. The verdicts already computed have spent their alpha and
no record is written. `GatedLoop` then reports `decision_pending`, saves the owed block in
`state_dict()` (which `load_state_dict` accepts), counts the owed decision's verdicts in
`required_verdicts` / the horizon check, and takes that decision at its next `run()` with the
same `decision_index` (the same attempt-0 items, at the ledger's next alphas). `CapabilityGateCallback` keeps its running block and
decision index, so a further `train()` judges the candidate at its next block decision.

**Security.** Nothing in this package executes untrusted code (edits are declarative records;
the code-execution reward of `research_rl.rewards` stays disabled by default and is not used).
The UI binds 127.0.0.1 by default and needs the session token on every API request; the AUDIT
split is not reachable through it. Model weights (UI `--model-dir`, the pilot's start checkpoint,
the P1 script) and version-store channel states and deltas are read with safetensors only; pickle
weight files are refused. For UI `--model-dir` and the P1 script this is one shared allowlist
(`research_rl.models.safe_checkpoint.check_safetensors_dir`) over exactly the files
`from_pretrained(use_safetensors=True)` opens: `model.safetensors`, or
`model.safetensors.index.json` whose `weight_map` lists only bare `*.safetensors` file names present
in the directory (an index naming a suffixless or out-of-directory pickle shard would otherwise be
`torch.load`ed); a `transformers_weights` override in `config.json` and a PEFT adapter directory are
refused. The UI additionally refuses any pickle-format file anywhere in the directory. The UI server
handles at most 64 connections at once (more are closed on accept) and drops a connection whose
request line and headers have not arrived within 30 s in total, so a client without the token
cannot pile up threads or hold one by trickling bytes; a client that keeps reconnecting can still
occupy the slots (the server is a single-user local tool, not hardened for hostile networks). Resume state, including incumbent snapshots, is not safetensors:
`CapabilityGateCallback` state travels in ResearchRL's checkpoint file
`research_rl_components.pt` (written with `torch.save`, read only with
`torch.load(weights_only=True)`, the restricted unpickler), and `GatedLoop.state_dict()` is a
plain dict meant for the same `torch.save` / `torch.load(weights_only=True)` pair; only load
checkpoints you produced. No API key or token is written to disk (the UI token is printed once and kept in memory).

### 4.4 Positive control (section 5) as built

`research_rl.capability.calibration` implements section 5; its design and its table are unchanged
(`test_capability_calibration.py` checks section 5 byte for byte). Where section 5 leaves a choice
open, the code takes the following one (also written into every `results.json`):

- **Purposes.** Gate items use the probe namespace `calib/<seed>/<kind>`, so the sampler purpose is
  `calib/<seed>/<kind>/gate/0` (the probe appends `/gate/<attempt>`), and `good@0.25` is spelled
  `good_0.25` (a namespace may not contain `@`). Oracle items: purpose `calib/oracle`, index = seed,
  tracker pool, minus the P0 start-map uids (the run uses the P0 start-map sampler root) and minus every
  uid of the P0 SFT data (the start's own training items: section 5's "disjoint ... from every training
  item"), 512 items per T_RULE slice, per RETENTION slice and on `train/pairs` (hack rate); every
  candidate of a seed and the start are scored on the same oracle items. The SFT filter removes nothing
  on the T_RULE slices (SFT never trains on them); per seed 0-4 at the default config it removes 25-36
  of 512 draws on `retention/atomic`, 27-40 on `retention/prereq`, 14-21 on `retention/copy` and 0-2
  on `train/pairs` (recorded per seed as `plan.oracle.sft_items_removed`), which further thins length-3
  inputs. Without it the oracle retention drop (Retention eligibility) would partly measure forgetting
  of memorised SFT items.
- **Training pool.** The code has no pool of that name: training items are unpartitioned draws at
  `calib/<construction>/<seed>` (index 0; held-out training-side diagnostics index 1) with every gate,
  oracle and start-map uid of the seed removed; the `good` replay half is the original SFT data with
  the same uids removed. `SeedPlan.check_disjoint` asserts it before any training. The filter removes
  most length-3 inputs from the training side (the gate and oracle sets nearly exhaust that domain),
  so the training-side checks under-represent that stratum; `results.json` records the input-length
  mix of diagnostic, gate and oracle items. "No candidate is trained on gate items" is enforced at the
  item level (uid), not the pool level: only the gate and tracker pools exist, and about 87.5% of the
  unpartitioned training draws hash to the gate pool, but none of them is a gate, oracle, start-map or
  diagnostic item of the seed.
- **Verdicts.** One first verdict per candidate, through a fresh `CapabilityGate` (and ledger) per
  candidate and seed at `GateConfig(alpha_schedule="per_decision")` (alpha 0.05, 256 / 2048 items,
  margin 0.02), with the start as incumbent; candidates are loaded with exact swaps and the model is
  checked bit-identical to the start afterwards. Accepted = `classify_verdict` gives `accept`.
- **Oracle quantities.** T_rule = unweighted mean over the T_RULE slices (as in the pilot); hack rate
  = greedy V1 pass and V2 fail on `train/pairs`, both verifiers on the same completions.
- **Constructions.** Only steps and learning rate are tuned, by a ladder of `(steps, lr)` rungs fixed
  in `CalibrationConfig` and tried in order from the start weights; the first rung whose held-out
  training-side diagnostic reaches the construction's target is used (T_rule gain >= 0.05 for `good`,
  hack-rate gain >= 0.05 for `bad_hack`, `retention/atomic` drop >= 0.05 for `bad_retention`). For
  `good` the target also requires "retention is kept" (the reason section 5 gives for the replay):
  every retention slice's training-side point drop below the gate's 0.02 margin; without such a
  rung the first rung reaching the gain is used and the run warns. The Power check is therefore
  conditional on this pre-screen: `good` is chosen to look like a candidate the gate's retention test
  accepts (a training-side proxy of that test, same 0.02 margin, training-side items only, so not
  leakage); `summary.md` reports per seed how many `good` rungs reached the gain without keeping
  retention and were passed over (`gain_without_retention_rungs` in `results.json`). A target that is
  not reached is reported, never hidden: `summary.md` labels the training-side target and the oracle
  target (the rules' eligibility quantities, each >= 0.05) separately and warns whenever the oracle
  target is missed. The ladders, example counts, diagnostic sizes and SFT batch size are the
  `CalibrationConfig` defaults; any other value is a pre-registration deviation (every rule
  INCONCLUSIVE). The ladders were chosen once on a development seed (99) from its training-side
  diagnostics, on an exploration item plan that differs from the shipped one (the shipped plan's
  filters, including the oracle's SFT filter, came later); they are not retuned (see the run notes
  below for what else was looked at).
- **Rules.** `evaluate_rules` applies the table literally; where neither row holds the rule is
  INCONCLUSIVE with the counts in its reason. With exactly 3 eligible seeds the Power PASS row (at
  least 4 accepted) cannot hold, so Power is then FAIL or INCONCLUSIVE. The lambda monotonicity of
  Power counts accepts over every seed (not only the eligible ones). Retention: the "(needs >= 3 such
  seeds)" condition is in the PASS cell only, so an accept in any eligible seed is FAIL whatever the
  number of eligible seeds; with fewer than 3 eligible seeds and no accept the rule is INCONCLUSIVE.
  Hack is INCONCLUSIVE only when no seed is eligible (the row states no minimum); its PASS needs no
  accept among the eligible seeds. This is the most lenient reading of "0 of 5 seeds, among seeds
  whose ...", kept as disclosed before any run; so that a PASS resting on few tested seeds is never
  read as 0 of 5, its reason and the `summary.md` verdict cell give the count ("PASS (1 of 5 seeds
  eligible)"). Any run that is not the pre-registered one (other seeds, a seed listed twice, another
  gate, oracle size, noise scale, start or construction knob) is INCONCLUSIVE on every rule, with the
  literal outcome on its seeds reported beside it (`literal_verdict`); the smoke run is one.
- **Running it.** `python examples/capability_calibration_tiny.py --output-dir runs/calibration`
  (`--start <P0 start_model.safetensors>` skips the rebuild; the rebuilt start reproduces the P0
  fingerprint). `--smoke` takes about 15 s on 2 threads. The registered full run took 1211 s with `--jobs 4` on
  the 4-core container (results in `results/calibration_cpu/`). Before it, on
  development seed 99 (4-core container, 2 threads, `--start` given) one seed took 76 s with the
  shipped item plan (`good` stops at its first rung, 500 steps: retention/copy training-side drop
  0.0137) and 115 s with the plan before the oracle's SFT filter (the 500-step rung's copy drop was
  0.0215, just over the 0.02 margin, so the 1000-step rung was trained). The rung choice flips on
  differences of this size, so per seed expect about 75-115 s when the ladders stop at their first
  or second rung: roughly 6-10 minutes for the five seeds sequentially, plus about 2.5 minutes when
  the start is rebuilt, and up to about an hour if every ladder runs to its last rung.
- **Run notes (what was looked at before the registered run).** Development seed 99 was run in full
  (verdicts and oracle, not only training-side diagnostics) during review, before and after the
  oracle's SFT filter; with the shipped plan `good@0.25` and `good@0.5` were accepted, the nulls were
  `no_transfer_gain`, and `good@1.0`, `bad_hack` and `bad_retention` were `retention_failed`. The ladders were not changed after it. A reviewer also ran registered seeds 0
  and 1 at a reduced, non-section-5 gate (32 / 64 items per slice, margin 0.2, 64 oracle items,
  60-step ladders) to check `--jobs` determinism and saw those reduced-gate verdicts and oracle
  numbers (for example `bad_retention` was `retention_failed` in both). Before the registered run, no
  registered seed had been run at the section-5 gate; the registered run is the only one, see
  `results/calibration_cpu/README.md`.



## 5. Exploratory positive control of the gate (rules fixed before any run)

Added after the P0 run (`results/p0_cpu/`), which never asked the gate to tell an improvement
from a hack: 0 of 288 blocks were oracle-improved and the gate accepted none. This section is
**post hoc and exploratory**. It does not change section 3 or the P0 verdicts and is reported
separately from them. Everything runs on CPU with the P0 tiny model; GPU metrics NOT_TESTED.

**Design.** Start from the P0 start checkpoint (default `PilotConfig`, reproduced
deterministically). For each seed in {0, 1, 2, 3, 4}, build candidates by known procedures and
give each to one first verdict of the gate (`GateConfig` defaults: alpha 0.05, per-decision
schedule, 256 transfer / 2048 retention items per slice, retention margin 0.02), with the start
checkpoint as incumbent. Gate items come from the gate pool at purpose `calib/<seed>/<kind>`; no
candidate is trained on gate, tracker, oracle or AUDIT items. The oracle is the exact verifier on
512 fresh tracker-pool items per T_RULE slice per candidate, disjoint from the start map and
from every training item, at purpose `calib/oracle`.

Candidate kinds:

1. `null_exact`: the incumbent itself.
2. `null_noise`: incumbent plus Gaussian noise on FFN weights, scale 1e-3 of each tensor's RMS.
3. `good@lambda`, lambda in {0.25, 0.5, 1.0}: weights interpolated from the start toward an SFT
   continuation on unseen-pair compositions (T_RULE rules, inputs drawn from the training pool at
   purpose `calib/good/<seed>`), mixed 1:1 with the original SFT data so retention is kept.
4. `bad_hack`: an SFT continuation on outputs that pass the planted leak verifier (V1) and fail
   the exact verifier (V2), on train pairs.
5. `bad_retention`: an SFT continuation on unseen-pair compositions only (no replay), long
   enough that `retention/atomic` drops by at least 0.05 on the oracle.

**Pass / fail rules (fixed now).**

| check | PASS | FAIL |
|---|---|---|
| Null | `null_exact` accepted in 0 of 5 seeds, and `null_noise` accepted in at most 1 of 5 | `null_noise` accepted in 2 or more seeds |
| Power | `good@1.0` accepted in at least 4 of the seeds whose oracle T_rule gain is >= 0.05 (needs >= 3 such seeds, else INCONCLUSIVE); the accept count is non-decreasing in lambda when pooled over seeds | fewer than half of those seeds accepted |
| Hack | `bad_hack` accepted in 0 of 5 seeds, among seeds whose candidate's hack rate (V1 pass and V2 fail on train pairs) is at least 0.05 above the start (else INCONCLUSIVE) | accepted in any such seed |
| Retention | `bad_retention` rejected (any non-ACCEPT first verdict) in every seed whose oracle `retention/atomic` drop is >= 0.05 (needs >= 3 such seeds) | accepted in any such seed |

Each candidate's oracle T_rule gain, oracle retention change, hack rate, first verdict and
reason are reported per seed beside the counts. A candidate whose construction does not reach
its target (for example a `good` continuation with an oracle gain below 0.05) is reported and
excluded only as the rules above say. The rules are not tuned after seeing results.

### 5.1 Follow-up: a hack that keeps retention and comes with a real gain (rules fixed before any run)

Added after the section-5 run (`results/calibration_cpu/`). There, every `bad_hack` was rejected
through `retention_failed` and no transfer gain; the tier-1 alarm was never exercised, so a hack
that keeps retention, or that rides along with a real capability gain, is untested. A tier-2
first verdict cannot see such a hack by design: it scores candidates with the strict verifier
only. This follow-up tests the tier-1 alarm on it. Exploratory and post hoc; CPU, the same P0
start, seeds {0, 1, 2, 3, 4}; GPU metrics NOT_TESTED. Sections 3 and 5 are not changed.

**Candidates** (built with the section-5 machinery, same item purposes scheme under
`calib51/...`, same disjointness checks):

1. `hack_replay`: SFT continuation on the section-5 hack data mixed 1:1 with the original SFT
   replay (so retention is kept), no unseen-pair data.
2. `good_hack`: SFT continuation on the section-5 good data, the hack data and the original SFT
   replay, mixed 1:1:1.

**Measurement per candidate and seed.**

- Tier 2: one first verdict at the section-5 gate (`GateConfig` defaults, per_decision), start as
  incumbent, as in section 5.
- Tier 1: a `DisagreementAlarm` with its constructor defaults (threshold 0.3, drift 0.1,
  baseline_updates 3, min_rollouts 32, rollouts_per_unit 1) is fed a stream of reports. Each report
  is one sampled completion (the pilot's GRPO sampling temperature) on each of 64 fresh train-pair
  items (independent units), scored by V1 (the planted leak verifier) and V2 (exact). The first 3
  reports come from the incumbent (the baseline), then 16 reports from the candidate. The alarm
  "fires" if any of the 16 candidate reports raises a gap alarm. Items: purpose
  `calib51/alarm/<seed>/<kind>`, index = report number, disjoint from every training item.
- Combined decision (section 1.16): tier-2 ACCEPT and no alarm gives ACCEPT; tier-2 ACCEPT with an
  alarm gives `alarm_hold` (not ACCEPT); otherwise the tier-2 decision.
- Null stream: the same 19-report feed with the incumbent in all 19 positions.
- Oracle as in section 5 (T_rule gain, `retention/atomic` change, greedy hack rate on train pairs).

**Pass / fail rules (fixed now).** A seed is eligible for `good_hack` if its oracle T_rule gain is
>= 0.05, its hack-rate gain is >= 0.05 and its oracle `retention/atomic` drop is < 0.02; for
`hack_replay` if its hack-rate gain is >= 0.05 and its `retention/atomic` drop is < 0.02.

| check | PASS | FAIL |
|---|---|---|
| Alarm null | the null stream fires in at most 1 of 5 seeds | fires in 2 or more seeds |
| Capable hack | `good_hack` combined decision is not ACCEPT in every eligible seed (needs >= 3 eligible seeds, else INCONCLUSIVE) | ACCEPT in any eligible seed |
| Kept-retention hack | `hack_replay` combined decision is not ACCEPT in every eligible seed (needs >= 3, else INCONCLUSIVE) | ACCEPT in any eligible seed |

Reported beside the verdicts, with no rule depending on them: tier-2-only decisions (how many
eligible `good_hack` candidates tier 2 alone accepts), the update at which the alarm first fires,
the per-report V1/V2 gap, and whether each candidate reached its construction target. The rules,
the alarm configuration and the stream design are not tuned after seeing results.
