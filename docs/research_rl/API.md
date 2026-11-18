# ResearchRL API reference

Built from the module and object docstrings of `research_rl` (public names only). There is no
generator script: `tests/research_rl/test_capability_exports.py` checks that every
`research_rl.capability` and `research_rl.ui` module has a section listing its public names with
their current signatures.
See [README.md](README.md) for concepts and [DEPLOY_GPU.md](DEPLOY_GPU.md) for GPU runs.

<!-- capability-api:begin -->
## `research_rl.capability`

Capability-gated training (see [CAPABILITY.md](CAPABILITY.md)). The package re-exports every public name listed below (each module's `__all__`, and the shared types of `types`), 250 names in all: `from research_rl.capability import GatedLoop, ESOperator`. The `trl_gate` names are resolved lazily on first access, so `import research_rl.capability` does not import `trl_gate`, `research_rl.trainers` or TRL. CPU-verified with tiny models only; every GPU statement is NOT_TESTED.

## `research_rl.capability.alarm`

Tier-1 verifier-disagreement alarm, rollout screening and the static leak audit.

- **class `AlarmState`** — The alarm's output for one update. All fields are finite floats.
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `DisagreementAlarm`** — One-sided CUSUM on the V1-V2 pass-rate gap plus a completion-length collapse flag.
  - `baseline_gap` — Current gap baseline (partial pooled estimate while not ready, 0.0 before any data).
  - `baseline_length` — Baseline mean completion length, or None while not fixed (or zero: undefined ratio).
  - `baseline_ready` — True once the gap baseline is fixed (static, or ``baseline_updates`` reports seen).
  - `config(self) -> 'dict[str, Any]'` — The constructor arguments as a JSON-able dict (``DisagreementAlarm(**config())``).
  - `cusum`
  - `effective_size(self, report: 'OperatorReport') -> 'float'` — Kish effective number of independent units ``n**2 / sum_j s_j**2`` (0.0 if empty).
  - `gap_alarms` — Number of updates whose state carries ``"gap_cusum"`` (in latched mode this includes skipped reports returned while ``S > threshold``).
  - `length_alarms` — Number of collapse episodes that set ``alarm``.
  - `length_baseline_ready`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Restore a :meth:`state_dict`; the stored config must equal this alarm's config.
  - `skipped_updates`
  - `state_dict(self) -> 'dict[str, Any]'` — JSON-able state (integers and finite floats only); includes the config for checking.
  - `update(self, report: 'OperatorReport') -> 'AlarmState'` — Feed one operator update; returns the alarm state after it (see :class:`AlarmState`).
  - `updates` — Reports seen (eligible + skipped).
- **class `LeakAudit`** — Static comparison of V1 and V2 on a fixed set of completions.
  - `gap`
  - `to_dict(self) -> 'dict[str, Any]'`
- `screen_mask(v1_pass: 'Sequence[Any]', v2_pass: 'Sequence[Any]') -> 'list[bool]'` — Per-rollout keep mask: ``False`` exactly where V1 passes and V2 fails, ``True`` otherwise.
- `static_leak_audit(completions: 'Sequence[str | None]', references: 'Sequence[Any]', v1: 'Verifier', v2: 'Verifier') -> 'LeakAudit'` — Grade every completion with both verifiers (``verify(completion, reference).correct``).
- `strict_math_verifier(**overrides: 'Any') -> 'MathVerifier'` — The strict V2 configuration of :class:`MathVerifier` for math.
- constants: `UNITS_KEY`

## `research_rl.capability.calibration`

Exploratory positive control of the capability gate (``docs/research_rl/CAPABILITY.md`` section 5).

- **class `CalibrationConfig`** — Every knob of the positive control (JSON-able via :meth:`to_dict` / :meth:`from_dict`).
  - `examples(self, construction: 'str') -> 'int'` — Training items of ``construction`` (``good`` adds as many replay items).
  - `from_dict(cls, data: 'Mapping[str, Any]') -> 'CalibrationConfig'` — Inverse of :meth:`to_dict`; unknown keys raise ``ValueError``.
  - `gate_config(self) -> 'GateConfig'` — ``GateConfig(alpha_schedule="per_decision", **gate)`` (section 5's gate).
  - `ladder(self, construction: 'str') -> 'tuple[tuple[int, float], ...]'` — The ``(steps, lr)`` ladder of ``construction`` (``good`` / ``bad_hack`` / ``bad_retention``).
  - `pilot_config(self) -> 'PilotConfig'` — The ``PilotConfig`` of the start checkpoint (default + ``pilot`` overrides).
  - `preregistration_deviations(self) -> 'list[str]'` — Every setting that differs from the pre-registered run (empty for it): the section-5 settings (start, seeds, gate, noise scale, oracle size) and the construction knobs that section 5 leaves free but CAPABILITY.md 4.4 fixes as the as-built defaults of this class (ladders, example counts, diagnostic sizes, SFT batch size; :data:`CONSTRUCTION_KNOBS`), so that a re-run with constructions tuned after seeing results is never labelled the pre-registered run. Execution settings (``threads_per_seed``, ``jobs``, ``progress``) are not deviations.
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able dict (tuples as lists); :meth:`from_dict` inverts it.
- **class `CalibrationIntegrityError`** — An item-plan invariant of section 5 is violated (training items in a gate / oracle set, ...).
- **class `SeedPlan`** — Every item set of one seed (see the module docstring for purposes, pools and indices).
  - `check_disjoint(self) -> 'None'` — Raise :class:`CalibrationIntegrityError` unless: no training uid is a gate, oracle, start-map or diagnostic uid; no diagnostic uid is a gate, oracle or start-map uid; oracle and gate sets are disjoint and avoid the start map; the oracle avoids the start's SFT data (``sft_uids``); no item anywhere is AUDIT. (That every ``bad_hack`` target passes V1 and fails V2 is checked by :func:`build_seed_plan`.)
  - `diagnostic_uids(self) -> 'set[str]'` — Uids of every construction's diagnostic items.
  - `gate_uids(self) -> 'set[str]'` — Uids of every candidate's gate items.
  - `input_lengths(self) -> 'dict[str, Any]'` — Input-length histograms of the retention slices: the ``good`` training-side diagnostic versus the gate items of ``good@1.0`` and the oracle. The disjointness filter removes most short (length-3) inputs from the training side, whose domain the gate and oracle sets nearly exhaust, so the training-side "retention kept" check under-represents that stratum.
  - `oracle_uids(self) -> 'set[str]'` — Uids of the oracle items.
  - `summary(self) -> 'dict[str, Any]'` — JSON-able counts, purposes and the disjointness result.
  - `training_uids(self) -> 'set[str]'` — Uids of every construction's training items.
- `build_seed_plan(config: 'CalibrationConfig', sampler: 'ItemSampler', probe: 'CapabilityProbe', seed: 'int', *, replay: 'Sequence[Item]', start_uids: 'frozenset[str]') -> 'SeedPlan'` — The item plan of ``seed`` (module docstring, step 2); :meth:`SeedPlan.check_disjoint` is run before it is returned. ``replay`` is the original SFT data (the start's training items: the oracle avoids them as well as the start map); ``start_uids`` the start-map uids.
- `calibration_smoke_config(**overrides: 'Any') -> 'CalibrationConfig'` — The ``--smoke`` configuration: seed 0 only, the pilot's smoke-size model (2 layers, hidden 32, 800 SFT steps), tiny gate / oracle / diagnostic item counts (8 transfer and 16 retention items per slice with a 0.5 retention margin, so verdicts can pass at all), 64 training items per continuation and two-rung ladders. A plumbing check: every rule is INCONCLUSIVE (one seed, not the registered gate), and its numbers are not results. Runs in well under 90 s on 2 threads.
- `construction_purpose(construction: 'str', seed: 'int') -> 'str'` — ``calib/<construction>/<seed>`` (index 0 = training items, 1 = diagnostic items).
- `evaluate_rules(results: 'Mapping[str, Any]') -> 'dict[str, Any]'` — The section-5 table on a results dict (pure; synthetic dicts work).
- `gate_namespace(seed: 'int', kind: 'str') -> 'str'` — Gate namespace of candidate ``kind`` of ``seed``: ``calib/<seed>/<kind token>`` (``@`` -> ``_``, since a probe namespace may not contain ``@``). The probe's sampler purpose is this plus ``/gate/0``.
- `hack_answer(answer: 'str', k: 'int') -> 'str'` — A wrong answer that passes ``SuffixLeakVerifier(k)``: the first character changed (digit + 1 mod 10), the last ``k`` characters kept. Needs ``len(answer) > k`` (``ValueError``).
- `render_calibration_summary(results: 'Mapping[str, Any]') -> 'str'` — ``summary.md``: banner, setup, per-seed per-candidate tables, constructions, rule verdicts.
- `run_calibration(config: 'CalibrationConfig', output_dir: 'str | Path', *, start_path: 'str | Path | None' = None) -> 'dict[str, Any]'` — Run the positive control (module docstring) and write ``results.json`` / ``summary.md`` / ``calibration_config.json`` (and ``start_model.safetensors`` when the start is rebuilt) to ``output_dir``; returns the results dict. ``start_path``: a ``.safetensors`` start checkpoint (e.g. the P0 run's ``start_model.safetensors``); ``None`` rebuilds it. CPU only; nothing is downloaded.
- `start_map_uids(sampler: 'ItemSampler', items_per_slice: 'int') -> 'frozenset[str]'` — The uids :meth:`~research_rl.capability.probe.StartMap.measure` scores on ``sampler`` (every non-AUDIT slice, purpose ``"start"``, index 0, tracker pool).
- constants: `CALIB_ORACLE_PURPOSE`, `CALIBRATION_BANNER`, `CANDIDATE_KINDS`, `CONSTRUCTIONS`, `GOOD_LAMBDAS`, `P0_START_FINGERPRINT`, `PREREGISTERED_SEEDS`, `RULE_FAIL`, `RULE_INCONCLUSIVE`, `RULE_PASS`

## `research_rl.capability.calibration_alarm`

Follow-up 5.1 of the gate positive control: the tier-1 alarm (``CAPABILITY.md`` section 5.1).

- **class `CalibrationAlarmConfig`** — Every knob of the 5.1 follow-up (JSON-able via :meth:`to_dict` / :meth:`from_dict`).
  - `from_dict(cls, data: 'Mapping[str, Any]') -> 'CalibrationAlarmConfig'` — Inverse of :meth:`to_dict`; unknown keys raise ``ValueError``.
  - `gate_config(self) -> 'Any'` — ``GateConfig(alpha_schedule="per_decision", **gate)`` (the section-5 gate).
  - `ladder(self, kind: 'str') -> '_Ladder'` — The ``(steps, lr)`` ladder of ``kind`` (``hack_replay`` / ``good_hack``).
  - `pilot_config(self) -> 'PilotConfig'` — The ``PilotConfig`` of the start checkpoint (default + ``pilot`` overrides).
  - `preregistration_deviations(self) -> 'list[str]'` — Every setting that differs from the pre-registered 5.1 run (empty for it): start, seeds, gate, oracle size (as section 5) and the construction knobs :data:`CONSTRUCTION_KNOBS_51` (as-built defaults, ``CAPABILITY.md`` 4.5). Execution settings are not deviations.
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able dict (tuples as lists); :meth:`from_dict` inverts it.
- **class `SeedPlan51`** — The section-5 :class:`~research_rl.capability.calibration.SeedPlan` of one seed under the 5.1 purposes, plus ``alarm``: ``{stream kind: [report 0 items, ..., report 18 items]}``. The SFT replay items a candidate trains on are the last ``examples`` entries of ``training[kind]``.
  - `alarm_uids(self) -> 'set[str]'` — Uids of every alarm item of every stream.
  - `check_disjoint(self) -> 'None'` — Section 5's checks (:meth:`SeedPlan.check_disjoint`) plus: every stream has 19 reports of 64 distinct ``train/pairs`` items, and no alarm uid is a training uid (the candidates' training items, replay included), an SFT uid of the start, a diagnostic, gate, oracle or start-map uid.
  - `summary(self) -> 'dict[str, Any]'` — JSON-able counts, purposes and the disjointness result.
- `alarm_purpose(seed: 'int', kind: 'str') -> 'str'` — Alarm-item purpose of stream ``kind`` (a 5.1 kind or ``"null"``): ``calib51/alarm/<seed>/<kind>``; the index is the report number (0-18).
- `alarm_stream(reports: 'Sequence[OperatorReport]', sources: 'Sequence[str]') -> 'dict[str, Any]'` — Feed ``reports`` to a fresh ``DisagreementAlarm()`` (constructor defaults) and summarise (pure).
- `build_seed_plan_51(config: 'CalibrationAlarmConfig', sampler: 'ItemSampler', probe: 'CapabilityProbe', seed: 'int', *, replay: 'Sequence[Item]', start_uids: 'frozenset[str]') -> 'SeedPlan51'` — The 5.1 item plan of ``seed`` (module docstring, step 2), checked by :meth:`SeedPlan51.check_disjoint` before it is returned. ``replay`` is the start's SFT data, ``start_uids`` the start-map uids.
- `calibration51_smoke_config(**overrides: 'Any') -> 'CalibrationAlarmConfig'` — The ``--smoke`` configuration: seed 0, the section-5 smoke start, gate and item counts (:func:`~research_rl.capability.calibration.calibration_smoke_config`), 64 items per data component and two-rung ladders. The stream keeps its 5.1 shape (3 + 16 reports of 64 items: fewer than the alarm's ``min_rollouts`` = 32 would skip every report). A plumbing check: every rule is INCONCLUSIVE and its numbers are not results. ``pilot`` / ``gate`` overrides are merged key by key.
- `combined_decision(tier2_reason: 'str', alarm_fired: 'bool') -> 'str'` — Section 1.16 as 5.1 states it: tier-2 ``accept`` and no firing gives ``accept``; tier-2 ``accept`` with a firing gives ``alarm_hold``; otherwise the tier-2 decision (pure).
- `construction_purpose_51(kind: 'str', seed: 'int') -> 'str'` — ``calib51/<kind>/<seed>`` (index 0 = training items, 1 = training-side diagnostic items).
- `descriptive_51(rows: 'Mapping[int, Mapping[str, Any]]') -> 'dict[str, Any]'` — The 5.1 descriptive quantities (no rule depends on them; pure): per kind, how many (eligible) candidates tier 2 alone accepts, how many of those the alarm holds, the first firing reports and whether each construction reached its target.
- `evaluate_rules_51(results: 'Mapping[str, Any]') -> 'dict[str, Any]'` — The 5.1 table on a results dict (pure; synthetic dicts work).
- `gate_namespace_51(seed: 'int', kind: 'str') -> 'str'` — Probe namespace of candidate ``kind``: ``calib51/<seed>/<kind>`` (sampler purpose + ``/gate/0``).
- `render_calibration51_summary(results: 'Mapping[str, Any]') -> 'str'` — ``summary.md``: banner, setup, the per-seed table, constructions, the 5.1 rules and the descriptive tier-2-only counts.
- `run_calibration_51(config: 'CalibrationAlarmConfig', output_dir: 'str | Path', *, start_path: 'str | Path | None' = None) -> 'dict[str, Any]'` — Run the 5.1 follow-up (module docstring) and write ``results.json`` / ``summary.md`` / ``calibration51_config.json`` (and ``start_model.safetensors`` when the start is rebuilt) to ``output_dir``; returns the results dict. ``start_path``: a ``.safetensors`` start checkpoint; ``None`` rebuilds it. CPU only; nothing is downloaded.
- constants: `ALARM_ITEMS_PER_REPORT`, `BASELINE_REPORTS`, `CALIB51_ORACLE_PURPOSE`, `CALIBRATION51_BANNER`, `CANDIDATE_REPORTS`, `KINDS_51`, `NULL_STREAM_KIND`, `RULES_51`

## `research_rl.capability.calibration_specificity`

Follow-up 5.2 of the gate positive control: alarm specificity on honest improvements.

- **class `CalibrationSpecificityConfig`** — Every knob of the 5.2 follow-up (JSON-able via :meth:`to_dict` / :meth:`from_dict`).
  - `from_dict(cls, data: 'Mapping[str, Any]') -> 'CalibrationSpecificityConfig'` — Inverse of :meth:`to_dict`; unknown keys raise ``ValueError``.
  - `gate_config(self) -> 'Any'` — ``GateConfig(alpha_schedule="per_decision", **gate)`` (the section-5 gate).
  - `pilot_config(self) -> 'PilotConfig'` — The ``PilotConfig`` of the start checkpoint (default + ``pilot`` overrides).
  - `preregistration_deviations(self) -> 'list[str]'` — Every setting that differs from the pre-registered 5.2 run (empty for it): start, seeds, gate, oracle size and the ``good`` construction knobs (section 5's as-built defaults, ``CAPABILITY.md`` 4.4: 5.2 builds ``good`` "exactly as in section 5"). Execution settings are not deviations.
  - `section5(self) -> 'CalibrationConfig'` — The section-5 config carrying this config's start, seeds, gate, oracle size and ``good`` construction knobs (validation, the ``pilot_config`` / ``gate_config`` builders, the ladder and the deviation check are section 5's).
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able dict (tuples as lists); :meth:`from_dict` inverts it.
- **class `SeedPlan52`** — The item plan of one seed under the 5.2 purposes: the section-5 :class:`~research_rl.capability.calibration.SeedPlan` fields (``gate`` per 5.2 kind, ``oracle``, ``training`` / ``diagnostic`` of the one construction ``good``: compositions followed by the 1:1 replay) and 5.1's ``alarm`` (``{kind: [report 0 items, ..., report 18 items]}``, one stream per candidate, no null stream).
  - `check_disjoint(self) -> 'None'` — Section 5's checks (:meth:`SeedPlan.check_disjoint`) plus 5.1's alarm checks for the two candidate streams: 19 reports of 64 distinct ``train/pairs`` items each, and no alarm uid is a training uid (replay included), an SFT uid of the start, a diagnostic, gate, oracle or start-map uid.
  - `summary(self) -> 'dict[str, Any]'` — JSON-able counts, purposes and the disjointness result.
- `alarm_purpose_52(seed: 'int', kind: 'str') -> 'str'` — Alarm-item purpose of the stream of ``kind``: ``calib52/alarm/<seed>/<kind token>``; the index is the report number (0-18).
- `build_seed_plan_52(config: 'CalibrationSpecificityConfig', sampler: 'ItemSampler', probe: 'CapabilityProbe', seed: 'int', *, replay: 'Sequence[Item]', start_uids: 'frozenset[str]') -> 'SeedPlan52'` — The 5.2 item plan of ``seed`` (module docstring, step 2), checked by :meth:`SeedPlan52.check_disjoint` before it is returned. ``replay`` is the start's SFT data, ``start_uids`` the start-map uids.
- `calibration52_smoke_config(**overrides: 'Any') -> 'CalibrationSpecificityConfig'` — The ``--smoke`` configuration: seed 0, the section-5 smoke start, gate, item counts and ``good`` ladder (:func:`~research_rl.capability.calibration.calibration_smoke_config`). The stream keeps its 5.1 shape (3 + 16 reports of 64 items). A plumbing check: every rule is INCONCLUSIVE and its numbers are not results. ``pilot`` / ``gate`` overrides are merged key by key.
- `construction_purpose_52(seed: 'int') -> 'str'` — ``calib52/good/<seed>`` (index 0 = training items, 1 = training-side diagnostic items).
- `descriptive_52(rows: 'Mapping[int, Mapping[str, Any]]') -> 'dict[str, Any]'` — The 5.2 descriptive quantities (no rule depends on them; pure): per kind the eligible seeds, the seeds whose stream fired, the per-report V1-V2 gaps and their ranges, the first firing report per seed, the tier-2 accepts and the **false holds** (tier-2 accepts the alarm turned into ``alarm_hold``: every one is a false hold of an honest candidate, eligible or not, listed both ways) and whether the ``good`` construction reached its training-side target.
- `evaluate_rules_52(results: 'Mapping[str, Any]') -> 'dict[str, Any]'` — The 5.2 table on a results dict (pure; synthetic dicts work).
- `gate_namespace_52(seed: 'int', kind: 'str') -> 'str'` — Probe namespace of candidate ``kind``: ``calib52/<seed>/<kind token>`` (``good@1.0`` -> ``good_1.0``; sampler purpose + ``/gate/0``).
- `hack_stream_gaps_51(results_51: 'Mapping[str, Any]') -> 'dict[str, Any]'` — The 5.1 hack streams' per-report V1-V2 gaps from a 5.1 ``results.json`` dict (pure; for comparison only): per kind (``hack_replay``, ``good_hack``) and seed the 19 gaps, the first firing report and the incumbent / candidate gap ranges over all seeds, and the null streams' gap range.
- `render_calibration52_summary(results: 'Mapping[str, Any]') -> 'str'` — ``summary.md``: banner, setup, the per-seed table (with the per-report gaps), the construction rungs, the 5.2 rules and the descriptive quantities (false holds, first firing reports, the 5.1 hack streams' gaps for comparison).
- `run_calibration_52(config: 'CalibrationSpecificityConfig', output_dir: 'str | Path', *, start_path: 'str | Path | None' = None, comparison_path: 'str | Path | None' = None) -> 'dict[str, Any]'` — Run the 5.2 follow-up (module docstring) and write ``results.json`` / ``summary.md`` / ``calibration52_config.json`` (and ``start_model.safetensors`` when the start is rebuilt) to ``output_dir``; returns the results dict. ``start_path``: a ``.safetensors`` start checkpoint; ``None`` rebuilds it. ``comparison_path``: the 5.1 ``results.json`` whose hack-stream gaps are reported for comparison (``None``: :data:`CALIBRATION51_RESULTS` in this repository; a missing file is reported, not an error). CPU only; nothing is downloaded.
- constants: `CALIB52_ORACLE_PURPOSE`, `CALIBRATION51_RESULTS`, `CALIBRATION52_BANNER`, `KINDS_52`, `LAMBDAS_52`, `RULES_52`

## `research_rl.capability.edits`

Declarative FFN-channel edits and an immutable, gate-checked version graph (``CAPABILITY.md`` 1.12).

- **class `CommitRefusedError`** — ``commit`` without a passing gate verdict and without an override reason, or out of scope.
- **class `EditError`** — An edit (or delta) is malformed or does not fit the model / store. Raised before any write.
- **class `EditOp`** — The closed set of edit operations (see the module docstring for their exact semantics).
  - members: `ABLATE`, `REVERT`, `SCALE`, `INTERPOLATE`, `APPLY_DELTA`
- **class `EditRecord`** — One declarative edit. Validated and normalised on construction (:class:`EditError`).
  - `channel_selection(self) -> 'ChannelSelection'` — The selection as a :class:`ChannelSelection`.
  - `copy(self) -> 'EditRecord'` — A re-validated, independent copy (``from_dict(to_dict())``): :class:`EditError` if the record's dicts were mutated into an invalid or non-normalised value.
  - `from_dict(cls, data: 'Any') -> 'EditRecord'` — Inverse of :meth:`to_dict`. ``op``, ``selection`` and ``params`` are required; unknown keys raise :class:`EditError`.
  - `from_json(cls, text: 'str') -> 'EditRecord'` — Inverse of :meth:`to_json` (:class:`EditError` on invalid JSON or content).
  - `references(self) -> 'tuple[str | None, str | None]'` — ``(version id, delta id)`` this edit reads (``None`` where not applicable).
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-shaped dict (string layer keys).
  - `to_json(self) -> 'str'` — Canonical JSON (sorted keys, no whitespace) of :meth:`to_dict`.
- **class `StagedEdit`** — Edits applied to the live model on top of ``parent``, not yet committed.
  - `last_verdict` — The verdict of the most recent evaluation, or ``None``.
  - `passed` — True iff the last evaluation passed transfer and retention without an alarm.
- **class `StoreError`** — The store's files are missing, unsafe, corrupted, or do not fit the model.
- **class `Version`** — An immutable version record (one JSON file ``versions/<id>.json``).
  - `build(cls, **fields_: 'Any') -> 'Version'` — Construct a version and compute its id from the other fields.
  - `compute_id(record: 'Mapping[str, Any]') -> 'str'` — SHA-256 hex of the canonical JSON of ``record`` (a JSON-shaped dict without ``id``).
  - `evaluation_pin(self) -> 'tuple[int, float, str] | None'` — ``(evaluation_index, alpha, alpha_schedule)`` of the recorded evaluation, or ``None`` without a verdict. :class:`EditError` if the context is malformed or its index differs from the verdict's ``decision_index``.
  - `from_dict(cls, data: 'Any') -> 'Version'` — Strict inverse of :meth:`to_dict`: the id must equal the hash of the stored content and the content must already be in normal form (``from_dict(d).to_dict() == d``).
  - `passed` — Whether the recorded verdict passed (transfer and retention, no alarm).
  - `record(self) -> 'dict[str, Any]'` — JSON-shaped dict of every field except ``id`` (the hashed content).
  - `to_dict(self) -> 'dict[str, Any]'` — The JSON record as stored in ``versions/<id>.json``.
- **class `VersionStore`** — Immutable version graph of a model's FFN channel state, persisted under ``root``.
  - `add_delta(self, delta: 'ChannelState') -> 'str'` — Store a delta (per layer: ``channels`` and floating ``gate``/``up``/``down`` (+ biases) increments, shaped like :class:`LayerChannels`); returns its content id. ``module_path`` ``None`` is taken as the model's path. Idempotent for identical content.
  - `commit(self, staged: 'StagedEdit', *, override: 'str | None' = None, note: 'str' = '') -> 'Version'` — Record the staged edit as a new version (child of its parent) and make it the head.
  - `delta_between(self, a: 'str', b: 'str', selection: 'Mapping[int, Sequence[int]] | None' = None) -> 'str'` — Store ``state(b) - state(a)`` on ``selection`` (default: :meth:`diff` of ``a``, ``b``), computed in float32 (float64 for float64 parameters); returns the delta id.
  - `deltas(self) -> 'list[str]'` — Ids of every stored delta, sorted.
  - `diff(self, a: 'str', b: 'str') -> 'dict[int, list[int]]'` — Per layer, the channels whose stored slices differ between versions ``a`` and ``b`` (any element ``!=``; ``-0.0 == 0.0``, and a NaN at the same place in both counts as changed). Layers without a changed channel are omitted.
  - `discard(self, staged: 'StagedEdit') -> 'None'` — Undo a staged edit: the affected parameters are restored bitwise to the parent's.
  - `evaluate(self, staged: 'StagedEdit', gate_config: 'GateConfig', probe: 'CapabilityProbe') -> 'GateVerdict'` — Gate verdict of the staged weights (candidate) against the parent's (incumbent).
  - `evaluation_count` — Evaluations run by this store so far (the next evaluation's ``decision_index``).
  - `gate_namespace` — Sampler namespace of this store's gate items, ``"edit/<gate_nonce>"`` (the nonce is drawn at random when the store is created and kept in ``store.json``), so two stores never share an item stream; ``None`` for a legacy store created without a nonce (its items use the training-run purpose ``gate/0``).
  - `get(self, version_id: 'str') -> 'Version'` — A copy of version ``version_id`` (:class:`EditError` for a malformed id, ``KeyError`` if unknown). Mutating the copy's dicts never changes the store.
  - `head` — Id of the version the live model was last put at by this store (``None`` before init).
  - `history(self, version_id: 'str | None' = None) -> 'list[Version]'` — Copies of the lineage root -> ``version_id`` (default: the head; ``[]`` for an empty store).
  - `init_from_model(self, note: 'str' = '') -> 'Version'` — Record the live model as the root version and make it the head. Only on an empty store.
  - `ledger` — A copy of the store's own alpha ledger (``None`` before the first evaluation).
  - `load_delta(self, delta_id: 'str') -> 'ChannelState'` — A stored delta (validated against the model).
  - `load_state(self, version_id: 'str') -> 'ChannelState'` — The stored channel state of a version (validated against the model).
  - `restore(self, version_id: 'str') -> 'Version'` — Put the live model at ``version_id`` bitwise and make it the head.
  - `stage(self, edit: 'EditRecord | Sequence[EditRecord]') -> 'StagedEdit'` — Apply one edit (or several, in order) to the live model on top of the head version.
  - `staged` — The active staged edit, if any.
  - `versions(self) -> 'list[Version]'` — Copies of every version, ordered by ``(created, id)``.
- constants: `STORE_FORMAT`

## `research_rl.capability.es`

Forward-only evolution-strategies update operator (``CAPABILITY.md`` section 1.7).

- **class `ESConfig`** — Hyper-parameters of :class:`ESOperator` (JSON-able via :meth:`to_dict`).
  - `from_dict(cls, d: 'Mapping[str, Any]') -> 'ESConfig'`
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `ESError`** — Invalid ES configuration, fitness result or state.
- **class `ESOperator`** — Evolution-strategies :class:`~research_rl.capability.types.UpdateOperator` over a space.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Load counters after checking that the config and ``dim`` match exactly (a different seed, population, etc. would silently change the perturbation stream), then re-base the space from the model's current weights (load the model weights before calling this).
  - `noise(self, update_index: 'int', noise_index: 'int') -> 'torch.Tensor'` — Unsigned standard-normal draw ``eps`` (CPU float32, length ``dim``) for ``(t, j)``.
  - `noise_index(self, member_index: 'int') -> 'tuple[int, float]'` — ``(noise index j, sign)`` of member ``member_index``.
  - `on_decision(self, decision: 'GateDecision') -> 'None'` — Re-base the space from the model's current weights (every decision; never by subtracting regenerated noise). The caller has already restored / shrunk / reverted.
  - `parameter_names(self) -> 'list[str]'`
  - `perturbation(self, update_index: 'int', member_index: 'int') -> 'torch.Tensor'` — Signed unit-scale perturbation of member ``i`` (the space receives ``sigma`` times it).
  - `state_dict(self) -> 'dict[str, Any]'` — JSON-able counters and config. Weights (and the space's base) are not included: they are part of the model / the gate's snapshots. Perturbations depend only on ``(seed, update_index, member_index)``, so no RNG state is needed.
  - `step(self, model: 'nn.Module', update_index: 'int') -> 'OperatorReport'` — One ES update in place on ``model`` (see the module docstring).
- **class `FitnessResult`** — Outcome of evaluating one ES member on the update's training items.
  - `forward_passes` — Generation calls of this evaluation: ``1 + extra.get("extra_forward_passes", 0)``.
- `shape_fitness(fitness: 'Sequence[float]', method: 'FitnessShaping') -> 'list[float]'` — Shaped fitness values (see the module docstring for the exact formulas).
- constants: `FitnessFn`, `FitnessShaping`

## `research_rl.capability.gate`

The pure gate verdict: paired statistics on one :class:`GateEvidence`, no weight changes.

- **class `CapabilityGate`** — Evaluates :class:`GateEvidence` into a :class:`GateVerdict` (statistics only).
  - `accepts(self, verdict: 'GateVerdict') -> 'bool'` — Whether ``verdict`` licenses promoting the candidate under this config's statistics.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Restore ledger position and verdict count; atomic (nothing changes on any error).
  - `state_dict(self) -> 'dict[str, Any]'`
  - `verdict(self, evidence: 'GateEvidence') -> 'GateVerdict'` — Test one decision's evidence; consumes exactly one alpha from the ledger.
  - `verdict_count` — Number of verdicts computed so far (each consumed one alpha).
- **class `GateConfig`** — Configuration of the capability gate and of the gated loop around it (JSON-able).
  - `from_dict(cls, data: 'Mapping[str, Any]') -> 'GateConfig'` — Inverse of :meth:`to_dict`. Unknown keys raise ``ValueError``; missing keys take defaults.
  - `make_ledger(self) -> 'AlphaLedger'` — A fresh :class:`AlphaLedger` with this config's ``alpha`` and ``alpha_schedule``.
  - `margin_for(self, slice_name: 'str') -> 'float'` — Retention margin of ``slice_name`` (override if present, else the default margin).
  - `retention_horizon(self, slice_name: 'str | None' = None, *, n: 'int | None' = None, n10: 'int' = 0, n01: 'int' = 0) -> 'int | None'` — How many verdicts a binary retention slice with this evidence can keep passing.
  - `to_dict(self) -> 'dict[str, Any]'` — Plain JSON-able dict (enum as its value, tuples as lists).

## `research_rl.capability.loop`

The gated training loop (``docs/research_rl/CAPABILITY.md`` 1.9 with the pinned semantics of 1.16).

- **class `GateController`** — One gate procedure per :meth:`run` (see the module docstring for the exact semantics).
  - `check_group_scope(self, model: 'nn.Module', incumbent: 'StateSnapshot') -> 'None'` — Raise :class:`~research_rl.capability.weights.SnapshotError` (a ``ValueError``) unless every channel group can be ranked and reverted against ``incumbent`` on ``model``.
  - `config`
  - `group_stage_enabled`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Restore a :meth:`state_dict`; ``ValueError`` on any mismatch (nothing changes then).
  - `max_verdicts_per_decision` — Worst-case verdicts of one decision: 1 when disabled, else ``1 + len(shrink_lambdas)`` plus ``min(max_group_rollbacks, len(channel_groups))`` when the group stage is enabled.
  - `run(self, model: 'nn.Module', operator: 'UpdateOperator', incumbent: 'StateSnapshot', decision_index: 'int', update_index: 'int', alarm: 'bool | AlarmState', reward_v1: 'float | None', reward_v2: 'float | None' = None) -> 'tuple[DecisionRecord, StateSnapshot]'` — Run one gate procedure on ``model``'s current weights; returns the record and the new incumbent (the same object unless the decision promoted the weights).
  - `state_dict(self) -> 'dict[str, Any]'` — JSON-able: hold streak, the verdict gate's state (ledger position, verdict count), ``enabled``, the rollback unit and the group names (checked on load).
  - `verdict_gate` — The gate that computes (and spends alpha for) every verdict: ``gate`` itself when enabled, a private copy with its own ledger when disabled.
- **class `GatedLoop`** — Operator updates, alarm, gate decisions, decision log and resume state.
  - `check_horizon(self, num_updates: 'int') -> 'None'` — Start-of-run check: ``verdict_gate.ledger.count + required_verdicts(num_updates)`` must not exceed :meth:`horizon`. Otherwise ``ValueError`` (``UserWarning`` with ``allow_short_horizon=True``). Run by :meth:`run` before the first update.
  - `decision_pending` — True when the running block is owed a decision that has not been recorded: it holds ``block_updates`` updates, or its latched alarm fired with ``alarm_triggers_decision``. This happens only when a decision raised (see :meth:`run`); the next :meth:`run` takes that decision before its first update.
  - `horizon(self) -> 'int | None'` — Last verdict index (1-based, ledger count) at which every retention slice of the probe can still pass with ``n10 = round(0.005 n)`` incumbent-only wins and no candidate-only win (``GateConfig.retention_horizon`` per slice, ``n = probe.items_per_retention_slice``, the size the probe actually tests, falling back to ``config.items_per_retention_slice``); ``None`` = no practical limit.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Continue from a :meth:`state_dict` (build every object exactly as the saved run did).
  - `required_verdicts(self, num_updates: 'int') -> 'int'` — :func:`worst_case_verdicts` of ``num_updates`` more updates from the current position (with a :attr:`decision_pending`: that decision's ``max_verdicts_per_decision`` plus the worst case of ``num_updates`` updates from an empty block).
  - `run(self, num_updates: 'int') -> 'list[DecisionRecord]'` — Run ``num_updates`` more operator updates (with their decisions); returns the decision records of this call (also appended to :attr:`records`). The horizon check runs first.
  - `state_dict(self) -> 'dict[str, Any]'` — Everything needed to continue the run exactly; loadable with ``torch.save`` / ``torch.load(weights_only=True)`` (tensors, numbers, strings, lists, dicts).
  - `updates_since_decision`
- **class `MetricsRecorder`** — Anything with ``log_metrics(step, logs)``, e.g. :class:`~research_rl.core.experiment.ExperimentRecorder`.
  - `log_metrics(self, step: 'int', logs: 'Mapping[str, Any]') -> 'None'`
- `classify_verdict(verdict: 'GateVerdict', *, require_transfer_gain: 'bool') -> 'str'` — The first-verdict reason of the module docstring's table (one of :data:`WOULD_REASONS`).
- `retention_unpassable(verdict: 'GateVerdict') -> 'bool'` — True iff some **failing** retention slice of ``verdict`` could not have passed at the verdict's alpha even with zero discordant pairs.
- `snapshot_from_state(state: 'Mapping[str, Any]') -> 'StateSnapshot'` — Inverse of :func:`snapshot_to_state`; the fingerprint is re-verified (:class:`~research_rl.capability.weights.SnapshotError` on mismatch).
- `snapshot_to_state(snapshot: 'StateSnapshot') -> 'dict[str, Any]'` — A :class:`StateSnapshot` as a plain dict loadable with ``torch.load(weights_only=True)``.
- `worst_case_verdicts(num_updates: 'int', *, block_updates: 'int', verdicts_per_decision: 'int', alarm_triggers_decision: 'bool', since: 'int' = 0) -> 'int'` — Upper bound on the verdicts of ``num_updates`` more updates of a :class:`GatedLoop`.
- constants: `OVERRIDE_GATE_DISABLED`, `REASON_ACCEPT`, `REASON_ALARM_HOLD`, `REASON_CONFIRMED_ALARM`, `REASON_HOLD_LIMIT`, `REASON_HORIZON_EXHAUSTED`, `REASON_NO_TRANSFER_GAIN`, `REASON_RETENTION_FAILED`, `WOULD_PREFIX`, `WOULD_REASONS`

## `research_rl.capability.operators`

Backprop update operators for the gated loop (``docs/research_rl/CAPABILITY.md`` 1.9 / 1.16).

- **class `Rollouts`** — The sampled and scored rollouts of one update (rows prompt-major).
  - `lengths` — Completion tokens before the first EOS, per row (all action tokens without EOS).
- **class `TorchGRPOOperator`** — Minimal on-policy GRPO :class:`~research_rl.capability.types.UpdateOperator` (module docstring for the exact semantics).
  - `advantages(self, rollouts: 'Rollouts') -> 'tuple[list[float], list[bool]]'` — ``(advantages, keep)`` of the rollouts (step 5; ``keep`` all True when ``screen`` is off).
  - `config(self) -> 'dict[str, Any]'` — The hyper-parameters as a JSON-able dict (checked by :meth:`load_state_dict`).
  - `items(self, update_index: 'int') -> 'list[Item]'` — The update's prompts: ``sampler.draw(Split.TRAIN, prompts_per_update, purpose=TRAIN_PURPOSE, index=update_index)`` (truncation contract checked).
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Restore the counters after checking that the configuration matches exactly (``ValueError`` otherwise; nothing changes then).
  - `on_decision(self, decision: 'GateDecision') -> 'None'` — No-op (GRPO keeps no weight copy of its own; the controller restores weights and optimizer state itself). Validates the type.
  - `parameter_names(self) -> 'list[str]'` — Model names of the optimizer's parameters, in ``named_parameters`` order.
  - `policy_loss(self, model: 'nn.Module', rollouts: 'Rollouts', advantages: 'Sequence[float]') -> 'torch.Tensor'` — The token-level policy-gradient loss of step 6 (a scalar with a graph). The forward runs under ``fork_rng`` seeded with :func:`loss_seed` ``(seed, rollouts.update_index)``.
  - `rollouts(self, model: 'nn.Module', update_index: 'int') -> 'Rollouts'` — Sample and score the rollouts of update ``update_index`` (steps 1-4 of the module docstring) without changing any weight. Deterministic in ``(seed, update_index)`` and the current weights; the global RNG state is unchanged.
  - `state_dict(self) -> 'dict[str, Any]'` — Configuration and counters (JSON-able). Optimizer state is not included (see the module docstring).
  - `step(self, model: 'nn.Module', update_index: 'int') -> 'OperatorReport'` — One GRPO update in place (module docstring); returns its report.
- `group_advantages(rewards: 'Sequence[float]', group_size: 'int', keep: 'Sequence[bool] | None' = None) -> 'list[float]'` — Group-normalised advantages (see the module docstring, step 5).
- `loss_seed(seed: 'int', update_index: 'int') -> 'int'` — Seed of the training forward of update ``update_index`` (its dropout masks): the first 8 bytes (little endian, top bit cleared) of ``SHA-256("research_rl.capability.grpo.loss:{seed}: {update_index}")``; a different domain from :func:`sample_seed`, so the two streams differ.
- `sample_seed(seed: 'int', update_index: 'int') -> 'int'` — Sampling seed of update ``update_index``: the first 8 bytes (little endian, top bit cleared) of ``SHA-256("research_rl.capability.grpo:{seed}:{update_index}")``.
- constants: `ADVANTAGE_EPS`, `TRAIN_PURPOSE`

## `research_rl.capability.pilot`

P0 capability pilot on CPU (``docs/research_rl/CAPABILITY.md`` 1.14 and the kill criteria of section 3).

- **class `ArmSpec`** — One arm: ``operator``, ``gate``, ``variant`` and ``seed``.
  - `file_stem`
  - `key`
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `PilotConfig`** — Every knob of the pilot (JSON-able via :meth:`to_dict` / :meth:`from_dict`).
  - `base_gate_config(self, *, items_per_retention_slice: 'int | None' = None) -> 'GateConfig'` — ``GateConfig`` with the overrides, ``rollback_unit=parameters`` and (if given) the retention item count; ``enabled`` is left at its default (set per arm).
  - `es_num_updates`
  - `from_dict(cls, data: 'Mapping[str, Any]') -> 'PilotConfig'` — Inverse of :meth:`to_dict`; unknown keys raise ``ValueError``.
  - `to_dict(self) -> 'dict[str, Any]'`
  - `updates_for(self, operator: 'str') -> 'int'` — Updates of an arm with ``operator``.
- `arm_specs(config: 'PilotConfig') -> 'list[ArmSpec]'` — Every arm of ``config`` in canonical order (variant, seed, operator, gate).
- `baseline_standard_error(p: 'float', n: 'int') -> 'float'` — Conservative standard error of a start-map mean ``p`` of ``n`` per-item values in ``[0, 1]``.
- `choose_retention_items(config: 'PilotConfig', retention_slices: 'Sequence[str]') -> 'dict[str, Any]'` — Pick ``items_per_retention_slice`` so that :class:`~research_rl.capability.loop.GatedLoop`'s start-of-run horizon check passes for the longest arm.
- `evaluate_criteria(results: 'Mapping[str, Any]') -> 'dict[str, Any]'` — Section-3 kill criteria on a results dict (pure; see :data:`CRITERIA_DEFINITIONS`).
- `gate_exercise(arms: 'Sequence[Mapping[str, Any]]', min_improved_blocks: 'int') -> 'dict[str, Any]'` — Whether the run exercised the gate enough for its error rates to mean anything.
- `interval_with_baseline_error(interval: 'Mapping[str, Any]', se_baseline: 'float', alpha: 'float' = 0.05, *, baseline_n: 'int | None' = None) -> 'dict[str, Any]'` — Widen a :func:`paired_interval` of ``value - baseline`` differences for a baseline that is one noisy measurement shared by every seed (so its error is common to all differences and the across-seed spread cannot see it).
- `oracle_exclusion_steps(config: 'PilotConfig', num_updates: 'int | None' = None) -> 'int'` — Highest tracker step whose items the oracle excludes, the same for every arm of ``config``.
- `oracle_items(sampler: 'ItemSampler', n: 'int', decision_index: 'int', exclude: 'set[str]', *, purpose: 'str' = 'oracle', slices: 'Sequence[str] | None' = None, pool: 'str | None' = 'track') -> 'dict[str, list[Item]]'` — ``n`` oracle items per T_RULE slice for one decision: purpose :data:`ORACLE_PURPOSE`, index = ``decision_index``, from the tracker pool (so disjoint from every gate item by construction), without any uid in ``exclude`` (start-map and tracker items). Deterministic in the key.
- `paired_interval(diffs: 'Sequence[float]', alpha: 'float' = 0.05) -> 'dict[str, Any]'` — Two-sided ``1 - alpha`` Student-t interval of the mean of paired differences.
- `render_summary(results: 'Mapping[str, Any]') -> 'str'` — ``summary.md``: banner, warnings, start map, one table per arm, gate error rates, criteria.
- `run_pilot(config: 'PilotConfig', output_dir: 'str | Path') -> 'dict[str, Any]'` — Run the P0 pilot (module docstring) and write its outputs to ``output_dir``; returns the results dict (also written as ``results.json``). Never downloads anything; CPU only.
- `smoke_config(**overrides: 'Any') -> 'PilotConfig'` — The ``--smoke`` configuration: 1 seed, 2 arms (``grpo+gate_on``, ``es_informed+gate_off``) on the leak variant, 4 updates in blocks of 2, a 2-layer hidden-32 model, tiny item counts (16 retention items per slice with a 0.5 retention margin, so verdicts can pass at all) and ``allow_short_horizon=True`` (a short horizon only warns).
- constants: `BANNER`, `GATES`, `OPERATORS`, `ORACLE_PURPOSE`, `VARIANTS`, `VERDICT_INCONCLUSIVE`, `VERDICT_KILL`, `VERDICT_SUCCESS`

## `research_rl.capability.probe`

Gate evidence and start-of-run capability maps (``docs/research_rl/CAPABILITY.md`` 1.8).

- **class `CapabilityProbe`** — Builds :class:`GateEvidence` for one gate evaluation.
  - `evidence(self, model: 'nn.Module', incumbent: 'StateSnapshot', decision_index: 'int', *, attempt: 'int' = 0, alarm: 'bool' = False, namespace: 'str | None' = None) -> 'GateEvidence'` — Paired evidence of the candidate (current weights) against ``incumbent``.
  - `from_config(cls, sampler: 'ItemSampler', score_fn: 'ScoreFn', config: 'GateConfig', *, transfer_slices: 'Sequence[str] | None' = None, retention_slices: 'Sequence[str] | None' = None) -> 'CapabilityProbe'` — Probe with ``config.items_per_transfer_slice`` / ``config.items_per_retention_slice``.
  - `gate_items(self, decision_index: 'int', attempt: 'int' = 0, *, namespace: 'str | None' = None) -> 'dict[str, list[Item]]'` — The items of evaluation ``(decision_index, attempt)``, per slice (transfer slices first, then retention, in construction order): :func:`draw_slice` with ``purpose=gate_purpose(attempt, namespace)``, ``index=decision_index`` and ``pool=GATE_POOL`` (so they never coincide with tracker or start-map items). Deterministic; a pure function of the key.
  - `items_per(self, slice_name: 'str') -> 'int'` — Items per evaluation of one of this probe's slices (``KeyError`` for other slices).
  - `score_slice(self, model: 'nn.Module', slice: 'str', n: 'int', *, purpose: 'str', index: 'int', pool: 'str | None' = None) -> 'SliceScores'` — :func:`score_slice` with this probe's sampler and score function. ``slice`` may be any non-AUDIT slice of the sampler (TRAIN included), not only this probe's slices.
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able configuration (slices, item counts, sampler identity).
- **class `SliceScores`** — Scores of one model state on the items of one slice drawn for ``(purpose, index)`` (from ``pool``, ``None`` = unpartitioned; see :func:`draw_slice`).
  - `mean` — Mean score (pass rate for binary scores); ``nan`` for no items.
  - `n`
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able: slice, split value, purpose, index, pool, n, mean, uids, scores (items by uid only).
  - `uids`
- **class `SliceStart`** — Start-checkpoint measurement of one slice.
  - `from_dict(cls, data: 'Mapping[str, Any]') -> 'SliceStart'` — Inverse of :meth:`to_dict` (derived fields are recomputed, not read).
  - `n`
  - `pass1`
  - `pass_k`
  - `sampled_mean`
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `StartMap`** — pass@1 (greedy) and pass@k (sampled) of the start checkpoint, per slice (JSON-able).
  - `from_dict(cls, data: 'Mapping[str, Any]') -> 'StartMap'`
  - `measure(cls, model: 'nn.Module', sampler: 'ItemSampler', score_fn: 'Any', *, items_per_slice: 'int', slices: 'Sequence[str] | None' = None, k: 'int' = 16, num_samples: 'int | None' = None, temperature: 'float' = 1.0, seed: 'int' = 0, sample_batch_size: 'int' = 16) -> 'StartMap'` — Measure the start map of ``model`` (its current weights).
  - `pass1(self, slice_name: 'str') -> 'float'` — Greedy pass rate of ``slice_name`` (``KeyError`` if not measured).
  - `pass_k(self, slice_name: 'str') -> 'float'` — Unbiased pass@k of ``slice_name`` (``KeyError`` if not measured).
  - `split_mean_pass1(self, split: 'Split') -> 'float'` — Unweighted mean over the measured slices of ``split`` of their greedy pass rates (``ValueError`` if no slice of that split was measured).
  - `summary(self) -> 'dict[str, dict[str, float]]'` — ``{slice: {"pass1": ..., "pass_k": ..., "sampled_mean": ...}}``.
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able (``json.dumps`` works); :meth:`from_dict` inverts it.
- `check_truncation_budget(sampler: 'ItemSampler', score_fn: 'Any') -> 'None'` — Raise ``ValueError`` if ``score_fn`` exposes an integer ``max_new_tokens`` smaller than ``sampler.splits.operators.max_len + 1`` (the longest possible answer plus the EOS token); no-op for score functions without ``max_new_tokens``.
- `draw_slice(sampler: 'ItemSampler', slice_name: 'str', n: 'int', *, purpose: 'str', index: 'int', pool: 'str | None' = None) -> 'list[Item]'` — ``n`` >= 1 distinct items of a non-AUDIT slice for the key ``(purpose, index)``.
- `gate_purpose(attempt: 'int', namespace: 'str | None' = None) -> 'str'` — Sampler purpose of gate evaluation ``attempt`` (0 = the raw candidate, then one per shrink factor / group revert of the same decision): ``f"gate/{attempt}"``, or ``f"{namespace}/gate/{attempt}"`` with a ``namespace`` (an independent item stream per namespace; :class:`~research_rl.capability.edits.VersionStore` uses one per store).
- `item_pool(sampler: 'ItemSampler', uid: 'str') -> 'str'` — The pool of item ``uid`` for ``sampler``: :data:`TRACK_POOL` iff ``_derive_seed("item_pool", sampler.root_seed, uid) % TRACK_POOL_DENOMINATOR == 0``, else :data:`GATE_POOL`. A fixed function of ``(root_seed, uid)``; every uid is in exactly one pool.
- `sample_completions(model: 'nn.Module', tokenizer: 'Any', prompt: 'str', *, num_samples: 'int', max_new_tokens: 'int', temperature: 'float' = 1.0, seed: 'int', batch_size: 'int' = 16) -> 'list[str]'` — ``num_samples`` sampled continuations of one ``prompt`` (decoded, EOS and after removed).
- `sample_completions_with_truncation(model: 'nn.Module', tokenizer: 'Any', prompt: 'str', *, num_samples: 'int', max_new_tokens: 'int', temperature: 'float' = 1.0, seed: 'int', batch_size: 'int' = 16) -> 'tuple[list[str], list[bool]]'` — :func:`sample_completions` plus one exact truncation flag per sample.
- `score_items(score_fn: 'ScoreFn', model: 'nn.Module', items: 'Sequence[Item]') -> 'list[float]'` — ``score_fn(model, items)`` with validation: one finite score in [0, 1] per item, every item within the truncation budget (``len(answer) + 1 <= score_fn.max_new_tokens`` when the score function exposes it), training mode restored afterwards. ``[]`` for no items (no call).
- `score_slice(sampler: 'ItemSampler', score_fn: 'ScoreFn', model: 'nn.Module', slice_name: 'str', n: 'int', *, purpose: 'str', index: 'int', pool: 'str | None' = None) -> 'SliceScores'` — Draw ``n`` items of ``slice_name`` for ``(purpose, index)`` from ``pool`` (:func:`draw_slice`; ``None`` = unpartitioned, may coincide with gate and tracker items) and score the model's **current** weights on them (:func:`score_items`).
- constants: `GATE_POOL`, `ITEM_POOLS`, `START_PURPOSE`, `TRACK_POOL`, `TRACK_POOL_DENOMINATOR`

## `research_rl.capability.scores`

Forward-only channel scoring and grouping (``docs/research_rl/CAPABILITY.md`` section 1.6).

- **class `ChannelStats`** — Per-layer, per-channel statistics of the SwiGLU ``down_proj`` input over real tokens.
  - `from_dict(cls, d: 'Mapping[str, Any]') -> 'ChannelStats'` — Inverse of :meth:`to_dict` (float64 values round-trip exactly through JSON).
  - `intermediate_size(self, layer: 'int') -> 'int'`
  - `layers(self) -> 'list[int]'` — Layer indices, ascending.
  - `merge(self, other: 'ChannelStats') -> 'ChannelStats'` — Statistics of the union of both text sets: means weighted by token counts (float64).
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able dict (string layer keys, Python ints / float lists); see :meth:`from_dict`.
- `collect_channel_stats(model: 'nn.Module', tokenizer: 'Any', texts: 'Sequence[str]', *, layers: 'Iterable[int] | None' = None, batch_size: 'int' = 8, threshold: 'float' = 0.0) -> 'ChannelStats'` — Run ``texts`` through ``model`` and collect :class:`ChannelStats` of every ``down_proj`` input.
- `contrast_scores(stats_correct: 'ChannelStats', stats_incorrect: 'ChannelStats') -> 'dict[int, torch.Tensor]'` — ``mean|h|`` on V2-correct rollouts minus ``mean|h|`` on V2-incorrect rollouts, per channel.
- `density_matched_random(selection: 'ChannelSelection', model: 'nn.Module', seed: 'int') -> 'ChannelSelection'` — A uniformly random selection with exactly ``selection``'s per-layer channel counts.
- `group_channels(scores: 'Mapping[int, torch.Tensor]', *, bands: 'int' = 4, layers: 'Iterable[int] | None' = None) -> 'list[ChannelGroup]'` — Partition every channel of the scored layers into ``layer x score-quantile band`` groups.
- `layer_prior(scores: 'Mapping[int, torch.Tensor]', weights: 'Mapping[int, float]') -> 'dict[int, torch.Tensor]'` — ``scores[layer] * weights[layer]`` (float64); layers absent from ``weights`` keep weight 1.0.
- `random_scores(model: 'nn.Module', seed: 'int', *, layers: 'Iterable[int] | None' = None) -> 'dict[int, torch.Tensor]'` — Uniform ``[0, 1)`` float64 scores per channel from ``torch.Generator().manual_seed(seed)``.
- `usage_ratio_scores(stats_train: 'ChannelStats', stats_retention: 'ChannelStats', eps: 'float' = 0.001) -> 'dict[int, torch.Tensor]'` — Activation-rate ratio ``active_rate_train / (active_rate_retention + eps)`` per channel.
- `wanda_scores(model: 'nn.Module', stats: 'ChannelStats') -> 'dict[int, torch.Tensor]'` — Wanda-style channel importance ``||W_down[:, c]||_2 * sqrt(mean h_c^2)`` per layer of ``stats``.

## `research_rl.capability.search`

Subset search with capability as the objective (``docs/research_rl/CAPABILITY.md`` 1.11).

- **class `BurstRecord`** — One training burst from the start weights.
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `FinalResult`** — A finalist or control over the final item sets: per-repeat objectives and their means.
  - `mean_gain`
  - `mean_objective`
  - `mean_worst_drop`
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `SearchEvaluation`** — One scoring of trained weights on one item set ``(purpose, index)``.
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `SearchReport`** — Everything one :meth:`SubsetSearch.run` did (JSON-able via :meth:`to_dict`; no timestamps, so equal runs give equal reports).
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `SliceResult`** — Start and trained-candidate mean scores of one slice in one evaluation.
  - `difference` — ``candidate_mean - start_mean`` (positive = the burst helped).
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `SubsetSearch`** — Successive-halving search over channel subsets (module docstring for exact semantics).
  - `config(self) -> 'dict[str, Any]'` — JSON-able settings (the probe's slices and sampler identity included).
  - `run(self, model: 'nn.Module', *, update_offset: 'int' = 0) -> 'SearchReport'` — Run the search on ``model`` and return the :class:`SearchReport`.
  - `survivors_after(self, n: 'int') -> 'int'` — Candidates kept from a round of ``n``: ``max(1, ceil(survivors_fraction * n))``.
- `burst_seed(seed: 'int') -> 'int'` — Seed of the global torch RNG inside every burst of a search with ``seed`` (a 63-bit int from SHA-256 of ``("research_rl.capability.search", "burst", seed)``).
- `combine_groups(groups: 'Sequence[ChannelGroup]', *, name: 'str | None' = None) -> 'ChannelGroup'` — One candidate made of the union of ``groups`` (``ValueError`` for no groups). The default name joins the group names with ``"+"``.
- `control_seed(seed: 'int') -> 'int'` — Seed passed to :func:`~research_rl.capability.scores.density_matched_random` for the ``"random"`` control of a search with ``seed``.
- `final_purpose(repeat: 'int') -> 'str'` — Sampler purpose of final repeat ``repeat`` (shared by finalists and controls): ``f"search/final/{repeat}"``.
- `objective(gain: 'float', worst_drop: 'float', *, penalty: 'float', margin: 'float') -> 'float'` — ``gain - penalty * max(0, worst_drop - margin)``.
- `score_seed(seed: 'int', purpose: 'str', index: 'int', slice_name: 'str') -> 'int'` — Seed of the global torch RNG inside every ``score_fn`` call on slice ``slice_name`` of the item set ``(purpose, index)`` of a search with ``seed`` (a 63-bit int from SHA-256 of ``("research_rl.capability.search", "score", seed, purpose, index, slice_name)``).
- `search_purpose(round_index: 'int', candidate_index: 'int') -> 'str'` — Sampler purpose of candidate ``candidate_index`` in round ``round_index``: ``f"search/{round_index}/{candidate_index}"``.
- `whole_layer_selection(selection: 'ChannelSelection', model: 'nn.Module') -> 'ChannelSelection'` — ``selection.count()`` channels as whole layers plus at most one contiguous partial block.
- constants: `CONTROL_KINDS`, `DEFAULT_MARGIN`, `DEFAULT_PENALTY`, `MakeOperator`, `SEARCH_PURPOSE`, `STATEMENT`

## `research_rl.capability.spaces`

Flat coordinate systems over model parameters for forward-only operators (``CAPABILITY.md`` 1.7).

- **class `ChannelGainSpace`** — P1: one multiplicative gain per selected channel (``dim == selection.count()``).
- **class `ChannelWeightSpace`** — P2: every element owned by the selected channels, ``W = base + v``.
- **class `FullParameterSpace`** — Every element of the named parameters, ``W = base + v``.
- **class `SpaceError`** — Invalid space construction or an invalid vector passed to ``write``.

## `research_rl.capability.stats`

Paired statistics for the capability gate (pure functions; standard library only).

- **class `AlphaLedger`** — Hands out the significance level of each gate decision.
  - `alpha_at(self, k: 'int') -> 'float'` — Alpha of the k-th decision (1-based) without consuming it.
  - `count` — Number of alphas handed out so far.
  - `expected_spent(self, count: 'int') -> 'float'` — ``sum_{k=1}^{count} alpha_at(k)``, accumulated in the same order as :meth:`next_alpha` (so it reproduces ``spent`` of a ledger that handed out ``count`` alphas bit for bit).
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Restore the position; atomic (nothing changes unless every check passes).
  - `next_alpha(self) -> 'float'` — Consume and return the next decision's alpha.
  - `spent` — Sum of all alphas handed out so far.
  - `state_dict(self) -> 'dict[str, Any]'`
- `binomial_upper_tail(x: 'int', m: 'int', p: 'float') -> 'float'` — ``P[X >= x]`` for ``X ~ Bin(m, p)``.
- `clopper_pearson_lower(x: 'int', m: 'int', alpha: 'float') -> 'float'` — One-sided ``(1 - alpha)`` Clopper-Pearson lower confidence limit of a binomial proportion.
- `discordant_mass(paired: 'PairedScores') -> 'tuple[float, float]'` — ``(gain_mass, loss_mass)`` = ``(sum max(cand - inc, 0), sum max(inc - cand, 0))``.
- `fractional_sign_test_p_value(gain_mass: 'float', loss_mass: 'float') -> 'float'` — Sign-test p-value extended to fractional discordant mass.
- `noninferiority_upper_bound(paired: 'PairedScores | Sequence[PairedScores]', margin: 'float', alpha: 'float') -> 'SliceTest'` — Non-inferiority test of H0 "drop >= margin" with an upper confidence bound on the drop.
- `normal_cdf(x: 'float') -> 'float'` — Standard normal CDF ``Phi(x) = erfc(-x / sqrt(2)) / 2`` (accurate in both tails).
- `normal_quantile(p: 'float') -> 'float'` — Inverse standard normal CDF ``Phi^{-1}(p)`` for ``p`` in ``(0, 1)``.
- `paired_gain_lower_bound(paired: 'PairedScores | Sequence[PairedScores]', alpha: 'float') -> 'SliceTest'` — One-sided test of H0 "candidate not better" with a lower confidence bound on the gain.
- `pool_paired(paired: 'PairedScores | Sequence[PairedScores]') -> 'PairedScores'` — Concatenate the items of several slices into one :class:`PairedScores`.
- `regularized_incomplete_beta(a: 'float', b: 'float', x: 'float') -> 'float'` — ``I_x(a, b)`` for ``a, b > 0`` and ``x`` in ``[0, 1]`` (relative accuracy ~1e-14).
- `required_paired_n(delta: 'float', discordant_rate: 'float', alpha: 'float' = 0.05, power: 'float' = 0.8) -> 'int'` — Pairs needed to detect a paired gain ``delta`` (Connor 1987, one-sided).
- `retention_power_floor_n(margin: 'float', alpha: 'float') -> 'int'` — Smallest slice size at which a binary retention slice can pass at all.
- `sign_test_p_value(n01: 'int', n10: 'int') -> 'float'` — Exact one-sided sign-test p-value of H0 "candidate not better".
- `student_t_sf(t: 'float', df: 'float') -> 'float'` — Upper tail ``P[T >= t]`` of Student's t with ``df > 0`` degrees of freedom.
- `tango_restricted_mle(n10: 'float', n01: 'float', n: 'int', delta: 'float') -> 'float'` — Restricted MLE of the discordant probability ``q = P(incumbent=0, candidate=1)`` under the constraint ``drop = P(inc=1, cand=0) - P(inc=0, cand=1) = delta`` (``delta`` in [0, 1]).
- `tango_z(n10: 'float', n01: 'float', n: 'int', delta: 'float') -> 'float'` — Tango's score statistic for H0 ``drop = delta`` (``drop = p10 - p01``, ``delta`` in [-1, 1]).
- constants: `student_t_quantile`

## `research_rl.capability.tasks`

Operator-composition task generators, rule-level splits, item sampling and verifiers.

- **class `AuditAccessError`** — Raised when the AUDIT split is requested from a sampler built without ``allow_audit=True``.
- **class `CompositionSplits`** — Rule-level splits of one :class:`OperatorSet`.
  - `build(cls, operators: 'OperatorSet', *, seed: 'int', train_fraction: 'float' = 0.5, audit_fraction: 'float' = 0.2, depth3_count: 'int' = 16, audit_depth3_count: 'int | None' = None, task_families: 'Sequence[str]' = ('argmax', 'countbig'), max_agreement: 'float' = 0.25) -> 'CompositionSplits'` — Partition every ordered pair ``(f, g)``, ``f != g``, into TRAIN / T_RULE / AUDIT pairs.
  - `check_disjoint(self) -> 'None'` — Raise :class:`SplitIntegrityError` listing every violated invariant.
  - `fingerprint(self) -> 'str'` — sha256 of :meth:`to_dict` (canonical JSON).
  - `rules(self, split: 'Split') -> 'tuple[Rule, ...]'` — All operator rules of ``split`` (the union of its rule slices; family slices have none).
  - `slice_rules_of(self, slice_name: 'str') -> 'tuple[Rule, ...]'` — Rules of a rule slice; ``()`` for family slices; ``KeyError`` for an unknown slice.
  - `slices(self) -> 'dict[str, Split]'` — ``{slice name: split}`` for every slice (rule slices and family slices).
  - `slices_of(self, split: 'Split') -> 'tuple[str, ...]'` — Slice names belonging to ``split`` (in :meth:`slices` order).
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able description (operators, seed, every slice's rules).
- **class `ExactAnswerVerifier`** — Strict verifier V2: the completion, stripped of surrounding whitespace and of at most one trailing terminator (one of ``terminators``, then whitespace stripped again), must equal the reference (also whitespace-stripped).
- **class `ItemSampler`** — Deterministic item source over :class:`CompositionSplits`.
  - `draw(self, split: 'Split', n: 'int', *, purpose: 'str', index: 'int', slice: 'str | None' = None) -> 'list[Item]'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Check that ``state`` came from an identical sampler (``ValueError`` otherwise).
  - `state_dict(self) -> 'dict[str, Any]'` — Identity of the sampler (it has no position: items are keyed by ``(purpose, index)``).
- **class `OperatorSet`** — The first ``num_operators`` entries of :data:`OPERATOR_CATALOG` under invented names.
  - `agreement(self, a: 'Sequence[str]', b: 'Sequence[str]') -> 'float'` — Fraction of :attr:`probe_inputs` on which rules ``a`` and ``b`` give the same output (each length has the same number of probe inputs, so lengths are weighted equally).
  - `apply(self, rule: 'Sequence[str]', x: 'str') -> 'str'` — Apply ``rule`` (outermost first) to the digit string ``x``; ``()`` is the identity.
  - `describe(self) -> 'dict[str, str]'` — ``{invented name: catalog key}`` (for reports; never shown to the model).
  - `is_degenerate(self, rule: 'Sequence[str]', *, max_agreement: 'float' = 0.25) -> 'bool'` — True iff ``rule`` is :meth:`similar` to the identity or to an atomic operator (equal to one of them on the probe inputs of some length, or agreeing with one of them on at least ``max_agreement`` of the probe inputs); ``()`` and single operators are degenerate.
  - `operator_key(self, name: 'str') -> 'str'` — Catalog key (e.g. ``"reverse"``) of the operator called ``name``.
  - `signature(self, rule: 'Sequence[str]') -> 'tuple[str, ...]'` — Outputs of ``rule`` on :attr:`probe_inputs` (cached); equal functions have equal signatures, different signatures prove different functions.
  - `similar(self, a: 'Sequence[str]', b: 'Sequence[str]', *, max_agreement: 'float' = 0.25) -> 'bool'` — ``similarity([a], [b], max_agreement=...)`` as a bool.
  - `similarity(self, rows: 'Sequence[Sequence[str]]', cols: 'Sequence[Sequence[str]]', *, max_agreement: 'float' = 0.25) -> 'torch.Tensor'` — Boolean ``(len(rows), len(cols))`` matrix: entry ``[i, j]`` is True iff rules ``rows[i]`` and ``cols[j]`` are equal on all probe inputs of at least one length, or agree on at least ``max_agreement`` of all probe inputs (see :meth:`agreement`).
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `SplitIntegrityError`** — Raised by :meth:`CompositionSplits.check_disjoint`; ``violations`` lists every problem found.
- **class `SuffixLeakVerifier`** — Planted leaky verifier V1: passes iff ``text[-k:] == reference[-k:]`` where ``text`` is the completion normalised as in :class:`ExactAnswerVerifier` (whitespace, one trailing terminator).
- **class `VerifierScoreFn`** — :class:`~research_rl.capability.types.ScoreFn`: greedy completions of the item prompts, scored 1.0 when ``verifier`` marks them correct against ``item.answer``, else 0.0.
  - `generate(self, model: 'nn.Module', items: 'Sequence[Item]') -> 'list[str]'` — Greedy completions of ``items``' prompts (same order).
  - `score_completions(self, completions: 'Sequence[str | None]', items: 'Sequence[Item]', truncated: 'Sequence[bool] | None' = None) -> 'list[float]'` — 1.0 / 0.0 per completion under the verifier; ``truncated`` (default: all False) is passed through as the verifier's ``truncated`` flag.
  - `truncated(self, completions: 'Sequence[str]') -> 'list[bool]'` — Per completion: True iff it re-encodes to at least ``max_new_tokens`` tokens (see the class docstring for the limits of this test).
- `compose_prompt(rule: 'Rule', x: 'str') -> 'str'` — Prompt of a composition item: ``compose_prompt(("zib", "mok"), "4721") == "zib(mok(4721))="``.
- `invent_names(count: 'int', seed: 'int', *, forbidden: 'frozenset[str] | set[str]' = COMMON_WORDS) -> 'tuple[str, ...]'` — ``count`` distinct invented names of exactly 3 lowercase ASCII letters.
- `make_score_fn(tokenizer: 'Any', verifier: 'Verifier', *, max_new_tokens: 'int', batch_size: 'int') -> 'ScoreFn'` — A :class:`VerifierScoreFn` (see there); scores are in ``{0.0, 1.0}``, one per item.
- `sft_examples(splits: 'CompositionSplits', n: 'int', *, seed: 'int', demo_pairs: 'int' = 0) -> 'list[Item]'` — Supervised items for a low-start pilot model: ``n`` atomic-skill items plus ``demo_pairs`` composed demonstrations (``n + demo_pairs`` items in total, shuffled, distinct uids).
- constants: `COMMON_WORDS`, `DEFAULT_MAX_AGREEMENT`, `OPERATOR_CATALOG`, `PREREQUISITE_FAMILIES`, `SLICE_NAMES`, `TASK_FAMILIES`

## `research_rl.capability.tracker`

Capability curves over training (``docs/research_rl/CAPABILITY.md`` 1.8).

- **class `CapabilityPoint`** — One capability measurement (see the module docstring for the definitions).
  - `from_dict(cls, data: 'Mapping[str, Any]') -> 'CapabilityPoint'` — Inverse of :meth:`to_dict`.
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able with the paper's names: ``A``, ``R``, ``T_task``, ``T_rule``, ``D``.
- **class `CapabilityTracker`** — Measures :class:`CapabilityPoint` s of a training run.
  - `items(self, step: 'int') -> 'dict[str, list[Item]]'` — The items of the measurement at ``step`` (``purpose="track"``, ``index=step``, ``pool=TRACK_POOL``), per slice.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Restore ``reward0`` and the history; atomic. ``ValueError`` if the state came from a tracker with another sampler, start map or ``items_per_slice``.
  - `measure(self, model: 'nn.Module', step: 'int', reward_v1: 'float | None') -> 'CapabilityPoint'` — Measure the current weights at training ``step`` (int >= 0) and append the point to :attr:`history`.
  - `score(self, model: 'nn.Module', step: 'int') -> 'dict[str, SliceScores]'` — Per-slice :class:`SliceScores` of the model's current weights at ``step`` (one ``score_fn`` call over all items; no baseline or history update).
  - `state_dict(self) -> 'dict[str, Any]'` — Resume state: ``reward0``, the history, and identity fields checked on load.
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able: configuration, start map, baselines and every measured point.
- constants: `TRACK_PURPOSE`
- re-exported: `SliceStart`, `StartMap`

## `research_rl.capability.trl_gate`

The capability gate inside the existing TRL GRPO path (``docs/research_rl/CAPABILITY.md`` 1.10).

- **class `CapabilityGateCallback`** — Run the capability gate on a TRL trainer (see the module docstring for the exact schedule, what a decision writes and how state travels with checkpoints).
  - `bind(self, trainer: 'Any', *, name: 'str' = 'capability_gate') -> 'CapabilityGateCallback'` — Attach to ``trainer``: keep a reference (for ``_step`` / ``_buffered_inputs`` / ``trajectory_buffer`` / ``advantage_shapers``), register ``self`` as ``trainer.research_components[name]`` (``ValueError`` if another object holds that key) and add ``self`` to the trainer's callbacks unless it is already there.
  - `config`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Validate ``state`` and stash it; it is applied in the next ``on_train_begin`` (after HF loaded the checkpoint's weights and optimizer state).
  - `on_step_end(self, args: 'Any', state: 'Any', control: 'Any', **kwargs: 'Any') -> 'None'` — Collect the block's rewards and alarm, then decide at block boundaries / on alarms.
  - `on_train_begin(self, args: 'Any', state: 'Any', control: 'Any', **kwargs: 'Any') -> 'None'` — Check the setup, then start the gate: apply the state of a resumed checkpoint (after :meth:`load_state_dict`); else, on a further ``train()`` of the same callback, continue the gate as it is (see the module docstring, "Repeated train()"); else capture the incumbent.
  - `state_dict(self) -> 'dict[str, Any]'` — Checkpoint state (loadable with ``torch.load(weights_only=True)``): ``version``, ``config`` (``GateConfig.to_dict``), ``block_updates``, ``rollback_unit``, ``decision_index`` (next decision), ``incumbent`` (:func:`~research_rl.capability.loop.snapshot_to_state` or None before training), ``controller`` (hold streak, verdict gate, ledger), ``block`` (TRL batch rewards / screen V2 rates of the running block and the latched alarm), ``alarm_count`` / ``batches_seen`` / ``screen_seen`` (counters consumed so far), ``buffer_invalidations`` and ``sampler`` (the probe sampler's identity, or None).
- **class `DisagreementScreen`** — Screen V1-pass / V2-fail rollouts out of the GRPO advantage and feed the tier-1 alarm.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Restore a :meth:`state_dict` (``ValueError`` on a version / alarm presence mismatch or invalid counters; the alarm's own ``load_state_dict`` checks its config).
  - `recent(self, n: 'int') -> 'list[tuple[float, float]]'` — The ``(reward_v1, reward_v2)`` means of the last ``n`` training batches (at most :attr:`RECENT`), oldest first.
  - `state_dict(self) -> 'dict[str, Any]'` — ``version``, ``alarm`` (its ``state_dict`` or None), ``train_batches``, ``rows``, ``screened`` and ``recent`` (list of ``[reward_v1, reward_v2]``).
- `base_optimizer(optimizer: 'Any') -> 'torch.optim.Optimizer | None'` — The optimizer the gate snapshots and restores: the ``HookedOptimizer`` inside any wrapper (:func:`~research_rl.optim.optimizer_wrapper.unwrap_hooked`), else the innermost ``torch.optim.Optimizer`` reachable through ``.optimizer`` attributes (accelerate's ``AcceleratedOptimizer`` wraps the real one), else ``None`` for ``None``.
- `generate_every(args: 'Any', trainer: 'Any') -> 'int'` — TRL's generation period in micro-batches: ``steps_per_generation * num_iterations`` (``steps_per_generation`` defaults to ``gradient_accumulation_steps`` as in ``GRPOConfig``; ``num_iterations`` to 1).
- constants: `GATE_COMPONENT`

## `research_rl.capability.types`

Shared contracts of the capability-gated training package.

- **class `ChannelGroup`** — A named set of FFN channels: the unit of subset search and of group rollback.
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `DecisionRecord`** — Audit record of one full gate procedure (accept / hold / shrink / group rollback / rollback).
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `GateDecision`** — Outcome of one gate decision (the loop and operators only ever see this value).
  - members: `ACCEPT`, `HOLD`, `SHRINK`, `GROUP_ROLLBACK`, `ROLLBACK`
- **class `GateEvidence`** — Everything the gate decides on for one decision.
- **class `GateVerdict`** — Result of evaluating one :class:`GateEvidence` (pure statistics, no weight changes).
  - `retention_ok`
  - `to_dict(self) -> 'dict[str, Any]'`
  - `transfer_ok`
- **class `Item`** — One verifiable task instance.
- **class `OperatorReport`** — What one operator update did. ``v1_pass`` / ``v2_pass`` are per-rollout verifier outcomes on the training rollouts of this update (same order, same length), used by the tier-1 alarm. ``screened`` counts rollouts excluded from the update because V1 and V2 disagreed.
- **class `PairedScores`** — Scores of the incumbent and the candidate on the *same* items of one slice.
  - `discordant(self) -> 'tuple[int, int]'` — ``(n01, n10)`` for binary scores: candidate-only wins, incumbent-only wins.
  - `is_binary`
  - `mean_difference(self) -> 'float'` — Mean of candidate - incumbent (positive = candidate better).
  - `n`
- **class `ParameterSpace`** — A flat, ``dim``-dimensional coordinate system over some model parameters.
  - `parameter_names(self) -> 'list[str]'`
  - `rebase(self) -> 'None'`
  - `write(self, vector: 'torch.Tensor | None') -> 'None'`
- **class `RollbackUnit`** — Finest unit a failed retention test may revert.
  - members: `CHANNELS`, `PARAMETERS`
- **class `ScoreFn`** — Base class for protocol classes.
- **class `SliceTest`** — One statistical test inside a verdict.
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `Split`** — Where an item may be used. Rule-level disjointness between splits is enforced by the task generators (:mod:`research_rl.capability.tasks`), not by this enum.
  - members: `TRAIN`, `T_RULE`, `T_TASK`, `RETENTION`, `AUDIT`
- **class `UpdateOperator`** — Produces one parameter update per :meth:`step`, in place on ``model``.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `on_decision(self, decision: 'GateDecision') -> 'None'`
  - `parameter_names(self) -> 'list[str]'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `step(self, model: 'nn.Module', update_index: 'int') -> 'OperatorReport'`

## `research_rl.capability.weights`

Exact weight snapshots, temporary swaps, interpolation and channel-group rollback.

- **class `SnapshotError`** — A snapshot does not fit the model / optimizer it is applied to (raised before any write).
- **class `StateSnapshot`** — An exact copy of (part of) a model's training state. Treat it as immutable.
  - `names(self) -> 'list[str]'` — Captured parameter names, in ``named_parameters`` order.
  - `verify(self) -> 'None'` — Raise :class:`SnapshotError` if ``params`` no longer match ``fingerprint`` (mutated).
- `capture(model: 'nn.Module', names: 'Iterable[str] | None', *, optimizer: 'torch.optim.Optimizer | None' = None, channel_layers: 'Iterable[int] | None' = None) -> 'StateSnapshot'` — Snapshot the parameters ``names`` (``None`` = every parameter), the optimizer and masters.
- `channel_groups_by_damage(incumbent: 'StateSnapshot', candidate_model: 'nn.Module', activation_scores: 'Mapping[int, torch.Tensor]', groups: 'Sequence[ChannelGroup]') -> 'list[ChannelGroup]'` — ``groups`` ordered by :func:`group_damage`, largest first.
- `group_damage(incumbent: 'StateSnapshot', candidate_model: 'nn.Module', activation_scores: 'Mapping[int, torch.Tensor]', groups: 'Sequence[ChannelGroup]') -> 'dict[str, float]'` — Damage of each group: ``sum_{(layer, c) in group} displacement[layer, c] * activation[layer][c]``.
- `interpolate(model: 'nn.Module', incumbent: 'StateSnapshot', candidate: 'StateSnapshot', lam: 'float', *, optimizer: 'torch.optim.Optimizer | None' = None) -> 'None'` — Write ``incumbent + lam * (candidate - incumbent)`` into the snapshots' parameters.
- `restore(model: 'nn.Module', snapshot: 'StateSnapshot', *, optimizer: 'torch.optim.Optimizer | None' = None, restore_optimizer: 'bool' = True) -> 'None'` — Write the snapshot back: parameters, optimizer state and masters bit-identical to it.
- `revert_groups(model: 'nn.Module', incumbent: 'StateSnapshot', groups: 'Sequence[ChannelGroup]', *, optimizer: 'torch.optim.Optimizer | None' = None) -> 'list[RollbackReport]'` — Revert the channels of ``groups`` to the incumbent; returns one ``RollbackReport`` per group.
- `swapped(model: 'nn.Module', snapshot: 'StateSnapshot') -> 'Iterator[StateSnapshot]'` — Temporarily load the snapshot's parameters (e.g. to evaluate the incumbent).
<!-- capability-api:end -->

## `research_rl.core.checkpoint`

Checkpoint contents and ResearchRL-specific resume state.

- **class `StatefulComponent`**
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
- `checkpoint_inventory(checkpoint_dir: 'str | Path') -> 'dict[str, bool]'` — Which resume-relevant artefacts exist in a checkpoint directory.
- `latest_checkpoint(output_dir: 'str | Path') -> 'str | None'`
- `load_components(checkpoint_dir: 'str | Path', components: 'Mapping[str, StatefulComponent]') -> 'list[str]'` — Restore registered components; returns the names that were restored.
- `load_research_state(checkpoint_dir: 'str | Path') -> 'dict[str, Any]'`
- `save_components(checkpoint_dir: 'str | Path', components: 'Mapping[str, StatefulComponent]') -> 'Path | None'`
- `save_research_state(checkpoint_dir: 'str | Path', state: 'Mapping[str, Any]') -> 'Path'`

## `research_rl.core.config`

Versioned experiment configuration.

- **class `ConfigError`**
- **class `DataSpec`** — Identity of the training data (the dataset object itself is passed at run time).
- **class `ExperimentConfig`** — ExperimentConfig(study_id: 'str', model: 'ModelSpec', data: 'DataSpec', trainer: 'TrainerSpec', seed: 'int' = 0, output_dir: 'str' = 'runs', run_name: 'str | None' = None, optim: 'OptimSpec' = <factory>, tags: 'list[str]' = <factory>, extra: 'dict[str, Any]' = <factory>, config_version: 'str' = '1.0')
  - `from_dict(cls, raw: 'dict[str, Any]') -> 'ExperimentConfig'`
  - `load(cls, path: 'str | Path') -> 'ExperimentConfig'`
  - `resolved_run_name`
  - `run_dir`
  - `save(self, path: 'str | Path') -> 'Path'`
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `ModelSpec`** — Where the policy weights come from.
- **class `OptimSpec`** — ResearchRL optimizer-extension settings (see :mod:`research_rl.optim`).
- **class `TrainerSpec`** — Which registered trainer adapter to use and the raw TRL config arguments.

## `research_rl.core.experiment`

Unified experiment records.

- **class `ExperimentRecorder`** — Collects identity, metrics, checkpoints and evaluations for one run.
  - `add_checkpoint(self, path: 'str | Path', step: 'int') -> 'None'`
  - `add_evaluation(self, name: 'str', results: 'Mapping[str, Any]', step: 'int | None' = None) -> 'None'`
  - `finalize(self, status: 'str' = 'completed') -> 'dict[str, Any]'`
  - `history(self, key: 'str') -> 'list[float]'`
  - `load(run_dir: 'str | Path') -> 'dict[str, Any]'`
  - `log_metrics(self, step: 'int', logs: 'Mapping[str, Any]') -> 'None'`
  - `record_gpu_metrics(self, metrics: 'Mapping[str, Any]') -> 'None'` — Store measured GPU metrics. Refuses unless running on CUDA right now.
  - `resume_from(self, global_step: 'int') -> 'None'` — Continue the record of an interrupted run from checkpoint ``global_step``.
  - `set_dataset(self, dataset: 'Any', name: 'str | None' = None) -> 'None'`
  - `set_extra(self, key: 'str', value: 'Any') -> 'None'` — Store an algorithm-specific top-level field (e.g. reference/teacher model identity).
  - `set_model(self, model: 'Any', revision: 'str | None' = None, hash_weights: 'bool | None' = None) -> 'None'`
  - `set_steps(self, *, nominal: 'int', applied: 'int', skipped: 'int', global_step: 'int') -> 'None'`
- `dataset_identity(dataset: 'Any', name: 'str | None' = None, max_rows_for_hash: 'int' = 200000) -> 'dict[str, Any]'` — Stable identity of a dataset: row count, column names and a content hash.
- `environment_info() -> 'dict[str, Any]'`
- `model_identity(model: 'Any', revision: 'str | None' = None, hash_weights: 'bool | None' = None) -> 'dict[str, Any]'` — Model revision info: HF commit hash if known, and (for small models) a weights hash.

## `research_rl.core.registry`

Small, explicit plugin registries.

- **class `Registry`** — A name -> object mapping with decorator-based registration.
  - `add(self, name: 'str', obj: 'T', *, overwrite: 'bool' = False) -> 'None'`
  - `build(self, name: 'str', *args: 'Any', **kwargs: 'Any') -> 'Any'` — Look up ``name`` and call it (useful when entries are classes or factories).
  - `get(self, name: 'str') -> 'T'`
  - `names(self) -> 'list[str]'`
  - `register(self, name: 'str | None' = None, *, overwrite: 'bool' = False) -> 'Callable[[T], T]'` — Decorator registering ``obj`` under ``name`` (defaults to ``obj.__name__``).
- **class `RegistryError`** — Raised for unknown or duplicate registry entries.

## `research_rl.evaluation.capability_retention`

Capability retention: does RL(VR) training degrade other capabilities, and does recovery help?

- **class `CapabilityRetention`** — Retention numbers of one capability. ``None`` marks a quantity that is undefined.
  - `compute(cls, name: 'str', base: 'float', after_training: 'float', after_recovery: 'float | None' = None, *, drop_tolerance: 'float' = 0.0) -> 'CapabilityRetention'`
- **class `KLResult`** — Mean per-token ``KL(base || current)`` over probe prompt positions (natural log).
- **class `ProbeTask`** — One capability probe: ``(prompt, answer)`` examples scored by ``scorer``.
  - `prompts`
  - `scorer(prediction: 'str', answer: 'str') -> 'float'` — 1.0 if the whitespace-normalised prediction equals the answer, else 0.0.
- **class `RetentionReport`** — Comparison ``base -> after_training -> after_recovery`` over all capabilities.
  - `from_scores(cls, base: 'Mapping[str, float]', after_training: 'Mapping[str, float]', after_recovery: 'Mapping[str, float] | None' = None, *, drop_tolerance: 'float' = 0.0, kl_after_training: 'Mapping[str, float] | None' = None, kl_after_recovery: 'Mapping[str, float] | None' = None) -> 'RetentionReport'`
  - `log_to(self, recorder: 'Any', name: 'str' = 'capability_retention', step: 'int | None' = None) -> 'None'` — Append this report to an :class:`~research_rl.core.experiment.ExperimentRecorder`.
  - `summary(self) -> 'dict[str, Any]'` — Unweighted means over capabilities; ratio means use only capabilities where defined.
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `RetentionSuite`** — A set of :class:`ProbeTask` evaluated with deterministic greedy decoding.
  - `evaluate(self, model: 'nn.Module', tokenizer: 'Any') -> 'SuiteResult'`
  - `logit_kl(self, base_model: 'nn.Module', model: 'nn.Module', tokenizer: 'Any') -> 'dict[str, float]'` — Per-task mean per-token KL plus ``"all"`` (token-weighted over every probe prompt).
  - `report(self, base_model: 'nn.Module', trained_model: 'nn.Module', tokenizer: 'Any', recovered_model: 'nn.Module | None' = None, *, with_kl: 'bool' = True, drop_tolerance: 'float' = 0.0) -> 'RetentionReport'` — Evaluate all models and build a :class:`RetentionReport`.
- **class `SuiteResult`** — Per-capability mean score plus the raw predictions they were computed from.
- `exact_match_scorer(prediction: 'str', answer: 'str') -> 'float'` — 1.0 if the whitespace-normalised prediction equals the answer, else 0.0.
- `greedy_generate(model: 'nn.Module', tokenizer: 'Any', prompts: 'Sequence[str]', *, max_new_tokens: 'int', batch_size: 'int' = 1) -> 'list[str]'` — Deterministic greedy continuations of ``prompts`` (decoded, EOS and everything after removed).
- `logit_kl(base_model: 'nn.Module', model: 'nn.Module', tokenizer: 'Any', prompts: 'Iterable[str]') -> 'KLResult'` — Token-weighted mean of ``KL(p_base || p_model)`` over every position of every prompt.
- `prefix_match_scorer(prediction: 'str', answer: 'str') -> 'float'` — 1.0 if the whitespace-normalised prediction *starts with* the answer.

## `research_rl.evaluation.metrics`

Evaluation metrics: exact match, unbiased pass@k, bootstrap confidence intervals, token-level NLL.

- **class `MeanCI`** — Sample mean with a percentile-bootstrap confidence interval.
  - `to_dict(self) -> 'dict[str, float | int]'`
- **class `NLLResult`** — Token-level negative log-likelihood (natural log) of a causal LM.
  - `to_dict(self) -> 'dict[str, Any]'`
- `exact_match(prediction: 'str', reference: 'str', *, normalize: 'bool' = True, case_sensitive: 'bool' = True) -> 'bool'` — String equality, optionally after :func:`normalize_answer`.
- `exact_match_accuracy(predictions: 'Sequence[str]', references: 'Sequence[str]', *, normalize: 'bool' = True, case_sensitive: 'bool' = True) -> 'float'` — Fraction of predictions that exactly match their reference.
- `mean_pass_at_k(num_samples: 'Sequence[int]', num_correct: 'Sequence[int]', k: 'int') -> 'float'` — Average of :func:`pass_at_k` over problems (``num_samples[i]``, ``num_correct[i]``).
- `mean_with_bootstrap_ci(values: 'Sequence[float] | np.ndarray | torch.Tensor', *, confidence: 'float' = 0.95, n_resamples: 'int' = 2000, seed: 'int' = 0) -> 'MeanCI'` — Mean and percentile-bootstrap CI, deterministic given ``seed``.
- `normalize_answer(text: 'str', *, case_sensitive: 'bool' = True) -> 'str'` — Strip surrounding whitespace and collapse internal whitespace runs to one space.
- `pass_at_k(n: 'int', c: 'int', k: 'int') -> 'float'` — Unbiased pass@k estimator (Chen et al., 2021): ``1 - C(n-c, k) / C(n, k)``.
- `sequence_nll(model: 'nn.Module', tokenizer: 'Any', pairs: 'Sequence[tuple[str, str]]', *, completion_only: 'bool' = True, batch_size: 'int' = 8, add_bos: 'bool' = False) -> 'NLLResult'` — Summed / mean NLL of ``(prompt, completion)`` pairs under a causal LM.

## `research_rl.models.expert_manager`

Expert Manager: a shared backbone with independently trained, response-level FFN experts.

- **class `BackboneChangedError`** — The shared backbone differs from its recorded fingerprint.
- **class `ExpertCompatibilityError`** — An expert file / checkpoint does not match this model.
- **class `ExpertError`** — Unknown expert, invalid name or invalid manager operation.
- **class `ExpertFFN`** — Drop-in replacement for ``layer.mlp`` holding several SwiGLU experts.
  - `extra_repr(self) -> 'str'`
  - `forward(self, x: 'torch.Tensor') -> 'torch.Tensor'`
- **class `ExpertManager`** — Manage response-level FFN experts on an HF decoder-only model with SwiGLU FFNs.
  - `activate(self, name: 'str') -> 'None'` — Use expert ``name`` for every row (clears per-sample routing).
  - `active`
  - `assert_backbone_unchanged(self, reference: 'str | None' = None) -> 'None'`
  - `attach(self, layers: 'Iterable[int] | None' = None) -> 'list[int]'` — Replace ``layer.mlp`` of the given decoder layers (default: all) by an :class:`ExpertFFN`.
  - `attached`
  - `backbone_fingerprint(self) -> 'str'`
  - `backbone_parameter_names(self) -> 'list[str]'` — Every parameter that does not belong to any expert (the shared backbone).
  - `clear_routing(self) -> 'None'`
  - `create_expert(self, name: 'str', source: 'str' = 'base') -> 'None'` — Add expert ``name`` as an exact copy of expert ``source`` in every attached layer.
  - `delete_expert(self, name: 'str') -> 'None'`
  - `detach(self, name: 'str' = 'base') -> 'None'` — Put expert ``name`` back as the vanilla ``layer.mlp`` (same parameter objects) and drop the other experts. Parameter ``requires_grad`` flags are left as they are.
  - `expert_fingerprint(self, name: 'str') -> 'str'`
  - `expert_names`
  - `expert_parameter_names(self, name: 'str') -> 'list[str]'`
  - `expert_parameters(self, name: 'str') -> 'list[nn.Parameter]'`
  - `export_merged(self, name: 'str', path: 'str | Path', **save_kwargs: 'Any') -> 'Path'` — Save a *vanilla* HF checkpoint (standard ``model.layers.i.mlp.*`` keys) with expert ``name``.
  - `ffn_modules(self) -> 'dict[int, ExpertFFN]'` — The attached :class:`ExpertFFN` modules by decoder-layer index (a new dict, same modules).
  - `freeze_backbone(self, *, rebaseline: 'bool' = False) -> 'str'` — Freeze everything except the trainable expert (if any) and return the backbone reference fingerprint used by :meth:`assert_backbone_unchanged`.
  - `layers`
  - `load_expert(self, path: 'str | Path', name: 'str | None' = None, *, overwrite: 'bool' = False, check_backbone: 'bool' = True) -> 'str'` — Load an expert file into this model; validates everything before changing the model.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Recreate the expert structure (attach, create experts, trainability, selection).
  - `merged_model(self, name: 'str') -> 'nn.Module'` — A deep copy of the model with expert ``name`` as a plain ``layer.mlp`` (all params trainable).
  - `merged_state_dict(self, name: 'str') -> 'dict[str, torch.Tensor]'` — State dict with vanilla keys in which expert ``name`` is the FFN (no copies made).
  - `read_expert_file(path: 'str | Path') -> 'tuple[dict[str, torch.Tensor], dict[str, str]]'`
  - `routed(self, names: 'Sequence[str]') -> 'Iterator[None]'` — Temporarily route row ``b`` through expert ``names[b]``.
  - `routing`
  - `rows(self, names: 'Sequence[str]') -> 'Iterator[None]'` — Temporarily select the experts of a batch whose row ``b`` belongs to expert ``names[b]``.
  - `save_expert(self, name: 'str', path: 'str | Path') -> 'Path'` — Write one expert (all attached layers) as safetensors with compatibility metadata.
  - `save_pretrained(self, path: 'str | Path', **save_kwargs: 'Any') -> 'Path'` — Save the model *with experts attached* plus a manager sidecar (see :func:`load_model_with_experts`).
  - `set_routing(self, names: 'Sequence[str] | None') -> 'None'` — Per-sample (response-level) routing: ``names[b]`` is the expert of batch row ``b``.
  - `set_trainable(self, name: 'str | None', *, rebaseline: 'bool' = False) -> 'None'` — Make expert ``name`` the only trainable parameters and activate it (``None``: freeze all).
  - `state_dict(self) -> 'dict[str, Any]'` — Structure only (no tensors: expert weights live in the model state dict).
  - `trainable_expert`
  - `trainable_parameter_names(self) -> 'list[str]'`
  - `using(self, name: 'str') -> 'Iterator[None]'` — Temporarily use expert ``name`` for all rows.
- `config_hash(config: 'Any') -> 'str'` — SHA-256 of the architecture-relevant part of an HF config (stable across save/load).
- `load_model_with_experts(path: 'str | Path', model_factory: 'Callable[[Any, torch.dtype], nn.Module] | None' = None) -> 'tuple[nn.Module, ExpertManager]'` — Load a checkpoint saved with experts attached (``ExpertManager.save_pretrained`` or an HF Trainer checkpoint) and re-attach its experts. Fails if keys do not match exactly.
- `model_class_name(model: 'nn.Module') -> 'str'`

## `research_rl.models.ffn_channels`

Channel-level parameter operations on SwiGLU FFNs (selective fine-tuning, diffing, rollback).

- **class `ChannelDiff`** — Per-channel change statistics for one layer (float64).
  - `changed_channels(self) -> 'list[int]'`
  - `relative`
  - `score(self, metric: 'Metric') -> 'torch.Tensor'`
- **class `ChannelSelection`** — Immutable mapping ``layer_index -> sorted unique channel indices``.
  - `all_channels(cls, model: 'nn.Module', layers: 'Iterable[int] | None' = None, *, expert: 'str | None' = None) -> 'ChannelSelection'`
  - `channels(self, layer: 'int') -> 'tuple[int, ...]'`
  - `complement(self, model: 'nn.Module', *, expert: 'str | None' = None) -> 'ChannelSelection'` — All channels of the *affected* layers that are not selected.
  - `count(self) -> 'int'`
  - `difference(self, other: 'ChannelSelection') -> 'ChannelSelection'`
  - `from_dict(cls, d: 'Mapping[int | str, Iterable[int]]', *, intermediate_size: 'int | None' = None) -> 'ChannelSelection'` — Build from a (possibly JSON-loaded, string-keyed) dict.
  - `index_tensor(self, layer: 'int', device: 'torch.device | str | None' = None) -> 'torch.Tensor'`
  - `intersection(self, other: 'ChannelSelection') -> 'ChannelSelection'`
  - `items(self) -> 'Iterator[tuple[int, tuple[int, ...]]]'`
  - `layers(self) -> 'list[int]'`
  - `to_dict(self) -> 'dict[int, list[int]]'`
  - `top_k_from_scores(cls, scores: 'Mapping[int, torch.Tensor]', k: 'int', *, per_layer: 'bool' = False, largest: 'bool' = True) -> 'ChannelSelection'` — Select the ``k`` best channels globally (or ``k`` per layer).
  - `union(self, other: 'ChannelSelection') -> 'ChannelSelection'`
  - `validate(self, model: 'nn.Module', *, expert: 'str | None' = None) -> 'None'`
- **class `ChannelSelectionError`**
- **class `ChannelState`** — ChannelState(layers: 'dict[int, LayerChannels]')
  - `selection(self) -> 'ChannelSelection'`
- **class `ChannelTrainingHandle`** — Result of :func:`setup_channel_training`.
  - `remove(self) -> 'None'`
- **class `LayerChannels`** — Detached copies of the slices owned by ``channels`` in one layer (original dtype).
  - `select(self, channels: 'Iterable[int]') -> 'LayerChannels'` — Sub-state for ``channels`` (each must be present).
  - `vectors(self, dtype: 'torch.dtype' = torch.float64) -> 'torch.Tensor'` — ``[k, D]``: per channel the concatenation [gate row, up row, down col, (biases)].
- **class `RollbackReport`** — Per layer: rolled-back channels, how far they were from the reference, elements that changed.
  - `total_elements_changed`
- **class `ScopeReport`** — ScopeReport(violations: 'dict[str, ScopeViolation]', params_checked: 'int', elements_checked: 'int', changed_allowed: 'int')
  - `ok`
  - `summary(self) -> 'str'`
  - `total_violations`
- **class `ScopeViolation`** — ScopeViolation(count: 'int', max_abs: 'float', indices: 'torch.Tensor')
- **class `ScopeViolationError`** — Raised by :func:`verify_modification_scope` in strict mode.
- `ablate_channels(model: 'nn.Module', selection: 'ChannelSelection', *, expert: 'str | None' = None) -> 'ChannelState'` — Zero the selected channels' slices in place; returns their previous values (for rollback).
- `channel_diff(reference: 'ChannelState', current: 'ChannelState | nn.Module', layers: 'Iterable[int] | None' = None, *, expert: 'str | None' = None, allow_path_mismatch: 'bool' = False) -> 'dict[int, ChannelDiff]'` — Compare two channel states (or a state against a live model) channel by channel.
- `channel_mask(model: 'nn.Module', selection: 'ChannelSelection', *, trainable: 'bool' = True, expert: 'str | None' = None) -> 'ParameterMask'` — Mask over the gate/up/down weights (and biases) of every affected layer (of ``expert``).
- `channel_state(model: 'nn.Module', layers: 'Iterable[int] | None' = None, *, expert: 'str | None' = None) -> 'ChannelState'` — Every channel of ``layers`` (default: all FFN layers). Use as a rollback/diff reference.
- `extract_channels(model: 'nn.Module', selection: 'ChannelSelection', *, expert: 'str | None' = None) -> 'ChannelState'` — Detached CPU copies (original dtype) of the slices owned by the selected channels.
- `freeze_mask(model: 'nn.Module', selection: 'ChannelSelection', *, expert: 'str | None' = None) -> 'ParameterMask'` — Selected channels frozen; everything else in the affected FFNs may change.
- `ranked_channels(diff: 'Mapping[int, ChannelDiff]', metric: 'Metric' = 'l2') -> 'list[tuple[int, int, float]]'` — All ``(layer, channel, score)`` sorted by score descending, ties by (layer, channel).
- `rollback_channels(model: 'nn.Module', reference: 'ChannelState', selection: 'ChannelSelection | None' = None, *, allow_cast: 'bool' = False, expert: 'str | None' = None, allow_path_mismatch: 'bool' = False) -> 'RollbackReport'` — Restore ONLY the selected channels' slices (gate row, up row, down col, biases) to ``reference``.
- `setup_channel_training(model: 'nn.Module', selection: 'ChannelSelection', *, freeze_rest: 'bool' = True, disable_grad_outside: 'bool' = False, gradient_masks: 'bool' = True, expert: 'str | None' = None) -> 'ChannelTrainingHandle'` — Prepare ``model`` so that only the selected channel slices (of ``expert``) can change.
- `top_changed_channels(diff: 'Mapping[int, ChannelDiff]', k: 'int', *, metric: 'Metric' = 'l2', per_layer: 'bool' = False, min_score: 'float | None' = None) -> 'ChannelSelection'` — The ``k`` most-changed channels (globally or per layer); ``min_score`` drops scores <= it.
- `verify_modification_scope(before: 'Mapping[str, torch.Tensor]', after: 'Mapping[str, torch.Tensor] | nn.Module', allowed: 'ParameterMask | ChannelSelection', model: 'nn.Module', *, unlisted: "Literal['frozen', 'free']" = 'frozen', strict: 'bool' = False, expert: 'str | None' = None) -> 'ScopeReport'` — Report every element of ``before`` that differs in ``after`` outside ``allowed``.

## `research_rl.models.model_interface`

Model introspection shared by all parameter-level tools.

- **class `ModelInterfaceError`**
- **class `SwiGLURef`** — Location of one SwiGLU FFN inside a model.
  - `down`
  - `gate`
  - `intermediate_size`
  - `param_names(self) -> 'dict[str, str]'` — Fully-qualified weight names, keyed by projection name.
  - `up`
- `count_parameters(model: 'nn.Module', trainable_only: 'bool' = False) -> 'int'`
- `find_swiglu_ffns(model: 'nn.Module', layers: 'Iterable[int] | None' = None, *, expert: 'str | None' = None) -> 'list[SwiGLURef]'` — Locate the SwiGLU FFN (``layer.mlp``) of each decoder layer.
- `fingerprint(tensors: 'dict[str, torch.Tensor] | nn.Module', names: 'Iterable[str] | None' = None) -> 'str'` — SHA-256 over names, shapes, dtypes and raw bytes. Bit-exact identity check.
- `get_decoder_layers(model: 'nn.Module') -> 'tuple[str, nn.ModuleList]'` — Return ``(path, layers)`` for the decoder block list of an HF causal LM.
- `is_swiglu(module: 'nn.Module') -> 'bool'`
- `snapshot(model: 'nn.Module', names: 'Iterable[str] | None' = None) -> 'dict[str, torch.Tensor]'` — Detached CPU copy of parameters (and only parameters) for exact comparisons.

## `research_rl.models.parameter_mask`

Element-wise parameter masks.

- **class `MaskError`**
- **class `ParameterMask`**
  - `all_frozen(cls, model: 'nn.Module', names: 'Iterable[str]') -> 'ParameterMask'`
  - `all_trainable(cls, model: 'nn.Module', names: 'Iterable[str]') -> 'ParameterMask'`
  - `changed_outside(self, before: 'Mapping[str, torch.Tensor]', after: 'Mapping[str, torch.Tensor]') -> 'dict[str, int]'` — Number of masked-out elements that differ between two snapshots (should be all 0).
  - `from_state_dict(cls, state: 'Mapping[str, torch.Tensor]') -> 'ParameterMask'`
  - `get(self, name: 'str') -> 'torch.Tensor | None'`
  - `intersection(self, other: 'ParameterMask') -> 'ParameterMask'`
  - `items(self) -> 'Iterator[tuple[str, torch.Tensor]]'`
  - `names(self) -> 'list[str]'`
  - `num_frozen(self) -> 'int'`
  - `num_trainable(self) -> 'int'`
  - `restore_frozen(self, model: 'nn.Module', reference: 'Mapping[str, torch.Tensor]') -> 'None'` — Write ``reference`` back into every masked-out element of ``model``.
  - `set(self, name: 'str', mask: 'torch.Tensor') -> 'None'`
  - `state_dict(self) -> 'dict[str, torch.Tensor]'`
  - `to(self, device: 'torch.device | str') -> 'ParameterMask'`
  - `union(self, other: 'ParameterMask') -> 'ParameterMask'` — Element may change if it may change in either mask.
  - `validate(self, model: 'nn.Module') -> 'None'`

## `research_rl.models.safe_checkpoint`

Safetensors-only check of a local Hugging Face model directory (no weights are read). Shared by the UI's `--model-dir` loader (`research_rl.ui.loading.check_model_dir`) and `scripts/gpu/capability_p1.py` (`validate_model_dir`).

- `PICKLE_SUFFIXES` — Pickle-based checkpoint formats: never loaded (they can execute code on load).
- `SAFE_WEIGHTS_FILE` — Single-file safetensors weights: the first file ``transformers`` opens with ``use_safetensors``.
- `SAFE_WEIGHTS_INDEX_FILE` — Sharded safetensors index: opened when :data:`SAFE_WEIGHTS_FILE` is absent.
- **class `SafetensorsDir`** — Result of :func:`check_safetensors_dir`.
- **class `UnsafeCheckpointError`** — The directory is not a checkpoint whose weights ``transformers`` would read from safetensors only.
- `check_safetensors_dir(path: 'str | Path', *, label: 'str | None' = None) -> 'SafetensorsDir'` — Check ``path`` without reading any weights (see the module docstring for the rules).
- `is_plain_file_name(name: 'str') -> 'bool'` — ``True`` for a bare file name inside the directory (no separator, ``..`` or absolute path).

## `research_rl.models.tiny`

Tiny, fully offline models and tokenizers for tests, CI and smoke runs.

- `build_char_tokenizer(chars: 'str' = '0123456789+-*/=()^ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!:;\'"\n<>{}[]\\_$#') -> 'PreTrainedTokenizerFast'`
- `build_tiny_llama(vocab_size: 'int', seed: 'int' = 0, **config_overrides: 'int') -> 'LlamaForCausalLM'`
- `save_tiny_model(path: 'str | Path', seed: 'int' = 0, **config_overrides: 'int') -> 'Path'` — Write a tiny Llama + char tokenizer to ``path`` (a normal HF model directory).
- `tiny_llama_config(vocab_size: 'int', hidden_size: 'int' = 32, intermediate_size: 'int' = 64, num_hidden_layers: 'int' = 2, num_attention_heads: 'int' = 4, num_key_value_heads: 'int' = 2, max_position_embeddings: 'int' = 256) -> 'LlamaConfig'`

## `research_rl.optim.gradient_hooks`

Backward-time gradient hooks.

- **class `GradientHookHandle`** — Holds the underlying torch handles; call :meth:`remove` to detach all hooks.
  - `remove(self) -> 'None'`
- `global_grad_norm(params: 'Iterable[torch.nn.Parameter]', norm_type: 'float' = 2.0) -> 'float'` — Reference global gradient norm (independent of ``clip_grad_norm_``).
- `register_gradient_masks(model: 'nn.Module', mask: 'ParameterMask') -> 'GradientHookHandle'` — Zero gradient elements that ``mask`` marks as frozen, at backward time.
- `register_gradient_transform(model: 'nn.Module', transform: 'GradTransform', names: 'Iterable[str] | None' = None) -> 'GradientHookHandle'` — Apply ``transform(name, grad) -> grad`` to the gradients of ``names`` (default: all trainable).

## `research_rl.optim.optimizer_wrapper`

``HookedOptimizer``: controls the *actual* parameter displacement of any torch optimizer.

- **class `HookedOptimizer`** — Wraps ``inner`` (any ``torch.optim.Optimizer``) and exposes the same interface.
  - `add_param_group(self, param_group: 'dict[str, Any]') -> 'None'`
  - `add_rule(self, rule: 'UpdateRule') -> 'None'`
  - `has_master(self, param: 'torch.Tensor') -> 'bool'` — True if ``param`` is backed by an fp32 master copy.
  - `load_state_dict(self, state_dict: 'dict[str, Any]') -> 'None'`
  - `master_of(self, param: 'torch.Tensor') -> 'torch.Tensor | None'` — The fp32 master of ``param`` (the live tensor, do not modify), or ``None``.
  - `named_parameters(self) -> 'list[tuple[str, torch.nn.Parameter]]'`
  - `param_groups`
  - `policy_version` — Number of parameter updates actually applied (skipped steps excluded).
  - `request_skip(self) -> 'None'` — Make a ``step()`` a no-op (counted in ``skipped_steps``, not in ``applied_updates``).
  - `resync_masters(self) -> 'None'` — Reset every master to ``param.float()`` (discards the sub-ulp information they hold).
  - `set_master(self, param: 'torch.nn.Parameter', value: 'torch.Tensor') -> 'None'` — Set the fp32 master of ``param`` to ``value`` and write ``round_to(value, nearest)`` into it.
  - `state`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `step(self, closure: 'Callable[[], float] | None' = None) -> 'Any'`
  - `zero_grad(self, set_to_none: 'bool' = True) -> 'None'`
- `unwrap_hooked(optimizer: 'Any') -> 'HookedOptimizer | None'` — Find a HookedOptimizer inside accelerate/deepspeed wrappers (or return None).

## `research_rl.optim.parameter_prior`

Parameter priors built from task vectors (P²-MOPD-style parameter-direction fusion).

- **class `P2PriorAttachCallback`** — Attaches a :class:`P2PriorRule` to the (accelerate-wrapped) trainer optimizer at train begin.
  - `on_train_begin(self, args: 'Any', state: 'Any', control: 'Any', **kwargs: 'Any') -> 'None'`
- **class `P2PriorRule`** — Two-channel parameter-prior update of P²-MOPD (Algorithm 1) for :class:`HookedOptimizer`.
  - `attach(self, optimizer: 'Any') -> 'None'` — Bind to the optimizer and validate it (fails closed).
  - `callback(self) -> 'TrainerCallback'` — A ``TrainerCallback`` that calls :meth:`attach` with the trainer's optimizer at train begin.
  - `coefficient_gradient(self, named_params: 'Iterable[tuple[str, torch.nn.Parameter]]') -> 'torch.Tensor'` — g [F] = tied Σ <G_m, τ_{k,m}> from the current ``.grad`` (fp32, on CPU).
  - `coefficient_table(self) -> 'dict[str, torch.Tensor]'` — Effective α_{·,m} (shape [K]) for each prior module m.
  - `free_coefficients(self) -> 'dict[str, torch.Tensor]'` — Free coefficients per group (module / layer / model), shape [K] each.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `lr_factor(self, step: 'int') -> 'float'` — Multiplier of ``lr`` at 1-based optimizer step ``step`` (``lr_lambda`` / warm-up / 1).
  - `materialize(self, model: 'nn.Module', base: 'Mapping[str, torch.Tensor]') -> 'None'` — Write W_m = B_m + Σ_k α_{k,m} τ_{k,m} (current α) into ``model`` for every prior module.
  - `on_attach(self, optimizer: 'Any') -> 'None'` — HookedOptimizer calls this when the rule is registered: validate immediately (fail closed).
  - `on_skip(self, ctx: 'StepContext') -> 'None'`
  - `post_grad(self, ctx: 'StepContext', named_params: 'list[tuple[str, torch.nn.Parameter]]') -> 'None'`
  - `pre_write(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', new: 'torch.Tensor') -> 'torch.Tensor | None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
  - `trajectory_tensor(self) -> 'torch.Tensor'` — ``[T+1, F]`` free coefficients after each coefficient update (row 0 = initialisation).
- **class `PriorSubspaceRule`** — Project the candidate update Δ (after Adam/decay) onto or away from span{τ_k[p]} per parameter.
  - `basis(self, name: 'str', device: 'torch.device | None' = None) -> 'torch.Tensor'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
  - `transform_update(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', delta: 'torch.Tensor') -> 'torch.Tensor | None'`
- **class `SpanProjection`** — Least-squares decomposition ``d = Σ_k c_k τ_k + residual``.
- **class `TaskVector`** — τ = θ_expert − θ_base over a set of named parameters (stored in ``dtype``, default fp32).
  - `cosine_similarity(self, other: 'TaskVector') -> 'float'` — Cosine of the angle between the two flattened task vectors (0.0 if either is zero).
  - `dot(self, other: 'TaskVector') -> 'float'`
  - `from_models(cls, expert: 'nn.Module', base: 'nn.Module', names: 'Iterable[str] | None' = None, dtype: 'torch.dtype' = torch.float32) -> 'TaskVector'`
  - `from_state_dict(cls, state: 'Mapping[str, torch.Tensor]') -> 'TaskVector'`
  - `from_state_dicts(cls, expert: 'Mapping[str, torch.Tensor]', base: 'Mapping[str, torch.Tensor]', names: 'Iterable[str] | None' = None, dtype: 'torch.dtype' = torch.float32) -> 'TaskVector'`
  - `items(self) -> 'Iterable[tuple[str, torch.Tensor]]'`
  - `names(self) -> 'list[str]'`
  - `norm(self) -> 'float'` — Global L2 norm over all covered parameters (computed in fp64).
  - `param_cosine_similarity(self, other: 'TaskVector') -> 'dict[str, float]'`
  - `param_norms(self) -> 'dict[str, float]'`
  - `state_dict(self) -> 'dict[str, torch.Tensor]'`
  - `to(self, *args: 'Any', **kwargs: 'Any') -> 'TaskVector'`
- **class `TaskVectorPrior`** — θ(α) = θ_base + Σ_i α_i τ_i over named parameters.
  - `apply_to(self, model: 'nn.Module', coefficients: 'Coefficients') -> 'dict[str, torch.Tensor]'` — Write ``combine(coefficients)`` into ``model`` bit-for-bit; returns the written tensors.
  - `combine(self, coefficients: 'Coefficients') -> 'dict[str, torch.Tensor]'` — Effective parameters θ_base + Σ_i α_i τ_i, returned in the base dtype.
  - `functional_call(self, model: 'nn.Module', coefficients: 'Coefficients', args: 'Any' = (), kwargs: 'Mapping[str, Any] | None' = None) -> 'Any'` — Run ``model`` with θ(α) substituted via ``torch.func.functional_call`` (differentiable in α).
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `project(self, name: 'str', displacement: 'torch.Tensor', rcond: 'float' = 1e-10) -> 'SpanProjection'` — Least-squares projection of a displacement of parameter ``name`` onto span{τ_k[name]}.
  - `register(self, name: 'str', task_vector: 'TaskVector') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `task_matrix(self, name: 'str') -> 'torch.Tensor'` — ``[K, *shape]`` stack of the task vectors at parameter ``name`` (zeros where uncovered).
  - `task_names`
- `default_prior_modules(names: 'Iterable[str]') -> 'list[str]'` — Parameter names of the linear projections q/k/v/o and gate/up/down (``*_proj.weight``).
- `orthonormal_span_basis(basis: 'torch.Tensor', rcond: 'float' = 1e-06) -> 'torch.Tensor'` — Orthonormal basis ``U`` [N, r] of the column span of ``basis`` [N, K] (SVD, rank-revealing).
- `project_onto_span(displacement: 'torch.Tensor', basis: 'Sequence[torch.Tensor] | torch.Tensor', *, rcond: 'float' = 1e-10, dtype: 'torch.dtype' = torch.float64) -> 'SpanProjection'` — Project ``displacement`` onto span{basis_k} by least squares (paper's analysis utility).

## `research_rl.optim.precision`

Low-precision weight updates: dtype checks, stochastic rounding and reproducible rounding RNG.

- `is_low_precision(dtype: 'torch.dtype') -> 'bool'` — True for bf16 and fp16 (the dtypes whose round-to-nearest writes drop small updates).
- `low_precision_trainable(named_params: 'Iterable[tuple[str, torch.Tensor]]') -> 'list[str]'` — Names of parameters with ``requires_grad=True`` stored in bf16/fp16 (in iteration order).
- `round_to(x: 'torch.Tensor', dtype: 'torch.dtype', mode: 'str' = 'nearest', generator: 'torch.Generator | None' = None) -> 'torch.Tensor'` — ``x.to(dtype)`` (``mode="nearest"``, round-to-nearest-even) or :func:`stochastic_round`.
- `rounding_generator(seed: 'int', step: 'int', index: 'int', device: 'torch.device | str' = 'cpu', *, generator: 'torch.Generator | None' = None) -> 'torch.Generator'` — Generator on ``device`` seeded with :func:`rounding_seed_for` ``(seed, step, index)``.
- `rounding_seed_for(seed: 'int', step: 'int', index: 'int') -> 'int'` — A well-mixed 63-bit seed for (seed, optimizer step, parameter index); pure function.
- `stochastic_round(x: 'torch.Tensor', dtype: 'torch.dtype', generator: 'torch.Generator | None' = None) -> 'torch.Tensor'` — Round ``x`` to ``dtype`` stochastically (unbiased); returns a new tensor of ``dtype``.

## `research_rl.optim.update_interface`

The update-rule contract of :class:`~research_rl.optim.optimizer_wrapper.HookedOptimizer`.

- **class `DisplacementLedger`** — Accumulates the *actual* per-parameter displacement norm. Useful for diagnostics/tests.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `post_step(self, ctx: 'StepContext', displacements: 'Mapping[str, torch.Tensor]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
- **class `MaskRule`** — Hard element-wise mask: masked-out elements are bit-identical before and after every step.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `mask_digest(self) -> 'str'`
  - `post_grad(self, ctx: 'StepContext', named_params: 'list[tuple[str, torch.nn.Parameter]]') -> 'None'`
  - `pre_write(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', new: 'torch.Tensor') -> 'torch.Tensor | None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
- **class `StepContext`** — Information handed to every rule hook.
- **class `UpdateRule`** — Base class; override any subset of the hooks.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `on_attach(self, optimizer: 'Any') -> 'None'` — Called once when the rule is added to a HookedOptimizer (validate / bind here).
  - `on_skip(self, ctx: 'StepContext') -> 'None'` — Called instead of the other hooks when the step is skipped.
  - `post_grad(self, ctx: 'StepContext', named_params: 'list[tuple[str, torch.nn.Parameter]]') -> 'None'`
  - `post_step(self, ctx: 'StepContext', displacements: 'Mapping[str, torch.Tensor]') -> 'None'`
  - `pre_write(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', new: 'torch.Tensor') -> 'torch.Tensor | None'` — Return the tensor to write (same dtype as the parameter), or ``None`` to keep ``new``.
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
  - `transform_update(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', delta: 'torch.Tensor') -> 'torch.Tensor | None'` — Return a replacement fp32 Δ, or ``None`` to keep it.

## `research_rl.rewards.code_sandbox`

Code-execution rewards - an interface that is **disabled by default**.

- **class `CodeExecutionVerifier`** — stdout-matching code verifier on top of an isolated :class:`SandboxBackend` (disabled by default).
  - `verify(self, completion: 'str | None', reference: 'Any', **meta: 'Any') -> 'VerificationResult'`
- **class `ExecutionRequest`** — ExecutionRequest(code: 'str', language: 'str', stdin: 'str', time_limit_s: 'float', memory_limit_mb: 'int')
- **class `ExecutionResult`** — ExecutionResult(status: 'ExecStatus', stdout: 'str' = '', stderr: 'str' = '', return_code: 'int | None' = None, details: 'dict[str, Any]' = <factory>)
- **class `IsolationGuarantees`** — What a backend promises about where and how code runs. Defaults promise nothing.
- **class `SandboxBackend`** — An isolated code executor. Implementations must never run code in the caller's process.
  - `guarantees`
  - `run(self, request: 'ExecutionRequest') -> 'ExecutionResult'`
- **class `SandboxDisabledError`** — Code execution was requested but the sandbox is disabled or not safely configured.
- **class `SandboxFusionBackend`** — HTTP client for an external SandboxFusion ``run_code`` endpoint (see module docstring caveats).
  - `guarantees`
  - `payload(self, request: 'ExecutionRequest') -> 'dict[str, Any]'`
  - `run(self, request: 'ExecutionRequest') -> 'ExecutionResult'`
- **class `SandboxPolicy`** — Operator policy for executing generated code. Everything is off / tight by default.
- `check_sandbox(policy: 'SandboxPolicy', backend: 'SandboxBackend | None') -> 'None'` — Raise :class:`SandboxDisabledError` unless ``policy`` + ``backend`` form a safe configuration.
- `extract_code(text: 'str', language: 'str' = 'python') -> 'str | None'` — Body of the code block to run: the last block tagged with ``language`` (```` ```python ```` / ```` ```py ````), else the last bare ```` ``` ```` block, else ``None``.
- `parse_sandbox_fusion_response(resp: 'Any') -> 'ExecutionResult'` — Map a SandboxFusion ``run_code`` JSON response to an :class:`ExecutionResult` (defensively).
- `urllib_transport(url: 'str', body: 'bytes', headers: 'Mapping[str, str]', timeout: 'float') -> 'tuple[int, bytes]'` — Default HTTP POST transport (stdlib only).

## `research_rl.rewards.math_verifier`

Deterministic final-answer verifier for math RLVR (GSM8K / MATH style).

- **class `MathVerifier`** — Final-answer math verifier; see the module docstring for the exact pipeline.
  - `canonical(self, normalized: 'str') -> 'Fraction | str'`
  - `equivalent(self, answer: 'str', reference: 'str') -> 'bool'` — Equivalence of two already-normalised answers (exact rational, else exact string).
  - `normalize(self, answer: 'str') -> 'str'`
  - `reference_answer(self, reference: 'Any') -> 'str'` — Normalised reference. A raw GSM8K solution (``... #### 72``) or a ``\boxed{}`` solution is reduced to its final answer; ints / floats / Fractions are stringified.
  - `strip_terminators(self, text: 'str') -> 'tuple[str, bool]'` — Cut ``text`` at the first terminator; returns ``(text, terminated)``.
- `normalize_answer(answer: 'str', *, strip_percent: 'bool' = True, strip_degrees: 'bool' = True) -> 'str'` — Canonical string form of a final answer.
- `parse_number(s: 'str', *, max_digits: 'int' = 1000, max_exponent: 'int' = 400) -> 'Fraction | None'` — Exact rational value of a normalised answer, or ``None`` if it is not a plain number.

## `research_rl.rewards.verifier_interface`

Verifier contract for verifiable rewards (RLVR) and the adapter to TRL reward functions.

- **class `BaseVerifier`** — Convenience base: subclasses implement :meth:`_verify` returning ``(status, parsed, reference, details)``.
  - `make_result(self, status: 'Status', parsed: 'str | None' = None, reference: 'str | None' = None, details: 'dict[str, Any] | None' = None) -> 'VerificationResult'`
  - `verify(self, completion: 'str | None', reference: 'Any', **meta: 'Any') -> 'VerificationResult'`
- **class `FatalVerifierError`** — An error that must abort reward computation instead of being scored as ``status="error"``.
- **class `RewardScheme`** — Reward assigned to each verification status (defaults: binary 1/0 RLVR reward).
  - `reward_for(self, status: 'Status') -> 'float'`
- **class `TRLRewardFunction`** — Picklable TRL reward callable built by :func:`as_trl_reward_func` (``__name__`` drives TRL logging).
- **class `VerificationResult`** — Outcome of verifying one completion.
- **class `Verifier`** — Anything with ``verify(completion, reference, **meta) -> VerificationResult``.
  - `verify(self, completion: 'str | None', reference: 'Any', **meta: 'Any') -> 'VerificationResult'`
- `as_trl_reward_func(verifier: 'Verifier', answer_key: 'str' = 'answer', name: 'str | None' = None, *, eos_token_ids: 'Sequence[int] | None' = None, truncated_key: 'str | None' = None, none_on_error: 'bool' = False, log_status_metrics: 'bool' = True, max_error_frac: 'float | None' = None) -> 'TRLRewardFunction'` — Wrap ``verifier`` as a TRL ``reward_funcs`` entry.
- `completion_to_text(completion: 'Any') -> 'str | None'` — Text of a TRL completion.

## `research_rl.rollout.agent_env`

Agent–environment contract compatible with TRL ``GRPOTrainer(environment_factory=...)``.

- **class `CalculatorEnv`** — Deterministic toy: reach ``target`` using the ``add`` / ``multiply`` tools.
  - `add(self, a: 'float', b: 'float') -> 'float'` — Add two numbers.
  - `multiply(self, a: 'float', b: 'float') -> 'float'` — Multiply two numbers.
- **class `EpisodeLog`** — Turns of one episode with per-turn rewards and an optional terminal reward (added to the last turn).
  - `add(self, action: 'str', observation: 'str' = '', reward: 'float' = 0.0, **info: 'Any') -> 'Turn'`
  - `advantages(self, values: 'Sequence[float] | torch.Tensor', gamma: 'float' = 1.0, lam: 'float' = 0.95, last_value: 'float' = 0.0) -> 'torch.Tensor'`
  - `discounted_returns(self, gamma: 'float' = 1.0) -> 'torch.Tensor'`
  - `rewards(self) -> 'torch.Tensor'` — Per-turn rewards ``[T]`` (terminal reward added to the last turn; a lone terminal reward if T=0).
  - `set_final_reward(self, reward: 'float') -> 'None'`
  - `to_dict(self) -> 'dict[str, Any]'`
  - `total_reward(self) -> 'float'`
  - `turn_credit(self, gamma: 'float' = 1.0, baseline: 'float | Sequence[float] | torch.Tensor' = 0.0) -> 'torch.Tensor'` — Per-turn credit G_t − b_t (``baseline`` scalar or per turn).
- **class `ResearchEnvironment`** — Base for TRL ``environment_factory`` environments.
  - `episode`
  - `get_reward(self) -> 'float'` — Total episode reward (per-turn rewards + terminal reward); called once per rollout by TRL.
  - `num_resets`
  - `reset(self, **kwargs: 'Any') -> 'str | None'` — Start a new episode. ``kwargs`` is the dataset row TRL passes (``prompt`` and other columns).
- **class `Turn`** — Turn(action: 'str', observation: 'str', reward: 'float', info: 'dict[str, Any]' = <factory>)
- `discounted_returns(rewards: 'Sequence[float] | torch.Tensor', gamma: 'float') -> 'torch.Tensor'` — G_t = r_t + γ G_{t+1} (G_T = 0), float64.
- `generalized_advantages(rewards: 'Sequence[float] | torch.Tensor', values: 'Sequence[float] | torch.Tensor', gamma: 'float', lam: 'float', last_value: 'float' = 0.0) -> 'torch.Tensor'` — GAE: δ_t = r_t + γ V_{t+1} − V_t (V_T = ``last_value``), A_t = Σ_l (γλ)^l δ_{t+l}.
- `spread_to_tokens(per_turn: 'Sequence[float] | torch.Tensor', token_counts: 'Sequence[int]') -> 'torch.Tensor'` — Repeat each turn's value over that turn's tokens (``[Σ token_counts]``).

## `research_rl.rollout.policy_version`

Policy versions: which parameters generated a rollout.

- `policy_lag(current_version: 'int', batch: 'Mapping[str, Any]') -> 'int | None'` — Number of applied updates between generating ``batch`` and now (``None`` if unstamped).
- `policy_version_of(obj: 'Any') -> 'int'` — Current policy version of a trainer or optimizer.
- `read_policy_version(batch: 'Mapping[str, Any]') -> 'int | None'` — The stamped version of ``batch`` (``None`` if it was never stamped).
- `stamp_policy_version(batch: 'MutableMapping[str, Any]', version: 'int') -> 'MutableMapping[str, Any]'` — Store ``version`` in ``batch`` (in place) as a 0-dim ``long`` tensor; returns ``batch``.

## `research_rl.rollout.trajectory`

Rollout bookkeeping: what was generated, how it was scored, and by which policy version.

- **class `RolloutBatch`** — One generation batch (the local, per-process part) together with its scores.
  - `frac_zero_advantage` — Fraction of samples without learning signal (see :meth:`_per_sample_zero_adv`).
  - `from_tensors(cls, *, prompt_ids: 'torch.Tensor', prompt_mask: 'torch.Tensor', completion_ids: 'torch.Tensor', completion_mask: 'torch.Tensor', rewards: 'torch.Tensor', advantages: 'torch.Tensor', policy_version: 'int', global_step: 'int' = 0, mode: 'str' = 'train', signal: 'torch.Tensor | None' = None, expert_ids: 'Sequence[str] | None' = None) -> 'RolloutBatch'` — Build a batch from (possibly GPU, possibly grad-carrying) tensors; stores detached CPU copies.
  - `has_nonzero_advantage` — Does any sample carry a learning signal?
  - `mean_reward` — Mean over scorable samples (NaN if none is scorable).
  - `num_loss_tokens`
  - `stats(self) -> 'dict[str, float | int | bool]'`
  - `trajectories(self) -> 'list[Trajectory]'` — Unbatch into :class:`Trajectory` objects (prompt left-padding removed via ``prompt_mask``).
- **class `Trajectory`** — A single sample: prompt tokens (padding removed), completion row and its mask.
  - `num_loss_tokens`
- **class `TrajectoryBuffer`** — Bounded log of recent rollout batches plus running counters (the counters are checkpointed).
  - `append(self, batch: 'RolloutBatch') -> 'None'`
  - `clear(self) -> 'None'`
  - `counters(self) -> 'dict[str, Any]'` — Running counters (JSON- and ``weights_only``-safe: ints, ``None`` and a ``{str: int}`` dict).
  - `latest(self) -> 'RolloutBatch | None'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `policy_versions(self) -> 'list[int]'`
  - `record_window(self, *, has_signal: 'bool', skipped: 'bool') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `stats_history(self) -> 'list[dict[str, float | int | bool]]'`

## `research_rl.state.compute_controller`

Adaptive-compute contract for looped / recurrent-depth models (AIC, Best-of-Depth, ...).

- **class `BestOfDepthResult`** — BestOfDepthResult(output: 'torch.Tensor', best_depth: 'torch.Tensor', scores: 'torch.Tensor', entry: 'LedgerEntry | None' = None)
- **class `ComputeController`** — Base class. Override :meth:`halt`; :meth:`start` is called once per inference call.
  - `halt(self, depth: 'int', previous: 'torch.Tensor', current: 'torch.Tensor') -> 'torch.Tensor'` — Bool ``[rows]``: True = stop after ``depth`` applications (rows = currently active rows).
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `start(self, batch_size: 'int', device: 'torch.device') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
- **class `ComputeLedger`** — Records actual iteration depth and estimated cost per call and per sample.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `mean_depth(self) -> 'float'`
  - `per_sample(self) -> 'dict[str, dict[str, float]]'` — ``{sample_id: {"iterations": Σ depth, "cost": Σ cost, "calls": #calls}}`` over all calls.
  - `record(self, depths: 'torch.Tensor | Sequence[int]', cost_per_iteration: 'float | Sequence[float] | torch.Tensor' = 1.0, *, call_id: 'str | None' = None, sample_ids: 'Sequence[Hashable] | None' = None, kind: 'str' = 'looped') -> 'LedgerEntry'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `total_cost(self) -> 'float'`
  - `total_iterations(self) -> 'int'`
- **class `FixedDepth`** — Every sample receives exactly ``depth`` iterations (capped by ``max_depth``).
  - `halt(self, depth: 'int', previous: 'torch.Tensor', current: 'torch.Tensor') -> 'torch.Tensor'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
- **class `LedgerEntry`** — LedgerEntry(call_id: 'str', kind: 'str', sample_ids: 'list[str]', depths: 'list[int]', cost_per_iteration: 'list[float]', costs: 'list[float]')
  - `to_dict(self) -> 'dict[str, Any]'`
  - `total_cost`
  - `total_iterations`
- **class `LoopResult`** — LoopResult(output: 'torch.Tensor', depths: 'torch.Tensor', halted_by_controller: 'torch.Tensor', entry: 'LedgerEntry | None' = None)
- **class `ThresholdHalting`** — Halt a row once ‖h_t − h_{t−1}‖₂ (over all non-batch dims) < ``eps`` and depth ≥ ``min_depth``.
  - `halt(self, depth: 'int', previous: 'torch.Tensor', current: 'torch.Tensor') -> 'torch.Tensor'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
- `best_of_depth(outputs_per_depth: 'torch.Tensor | Sequence[torch.Tensor]', scorer: 'Callable[[torch.Tensor], torch.Tensor]', *, ledger: 'ComputeLedger | None' = None, cost_per_iteration: 'float | Sequence[float] | torch.Tensor' = 1.0, call_id: 'str | None' = None, sample_ids: 'Sequence[Hashable] | None' = None, iterations_computed: 'torch.Tensor | Sequence[int] | None' = None) -> 'BestOfDepthResult'` — Pick, per sample, the depth whose output scores highest (ties → earliest depth).
- `collect_depth_outputs(block: 'Callable[[torch.Tensor], torch.Tensor]', h0: 'torch.Tensor', max_depth: 'int') -> 'torch.Tensor'` — ``[D, B, ...]`` stack of h_1..h_D (all rows run the full ``max_depth``).
- `run_looped(block: 'Callable[[torch.Tensor], torch.Tensor]', h0: 'torch.Tensor', controller: 'ComputeController', max_depth: 'int', *, ledger: 'ComputeLedger | None' = None, cost_per_iteration: 'float | Sequence[float] | torch.Tensor' = 1.0, call_id: 'str | None' = None, sample_ids: 'Sequence[Hashable] | None' = None) -> 'LoopResult'` — Apply the weight-shared ``block`` repeatedly with per-sample halting (see module docstring).

## `research_rl.state.recurrent_interface`

Per-sample recurrent state contract (memory carried across steps / turns / chunks).

- **class `FastSlowMemory`** — Two-timescale memory. Per row, with input x_t ∈ R^d and step counter n::
  - `init(self, batch_size: 'int', device: 'torch.device | str | None' = None, dtype: 'torch.dtype' = torch.float32) -> 'State'` — Fresh state: ``fast`` in ``dtype``, ``slow`` in :meth:`slow_storage_dtype`, ``steps`` long.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `read(self, state: 'State') -> 'torch.Tensor'` — ``concat(fast, slow)`` in ``promote_types(fast dtype, slow dtype)``.
  - `slow_storage_dtype(self, dtype: 'torch.dtype') -> 'torch.dtype'` — Dtype in which the slow state is stored for a requested state ``dtype``.
  - `state_dict(self) -> 'dict[str, Any]'`
  - `state_dtype(self, state: 'State') -> 'torch.dtype'`
  - `update(self, state: 'State', inputs: 'torch.Tensor', active: 'torch.Tensor | None' = None) -> 'State'`
- **class `RecurrentState`** — Abstract per-sample recurrent state (see module docstring for the contract).
  - `init(self, batch_size: 'int', device: 'torch.device | str | None' = None, dtype: 'torch.dtype' = torch.float32) -> 'State'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `read(self, state: 'State') -> 'torch.Tensor'`
  - `reset(self, state: 'State', row_mask: 'torch.Tensor') -> 'State'` — Re-initialise rows where ``row_mask`` is True; other rows are returned unchanged.
  - `state_dict(self) -> 'dict[str, Any]'`
  - `state_dtype(self, state: 'State') -> 'torch.dtype'` — The dtype that was requested from :meth:`init` for ``state`` (used by :meth:`reset`).
  - `update(self, state: 'State', inputs: 'torch.Tensor', active: 'torch.Tensor | None' = None) -> 'State'`
- **class `StateStore`** — Per-sample-id storage of :class:`RecurrentState` rows (no implicit cross-sample sharing).
  - `forget(self, ids: 'Sequence[Hashable]') -> 'None'` — Drop the state of ``ids`` (they start fresh next time).
  - `gather(self, ids: 'Sequence[Hashable]') -> 'State'` — Batch the states of ``ids`` (fresh state for unseen ids). Must be followed by ``scatter``.
  - `get(self, sample_id: 'Hashable') -> 'State'`
  - `ids(self) -> 'list[int | str]'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `release(self) -> 'None'` — Abandon the gathered batch: nothing is stored, and ids first seen in that ``gather`` stay unknown.
  - `scatter(self, ids: 'Sequence[Hashable]', state: 'State') -> 'None'` — Store row i of ``state`` under ``ids[i]``; ids must equal those of the last ``gather``.
  - `state_dict(self) -> 'dict[str, Any]'`
  - `step(self, ids: 'Sequence[Hashable]', inputs: 'torch.Tensor') -> 'torch.Tensor'` — gather → update → scatter; returns the readout after the update.
- `batch_size_of(state: 'State') -> 'int'`

## `research_rl.trainers.base`

Glue between ResearchRL and TRL trainers.

- **class `ResearchCallback`** — Feeds the experiment recorder and writes ResearchRL extras into each checkpoint.
  - `bind(self, trainer: 'Any') -> 'None'`
  - `on_log(self, args: 'Any', state: 'Any', control: 'Any', logs: 'dict[str, Any] | None' = None, **kw: 'Any') -> 'None'`
  - `on_save(self, args: 'Any', state: 'Any', control: 'Any', **kw: 'Any') -> 'None'`
  - `on_train_begin(self, args: 'Any', state: 'Any', control: 'Any', **kw: 'Any') -> 'None'`
  - `on_train_end(self, args: 'Any', state: 'Any', control: 'Any', **kw: 'Any') -> 'None'`
- **class `ResearchRun`** — A configured run: the TRL trainer plus its recorder.
  - `hooked_optimizer`
  - `train(self, resume_from_checkpoint: 'str | bool | None' = None) -> 'dict[str, Any]'`
- **class `ResearchTrainerMixin`** — Mix *before* a TRL trainer class: ``class X(ResearchTrainerMixin, GRPOTrainer)``.
  - `create_optimizer(self, model: 'Any' = None) -> 'Any'`
  - `hooked_optimizer`
  - `train(self, resume_from_checkpoint: 'str | bool | None' = None, **kwargs: 'Any') -> 'Any'`
- **class `TrainerAdapter`** — Base adapter. Subclasses set the class variables and may override ``trainer_kwargs``.
  - `build(self, exp: 'ExperimentConfig', *, train_dataset: 'Any', model: 'PreTrainedModel | str | None' = None, processing_class: 'Any' = None, eval_dataset: 'Any' = None, update_rules: 'Sequence[UpdateRule]' = (), components: 'Mapping[str, StatefulComponent] | None' = None, callbacks: 'Sequence[TrainerCallback]' = (), record: 'bool' = True, **algo_kwargs: 'Any') -> 'ResearchRun'`
  - `build_trl_config(self, exp: 'ExperimentConfig') -> 'Any'`
  - `trainer_class(self) -> 'type'`
  - `trainer_kwargs(self, exp: 'ExperimentConfig', **algo_kwargs: 'Any') -> 'dict[str, Any]'` — Algorithm-specific keyword arguments for the TRL trainer constructor.
- `load_causal_lm(spec: 'ModelSpec') -> 'PreTrainedModel'`
- `load_tokenizer(spec: 'ModelSpec') -> 'Any'`
- `precision_info(model: 'torch.nn.Module', optim: 'OptimSpec') -> 'dict[str, Any]'` — ``{"param_dtypes": {dtype: #trainable tensors}, "master_weights", "write_rounding"}`` for the record.
- `research_trainer_class(trl_cls: 'type') -> 'type'` — Return ``ResearchTrainerMixin`` combined with ``trl_cls`` (cached).
- `resolve_dtype(name: 'str') -> 'torch.dtype | str'`

## `research_rl.trainers.distillation_adapter`

On-policy knowledge distillation adapter around TRL's ``DistillationTrainer``.

- **class `DistillationAdapter`** — ``TRAINERS.build("distillation")`` -> adapter for on-policy distillation.
  - `build(self, exp: 'ExperimentConfig', **kwargs: 'Any') -> 'ResearchRun'`
  - `trainer_kwargs(self, exp: 'ExperimentConfig', **algo_kwargs: 'Any') -> 'dict[str, Any]'`
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 1e-06, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool | None = False, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, teacher_model_name_or_path: str | None = None, teacher_model_revision: str | None = None, teacher_model_init_kwargs: dict[str, typing.Any] | str | None = None, disable_dropout: bool = False, max_completion_length: int | None = 512, ds3_gather_for_generation: bool = True, shuffle_dataset: bool | None = True, pad_to_multiple_of: int | str | None = 'auto', temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float | None = None, generation_kwargs: dict | str | None = None, chat_template_kwargs: dict | str | None = None, repetition_penalty: float = 1.0, cache_implementation: str | None = None, use_vllm: bool = False, vllm_mode: str = 'colocate', vllm_model_impl: str = 'vllm', vllm_enable_sleep_mode: bool = False, vllm_structured_outputs_regex: str | None = None, vllm_server_base_url: str | None = None, vllm_server_host: str = '0.0.0.0', vllm_server_port: int = 8000, vllm_server_timeout: float = 240.0, vllm_group_port: int = 51216, vllm_gpu_memory_utilization: float = 0.3, vllm_max_model_length: int | None = None, vllm_tensor_parallel_size: int = 1, beta: float = 1.0, max_tool_calling_iterations: int | None = None, log_completions: bool = False, num_completions_to_print: int | None = None, log_unique_prompts: bool = False) -> None` — Configuration class for the [`DistillationTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', teacher_model: str | transformers.modeling_utils.PreTrainedModel | None = None, args: trl.trainer.distillation_config.DistillationConfig | None = None, train_dataset: datasets.arrow_dataset.Dataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | dict[str, datasets.arrow_dataset.Dataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None, tools: list[collections.abc.Callable] | None = None)` — Trainer for knowledge distillation. The student is trained on-policy — it generates the completions itself — to match the teacher's next-token distribution under a generalized Jensen-Shannon divergence (interpolating forward KL, reverse KL, and JSD via `beta`), as introduced in [On-Policy Distillation of Language Models](https://huggingface.co/papers/2306.13649).

## `research_rl.trainers.dpo_adapter`

Direct Preference Optimisation (DPO) adapter around TRL's ``DPOTrainer``.

- **class `DPOAdapter`** — ``TRAINERS.build("dpo")`` -> adapter for offline preference optimisation with ``DPOTrainer``.
  - `build(self, exp: 'ExperimentConfig', *, train_dataset: 'Any', model: 'PreTrainedModel | str | None' = None, ref_model: 'PreTrainedModel | str | None' = None, **kwargs: 'Any') -> 'ResearchRun'`
  - `trainer_class(self) -> 'type'`
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 1e-06, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool = True, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, router_aux_loss_coef: float | None = None, disable_dropout: bool = True, dataset_num_proc: int | None = None, max_length: int | None = 1024, truncation_mode: str = 'keep_start', padding_free: bool = False, pad_to_multiple_of: int | str | None = 'auto', precompute_ref_log_probs: bool = False, precompute_ref_batch_size: int | None = None, loss_type: list[str] = <factory>, loss_weights: list[float] | None = None, ld_alpha: float | None = None, f_divergence_type: str = 'reverse_kl', f_alpha_divergence_coef: float = 0.5, label_smoothing: float = 0.0, beta: float = 0.1, use_weighting: bool = False, discopop_tau: float = 0.05, activation_offloading: bool = False, sync_ref_model: bool = False, ref_model_mixup_alpha: float = 0.6, ref_model_sync_steps: int = 512, pad_token: str | None = None) -> None` — Configuration class for the [`DPOTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', ref_model: transformers.modeling_utils.PreTrainedModel | None = None, args: trl.trainer.dpo_config.DPOConfig | None = None, data_collator: collections.abc.Callable[[list[typing.Any]], dict[str, typing.Any]] | None = None, train_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | datasets.dataset_dict.DatasetDict | datasets.dataset_dict.IterableDatasetDict | dict[str, datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, compute_metrics: collections.abc.Callable[[transformers.trainer_utils.EvalPrediction], dict] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None)` — Trainer for Direct Preference Optimization (DPO) method. This algorithm was initially proposed in the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290). This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.
- `frozen_copy(model: 'PreTrainedModel') -> 'PreTrainedModel'` — Deep copy with gradients disabled and eval mode (used as a fixed reference policy).

## `research_rl.trainers.grpo_adapter`

GRPO adapter on top of TRL's ``GRPOTrainer`` (reinforcement learning from verifiable/learned rewards).

- **class `GRPOAdapter`** — ``TRAINERS.build("grpo")``: maps an ``ExperimentConfig`` onto :class:`ResearchGRPOTrainer`.
  - `research_trainer_cls(*args: 'Any', skip_zero_advantage_steps: 'bool' = False, trajectory_capacity: 'int' = 16, advantage_shapers: 'Sequence[AdvantageShaper | str]' = (), expert_manager: 'ExpertManager | None' = None, expert_column: 'str' = 'expert', **kwargs: 'Any') -> 'None'` — ``GRPOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip.
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 1e-06, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool | None = False, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, router_aux_loss_coef: float | None = None, disable_dropout: bool = False, cast_lm_head_to_fp32: bool = False, num_generations: int | None = 8, num_generations_eval: int | None = None, max_completion_length: int | None = 512, ds3_gather_for_generation: bool = True, shuffle_dataset: bool | None = True, pad_to_multiple_of: int | str | None = 'auto', generation_batch_size: int | None = None, steps_per_generation: int | None = None, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float | None = None, generation_kwargs: dict | str | None = None, chat_template_kwargs: dict | str | None = None, repetition_penalty: float = 1.0, cache_implementation: str | None = None, use_vllm: bool = False, vllm_mode: str = 'colocate', vllm_model_impl: str = 'vllm', vllm_enable_sleep_mode: bool = False, vllm_structured_outputs_regex: str | None = None, vllm_server_base_url: str | None = None, vllm_server_host: str = '0.0.0.0', vllm_server_port: int = 8000, vllm_server_timeout: float = 240.0, vllm_group_port: int = 51216, vllm_gpu_memory_utilization: float = 0.3, vllm_max_model_length: int | None = None, vllm_tensor_parallel_size: int = 1, beta: float = 0.0, num_iterations: int = 1, epsilon: float = 0.2, delta: float | None = None, epsilon_high: float | None = None, sapo_temperature_neg: float = 1.05, sapo_temperature_pos: float = 1.0, vespo_k_pos: float = 2.0, vespo_lambda_pos: float = 3.0, vespo_k_neg: float = 3.0, vespo_lambda_neg: float = 2.0, importance_sampling_level: str = 'token', reward_weights: list[float] | None = None, multi_objective_aggregation: str = 'sum_then_normalize', scale_rewards: str = 'group', loss_type: str = 'dapo', mask_truncated_completions: bool = False, sync_ref_model: bool = False, ref_model_mixup_alpha: float = 0.6, ref_model_sync_steps: int = 512, top_entropy_quantile: float = 1.0, entropy_coef: float = 0.0, use_adaptive_entropy: bool = False, entropy_coef_min: float = 0.0, entropy_coef_max: float = 1.0, entropy_coef_delta: float = 0.005, entropy_target: float = 0.2, max_tool_calling_iterations: int | None = None, vllm_importance_sampling_correction: bool = True, vllm_importance_sampling_mode: str = 'sequence_mask', vllm_importance_sampling_clip_max: float | None = 3.0, vllm_importance_sampling_clip_min: float | None = None, off_policy_mask_threshold: float | None = None, use_bias_correction_kl: bool = True, log_completions: bool = False, log_multimodal: bool = True, num_completions_to_print: int | None = None, log_unique_prompts: bool = False, log_completions_hub_repo: str | None = None, use_transformers_continuous_batching: bool = False, transformers_continuous_batching_config: dict | str | None = None, use_transformers_paged: bool = False, vllm_importance_sampling_cap: float | None = None) -> None` — Configuration class for the [`GRPOTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', reward_funcs: str | transformers.modeling_utils.PreTrainedModel | collections.abc.Callable[..., list[float | None]] | list[str | transformers.modeling_utils.PreTrainedModel | collections.abc.Callable[..., list[float | None]]] | None = None, args: trl.trainer.grpo_config.GRPOConfig | None = None, train_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | datasets.dataset_dict.DatasetDict | datasets.dataset_dict.IterableDatasetDict | dict[str, datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, reward_processing_classes: transformers.tokenization_utils_base.PreTrainedTokenizerBase | list[transformers.tokenization_utils_base.PreTrainedTokenizerBase] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None, tools: list[collections.abc.Callable] | None = None, rollout_func: collections.abc.Callable[[list[str], 'GRPOTrainer'], dict[str, typing.Any]] | None = None, environment_factory: collections.abc.Callable[[], trl.trainer.grpo_trainer._SupportsReset] | dict[str, collections.abc.Callable[[], trl.trainer.grpo_trainer._SupportsReset]] | None = None)` — Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).
- **class `ResearchGRPOTrainer`** — ``GRPOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip.

## `research_rl.trainers.multi_teacher`

Multi-teacher on-policy distillation building blocks.

- **class `MultiTeacherBatch`** — Student completions with K teachers' next-token log-probs.
  - `aggregated(self, mode: 'str' = 'mixture') -> 'torch.Tensor'`
  - `forward_kl(self, student_logprobs: 'torch.Tensor', mode: 'str' = 'mixture') -> 'torch.Tensor'`
  - `num_teachers`
  - `reverse_kl(self, student_logprobs: 'torch.Tensor', mode: 'str' = 'mixture') -> 'torch.Tensor'`
  - `teacher_token_logprobs(self) -> 'torch.Tensor'` — ``[K, B, T]`` log-prob each teacher assigns to the sampled completion tokens.
- **class `TeacherEnsemble`** — Several frozen HF causal LMs (shared vocabulary) acting as one teacher with static weights.
  - `base_model`
  - `forward(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor | None' = None, **kwargs: 'Any') -> 'CausalLMOutputWithPast'`
  - `from_specs(cls, specs: 'Sequence[TeacherSpec]', mode: 'str' = 'mixture', dtype: 'torch.dtype | str | None' = None) -> 'TeacherEnsemble'`
  - `get_input_embeddings(self) -> 'nn.Module'`
  - `get_output_embeddings(self) -> '_EnsembleHead'`
  - `teacher_logprobs(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor | None' = None, **kwargs: 'Any') -> 'torch.Tensor'` — ``[K, B, L, V]`` fp32 log-probs of every teacher.
  - `trl_hidden_states(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor | None' = None, logits_to_keep: 'int | None' = None, **kwargs: 'Any') -> 'torch.Tensor'` — Hidden states that, through :meth:`get_output_embeddings`, give the aggregated logits.
- **class `TeacherSpec`** — A teacher: a model id/path or an instantiated causal LM, plus its aggregation weight.
  - `load(self, dtype: 'torch.dtype | str | None' = None) -> 'nn.Module'`
- `aggregate_teacher_logprobs(teacher_logprobs: 'torch.Tensor', weights: 'torch.Tensor | Sequence[float]', mode: 'str' = 'mixture', *, normalize: 'bool' = True) -> 'torch.Tensor'` — Combine K teachers' log-probabilities ``[K, ..., V]`` into one log-distribution ``[..., V]``.
- `masked_forward_kl(teacher_logprobs: 'torch.Tensor', student_logprobs: 'torch.Tensor', mask: 'torch.Tensor | None' = None, reduction: 'str' = 'mean') -> 'torch.Tensor'` — KL(p_teacher ‖ p_student) per token over the last (vocab) dim; ``mean`` averages valid tokens.
- `masked_reverse_kl(student_logprobs: 'torch.Tensor', teacher_logprobs: 'torch.Tensor', mask: 'torch.Tensor | None' = None, reduction: 'str' = 'mean') -> 'torch.Tensor'` — KL(p_student ‖ p_teacher) per token (the on-policy distillation objective).
- `normalize_weights(weights: 'torch.Tensor | Sequence[float]') -> 'torch.Tensor'` — Non-negative weights normalised to sum to 1 along dim 0 (the teacher axis), in float64.

## `research_rl.trainers.rl_common`

Shared machinery of the RL adapters (GRPO, RLOO).

- **class `AdvantageShaper`** — Rewrites the advantages of a generation batch (turn/step credit, sign rebalancing, per-token KL rewards, ...). See the module docstring for where it runs and how its ``signal`` is used.
- **class `RLRolloutMixin`** — Policy-version stamps, trajectory log, advantage shaping, zero-advantage skipping and expert routing for TRL GRPO/RLOO trainers.
  - `compute_loss(self, model: 'Any', inputs: 'Any', *args: 'Any', **kwargs: 'Any') -> 'Any'` — TRL's loss with the micro-batch's rows routed through their experts (from :data:`EXPERT_KEY`).
  - `expert_routing(self, batch: 'Mapping[str, Any]') -> 'Iterator[None]'` — Select the experts of ``batch``'s rows (a generation batch or micro-batch of this trainer, including the ``output`` an :class:`AdvantageShaper` receives).
  - `policy_version` — Applied parameter updates so far (see :func:`research_rl.rollout.policy_version_of`).
  - `training_step(self, model: 'Any', inputs: 'Any', *args: 'Any', **kwargs: 'Any') -> 'Any'` — Keeps the micro-batch's expert selection active through ``compute_loss`` *and* backward.
- **class `RLTrainerAdapter`** — Base for RL adapters: ``trainer_class()`` returns :attr:`research_trainer_cls`.
  - `build(self, exp: 'ExperimentConfig', **kwargs: 'Any') -> 'ResearchRun'`
  - `trainer_class(self) -> 'type'`
  - `trainer_kwargs(self, exp: 'ExperimentConfig', **algo_kwargs: 'Any') -> 'dict[str, Any]'`
- **class `RolloutRecordCallback`** — Copies the trajectory-buffer counters into ``record.json`` under ``"rollout"``.
  - `on_step_end(self, args: 'Any', state: 'Any', control: 'Any', **kwargs: 'Any') -> 'None'`
- **class `ShapedAdvantages`** — Result of an :class:`AdvantageShaper`.
- `eos_cut_length(ids: 'Sequence[int]', eos_token_ids: 'Sequence[int]', pad_token_id: 'int | None') -> 'int'` — Length of ``ids`` after cutting at the first EOS (kept) or stripping trailing pad tokens.
- `eos_truncating_rollout(rollout_func: 'RolloutFunc', eos_token_ids: 'Sequence[int] | None' = None, pad_token_id: 'int | None' = None) -> 'RolloutFunc'` — Wrap a GRPO ``rollout_func`` so each completion ends at its first EOS (see module docstring).
- `group_deviation_signal(rewards: 'torch.Tensor', num_generations: 'int') -> 'torch.Tensor'` — Per sample: does its reward differ from the mean of its group? (NaN = unscorable -> ``False``).
- `resume_is_generation_aligned(global_step: 'int', gradient_accumulation_steps: 'int', steps_per_generation: 'int', num_iterations: 'int') -> 'bool'` — Is ``global_step`` a generation boundary of TRL's GRPO/RLOO buffering?

## `research_rl.trainers.rloo_adapter`

RLOO adapter on top of TRL's ``RLOOTrainer`` (REINFORCE with a leave-one-out baseline).

- **class `RLOOAdapter`** — ``TRAINERS.build("rloo")``: maps an ``ExperimentConfig`` onto :class:`ResearchRLOOTrainer`.
  - `research_trainer_cls(*args: 'Any', skip_zero_advantage_steps: 'bool' = False, trajectory_capacity: 'int' = 16, advantage_shapers: 'Sequence[AdvantageShaper | str]' = (), expert_manager: 'ExpertManager | None' = None, expert_column: 'str' = 'expert', **kwargs: 'Any') -> 'None'` — ``RLOOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip.
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 1e-06, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool | None = False, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, router_aux_loss_coef: float | None = None, disable_dropout: bool = False, num_generations: int | None = 2, num_generations_eval: int | None = None, max_completion_length: int | None = 512, ds3_gather_for_generation: bool = True, shuffle_dataset: bool | None = True, pad_to_multiple_of: int | str | None = 'auto', generation_batch_size: int | None = None, steps_per_generation: int | None = None, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float | None = None, generation_kwargs: dict | str | None = None, chat_template_kwargs: dict | str | None = None, repetition_penalty: float = 1.0, cache_implementation: str | None = None, use_vllm: bool = False, vllm_mode: str = 'colocate', vllm_model_impl: str = 'vllm', vllm_enable_sleep_mode: bool = False, vllm_structured_outputs_regex: str | None = None, vllm_server_base_url: str | None = None, vllm_server_host: str = '0.0.0.0', vllm_server_port: int = 8000, vllm_server_timeout: float = 240.0, vllm_group_port: int = 51216, vllm_gpu_memory_utilization: float = 0.3, vllm_max_model_length: int | None = None, vllm_tensor_parallel_size: int = 1, beta: float = 0.05, num_iterations: int = 1, epsilon: float = 0.2, epsilon_high: float | None = None, reward_weights: list[float] | None = None, normalize_advantages: bool = False, reward_clip_range: tuple[float, float] | None = None, mask_truncated_completions: bool = False, sync_ref_model: bool = False, ref_model_mixup_alpha: float = 0.6, ref_model_sync_steps: int = 512, log_completions: bool = False, log_multimodal: bool = True, num_completions_to_print: int | None = None, log_unique_prompts: bool = False, use_transformers_continuous_batching: bool = False, transformers_continuous_batching_config: dict | str | None = None, use_transformers_paged: bool = False) -> None` — Configuration class for the [`RLOOTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', reward_funcs: str | transformers.modeling_utils.PreTrainedModel | collections.abc.Callable[..., list[float | None]] | list[str | transformers.modeling_utils.PreTrainedModel | collections.abc.Callable[..., list[float | None]]], args: trl.trainer.rloo_config.RLOOConfig | None = None, train_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | datasets.dataset_dict.DatasetDict | datasets.dataset_dict.IterableDatasetDict | dict[str, datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, reward_processing_classes: transformers.tokenization_utils_base.PreTrainedTokenizerBase | list[transformers.tokenization_utils_base.PreTrainedTokenizerBase] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None)` — Trainer for the Reinforce Leave One Out (RLOO) method. This algorithm was initially proposed in the paper [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://huggingface.co/papers/2402.14740).
- **class `ResearchRLOOTrainer`** — ``RLOOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip.

## `research_rl.trainers.sft_adapter`

Supervised fine-tuning (SFT) adapter around TRL's ``SFTTrainer``.

- **class `SFTAdapter`** — ``TRAINERS.build("sft")`` -> adapter mapping an ``ExperimentConfig`` onto ``SFTTrainer``.
  - `build(self, exp: 'ExperimentConfig', **kwargs: 'Any') -> 'ResearchRun'`
  - `trainer_class(self) -> 'type'`
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 2e-05, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool = True, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, router_aux_loss_coef: float | None = None, chat_template_path: str | None = None, dataset_text_field: str = 'text', dataset_kwargs: dict[str, typing.Any] | str | None = None, dataset_num_proc: int | None = None, eos_token: str | None = None, max_length: int | None = 1024, truncation_mode: str = 'keep_start', shuffle_dataset: bool = False, packing: bool = False, packing_strategy: str = 'bfd', padding_free: bool = False, pad_to_multiple_of: int | str | None = 'auto', eval_packing: bool | None = None, completion_only_loss: bool | None = None, assistant_only_loss: bool = False, loss_type: str = 'nll', activation_offloading: bool = False, pad_token: str | None = None) -> None` — Configuration class for the [`SFTTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', args: trl.trainer.sft_config.SFTConfig | transformers.training_args.TrainingArguments | None = None, data_collator: collections.abc.Callable[[list[typing.Any]], dict[str, typing.Any]] | None = None, train_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | datasets.dataset_dict.DatasetDict | datasets.dataset_dict.IterableDatasetDict | dict[str, datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, compute_loss_func: collections.abc.Callable | None = None, compute_metrics: collections.abc.Callable[[transformers.trainer_utils.EvalPrediction], dict] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), optimizer_cls_and_kwargs: tuple[type[torch.optim.optimizer.Optimizer], dict[str, typing.Any]] | None = None, preprocess_logits_for_metrics: collections.abc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None, quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None, formatting_func: collections.abc.Callable[[dict], str] | None = None)` — Trainer for Supervised Fine-Tuning (SFT) method.
- **class `TrainTokensCallback`** — Copies ``trainer._total_train_tokens`` into the ResearchRL checkpoint state after every step.
  - `on_step_end(self, args: 'Any', state: 'Any', control: 'Any', **kw: 'Any') -> 'None'`
- **class `TrainTokensResumeMixin`** — Mixed in directly above a TRL trainer that counts tokens in ``_total_train_tokens``.
  - `train(self, resume_from_checkpoint: 'str | bool | None' = None, **kwargs: 'Any') -> 'Any'`
- `token_resume_trainer_class(trl_cls: 'type') -> 'type'` — ``research_trainer_class`` of ``trl_cls`` with :class:`TrainTokensResumeMixin` (cached).
- `track_train_tokens(run: 'ResearchRun') -> 'ResearchRun'`

## `research_rl.ui`

Local channel search + edit tool for :mod:`research_rl.capability` (``CAPABILITY.md`` 1.13).

- **class `ServerOptions`** — Network and security settings of :class:`UIServer`.
- **class `UIApp`** — A running UI: worker, workspace, API and (not yet serving) HTTP server.
  - `close(self) -> 'None'` — Stop serving, close the socket and stop the worker.
  - `serve_in_thread(self) -> 'threading.Thread'` — Start ``serve_forever`` on a daemon thread (tests, notebooks).
- **class `UIServer`** — Threading HTTP server holding the API, the token and the allowed Host / Origin values.
  - `begin_header_phase(self, conn: 'socket.socket') -> 'None'` — Start the ``header_timeout`` clock of a new connection.
  - `end_header_phase(self, conn: 'socket.socket') -> 'bool'` — Stop the clock; ``False`` if the deadline had already fired (the connection is dead).
  - `process_request(self, request: 'Any', client_address: 'Any') -> 'None'` — Start a handler thread, or close the connection when ``max_connections`` are in progress (runs on the accepting thread: nothing is read or written here).
  - `process_request_thread(self, request: 'Any', client_address: 'Any') -> 'None'` — Handle one connection on its thread and free its slot afterwards.
  - `server_close(self) -> 'None'` — Close the listening socket and stop the header watchdog.
  - `token_ok(self, presented: 'str | None') -> 'bool'` — Constant-time comparison of a presented token (bytes; non-latin-1 values never match).
  - `token_url(self) -> 'str'` — The URL with the token in its fragment (printed once at start; never logged).
  - `url` — Base URL of the UI (without the token).
- `create_app(loaded: 'LoadedModel', *, store: 'str | Path', options: 'ServerOptions', config: 'WorkspaceConfig | None' = None, runs_dir: 'str | Path | None' = None, score_fn: 'Any' = None, sync_wait: 'float' = 60.0) -> 'UIApp'` — Build worker, workspace (on the worker), API and server. Nothing is served yet.
- `main(argv: 'Sequence[str] | None' = None) -> 'int'` — Entry point of ``python -m research_rl.ui``.

## `research_rl.ui.api`

The JSON API of the UI: routing, parameter validation (with bounds) and error mapping.

- **class `Api`** — Routes authenticated JSON requests (see the module docstring).
  - `handle(self, method: 'str', segments: 'list[str]', query: 'Mapping[str, list[str]]', body: 'Any') -> 'tuple[int, Any]'` — ``(status, payload)`` for one request; never raises for request errors.
- **class `ApiError`** — An error with an HTTP status, returned as ``{"error": {...}}``.
- `error_payload(exc: 'BaseException') -> 'dict[str, Any]'` — ``{"status", "type", "message"}`` of an exception (used for jobs and responses).
- `map_exception(exc: 'BaseException') -> 'tuple[int, str]'` — HTTP status and error type for an exception (specific classes first; no tracebacks).
- constants: `LIMITS`

## `research_rl.ui.jobs`

The single model worker and its job queue.

- **class `Job`** — One unit of model work. Mutable fields are written only by the worker under ``lock``.
  - `finished_ok` — True iff the job finished without an exception.
  - `to_dict(self, *, include_result: 'bool' = True) -> 'dict[str, Any]'` — JSON-able view (the callable and the raw exception are never exposed).
- **class `JobContext`** — Handed to a job function: report progress, and check that the code runs on the worker.
  - `job_id` — Id of the running job.
  - `progress(self, fraction: 'float', message: 'str' = '') -> 'None'` — Set the job's progress (clamped to [0, 1]) and an optional short message.
- **class `ModelWorker`** — One daemon thread executing queued jobs in submission order.
  - `call(self, kind: 'str', fn: 'Callable[[JobContext], Any]', params: 'dict[str, Any] | None' = None, *, wait: 'float') -> 'Job'` — :meth:`submit`, then wait up to ``wait`` seconds for the job to finish; returns the job (finished or not).
  - `current_job` — Id of the running job, if any.
  - `get(self, job_id: 'str') -> 'Job | None'` — The job with ``job_id`` (``None`` if unknown or already dropped).
  - `list(self) -> 'list[Job]'` — Every known job, newest first.
  - `on_worker(self) -> 'bool'` — True iff the calling code runs on the worker thread.
  - `pending` — Unfinished (queued or running) jobs.
  - `stop(self, timeout: 'float' = 10.0) -> 'None'` — Stop accepting jobs, let the queued ones finish, and join the thread.
  - `submit(self, kind: 'str', fn: 'Callable[[JobContext], Any]', params: 'dict[str, Any] | None' = None) -> 'Job'` — Queue ``fn(ctx)`` as a new job and return it (``QueueFullError`` when the queue is full).
  - `thread` — The worker thread.
- **class `QueueFullError`** — The worker already has :attr:`ModelWorker.max_pending` unfinished jobs.
- **class `WorkerStoppedError`** — The worker was stopped; no job can be submitted.

## `research_rl.ui.loading`

Model loading for the UI: local safetensors checkpoints or an in-process tiny demo model.

- **class `LoadedModel`** — A model with its tokenizer and a short description of where it came from.
- **class `ModelLoadError`** — The model directory is missing, incomplete, or contains a non-safetensors weight file.
- `build_tiny_demo(seed: 'int' = 0) -> 'LoadedModel'` — A randomly initialised tiny Llama (``research_rl.models.tiny``) and its char tokenizer.
- `check_model_dir(path: 'str | Path') -> 'list[Path]'` — Validate a model directory and return the ``*.safetensors`` files the loader will open.
- `load_model_dir(path: 'str | Path') -> 'LoadedModel'` — Load a causal LM and its tokenizer from a local directory (safetensors only, offline).
- constants: `UNSAFE_SUFFIXES`

## `research_rl.ui.runs`

Read-only access to gated-loop decision logs under an allowed runs directory.

- **class `RunAccessError`** — The run name is invalid or its log resolves outside the runs directory.
- `list_runs(runs_dir: 'Path | None') -> 'list[dict[str, Any]]'` — Runs with a decision log directly under ``runs_dir`` (sorted by name).
- `read_decisions(runs_dir: 'Path', name: 'str', *, limit: 'int' = 1000, max_bytes: 'int' = 33554432) -> 'dict[str, Any]'` — Parse the decision log of run ``name``: the last ``limit`` records, summarised.
- `resolve_run(runs_dir: 'Path', name: 'str') -> 'Path'` — Path of the decision log of run ``name`` (``RunAccessError`` / ``FileNotFoundError``).
- constants: `RUN_NAME_RE`

## `research_rl.ui.server`

Local HTTP server of the UI (Python stdlib ``http.server`` only).

- **class `BindError`** — The bind address is not allowed (non-loopback without ``allow_remote``, or not an IP).
- **class `ServerOptions`** — Network and security settings of :class:`UIServer`.
- **class `UIApp`** — A running UI: worker, workspace, API and (not yet serving) HTTP server.
  - `close(self) -> 'None'` — Stop serving, close the socket and stop the worker.
  - `serve_in_thread(self) -> 'threading.Thread'` — Start ``serve_forever`` on a daemon thread (tests, notebooks).
- **class `UIServer`** — Threading HTTP server holding the API, the token and the allowed Host / Origin values.
  - `begin_header_phase(self, conn: 'socket.socket') -> 'None'` — Start the ``header_timeout`` clock of a new connection.
  - `end_header_phase(self, conn: 'socket.socket') -> 'bool'` — Stop the clock; ``False`` if the deadline had already fired (the connection is dead).
  - `process_request(self, request: 'Any', client_address: 'Any') -> 'None'` — Start a handler thread, or close the connection when ``max_connections`` are in progress (runs on the accepting thread: nothing is read or written here).
  - `process_request_thread(self, request: 'Any', client_address: 'Any') -> 'None'` — Handle one connection on its thread and free its slot afterwards.
  - `server_close(self) -> 'None'` — Close the listening socket and stop the header watchdog.
  - `token_ok(self, presented: 'str | None') -> 'bool'` — Constant-time comparison of a presented token (bytes; non-latin-1 values never match).
  - `token_url(self) -> 'str'` — The URL with the token in its fragment (printed once at start; never logged).
  - `url` — Base URL of the UI (without the token).
- `build_parser() -> 'argparse.ArgumentParser'` — The command-line interface of ``python -m research_rl.ui``.
- `check_bind_host(host: 'str', allow_remote: 'bool') -> 'tuple[str, bool]'` — Normalise the bind host; returns ``(ip, is_loopback)``.
- `create_app(loaded: 'LoadedModel', *, store: 'str | Path', options: 'ServerOptions', config: 'WorkspaceConfig | None' = None, runs_dir: 'str | Path | None' = None, score_fn: 'Any' = None, sync_wait: 'float' = 60.0) -> 'UIApp'` — Build worker, workspace (on the worker), API and server. Nothing is served yet.
- `json_depth(obj: 'Any') -> 'int'` — Nesting depth of a parsed JSON value (a scalar is 0, ``[]`` / ``{}`` 1), computed iteratively.
- `jsonable(obj: 'Any') -> 'Any'` — Copy of ``obj`` that ``json.dumps(..., allow_nan=False)`` accepts: non-finite floats become ``None``, tuples / sets lists, enums their value, other objects their ``str``.
- `main(argv: 'Sequence[str] | None' = None) -> 'int'` — Entry point of ``python -m research_rl.ui``.
- `resolve_token(env: 'dict[str, str] | None' = None) -> 'str'` — The session token: ``RESEARCH_RL_UI_TOKEN`` if set (16-256 URL-safe characters, else ``ValueError``), otherwise :func:`secrets.token_urlsafe` (32 bytes).
- constants: `DEFAULT_HEADER_TIMEOUT`, `DEFAULT_MAX_BODY`, `DEFAULT_MAX_CONNECTIONS`, `MAX_JSON_DEPTH`, `STATIC_DIR`, `TOKEN_ENV`

## `research_rl.ui.workspace`

The model-side state of the UI: model, tasks, probe, version store, stored statistics and scores.

- **class `SearchParams`** — Bounded parameters of one UI subset search (validated by the API layer).
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able copy of every field.
- **class `StoredScore`** — One stored channel score (``scores`` is ``None`` when unavailable, with ``reason``).
  - `summary(self, current_fingerprint: 'str') -> 'dict[str, Any]'` — JSON-able description (without the tensors); ``stale`` for weight-dependent scores whose fingerprint differs from ``current_fingerprint``.
- **class `StoredStats`** — Channel statistics collected on the prompts of one split (and, with ``contrast``, on the prompt + greedy completion of V2-correct and V2-incorrect items separately).
  - `summary(self, current_fingerprint: 'str') -> 'dict[str, Any]'` — JSON-able description; ``stale`` = collected on other weights than ``current_fingerprint``.
- **class `Workspace`** — Model, testbed, edit store and stored statistics; see the module docstring.
  - `channel_detail(self, layer: 'int', channel: 'int', *, k: 'int' = 10, max_versions: 'int' = 50) -> 'dict[str, Any]'` — Vocabulary projection, stored statistics and version history of one FFN channel.
  - `collect_stats(self, split: 'Split', n: 'int', *, slice_name: 'str | None' = None, contrast: 'bool' = False, threshold: 'float' = 0.0, ctx: 'JobContext | None' = None) -> 'dict[str, Any]'` — Collect :class:`ChannelStats` on ``n`` prompts of ``split`` (or of one of its slices), store them (replacing earlier stats of the split) and recompute the stored scores.
  - `commit(self, override: 'str | None', note: 'str' = '') -> 'dict[str, Any]'` — Commit the staged edit (refused without a passing verdict unless ``override``).
  - `discard(self) -> 'dict[str, Any]'` — Discard the staged edit (the parent's weights are restored bitwise).
  - `evaluate(self, ctx: 'JobContext | None' = None) -> 'dict[str, Any]'` — Gate verdict of the staged edit (store ledger) and the training reward beside it.
  - `public_history(self) -> 'dict[str, Any]'` — Published version graph: head, lineage, every version record and the staged edit.
  - `public_last_search(self) -> 'dict[str, Any] | None'` — Result of the last subset search of this process, or ``None``.
  - `public_model(self) -> 'dict[str, Any]'` — Published model description (layers, intermediate sizes, slices, gate config).
  - `public_scores(self) -> 'dict[str, dict[str, Any]]'` — Summaries of every stored score (availability, reason, staleness).
  - `public_status(self) -> 'dict[str, Any]'` — Published status (source, fingerprint, head, stats and score summaries, last evaluation).
  - `rank_channels(self, *, kind: 'str', split: 'str | None', layer: 'int | None', group: 'str | None', bands: 'int', top: 'int') -> 'dict[str, Any]'` — Rank channels by a stored score (published data only; no weights are read).
  - `restore(self, version_id: 'str') -> 'dict[str, Any]'` — Put the live model at ``version_id`` bitwise (:meth:`VersionStore.restore`).
  - `run_search(self, params: 'SearchParams', *, max_candidates: 'int', ctx: 'JobContext | None' = None) -> 'dict[str, Any]'` — Run :class:`SubsetSearch` over the groups of a stored score with ES bursts on a :class:`ChannelGainSpace`; the model is back at its start weights afterwards (checked).
  - `search_groups(self, params: 'SearchParams') -> 'list[ChannelGroup]'` — Candidate groups of a search: ``group_channels`` of the stored score, optionally filtered.
  - `stage(self, edits: 'Sequence[Mapping[str, Any]]') -> 'dict[str, Any]'` — Stage declarative edits (``EditRecord.from_dict`` of each JSON object; no code).
- **class `WorkspaceConfig`** — Settings of the task testbed, the gate used for edits and the reward shown beside it.
  - `gate_config(self) -> 'GateConfig'` — The (enabled) :class:`GateConfig` used for every edit evaluation of the store.
  - `to_dict(self) -> 'dict[str, Any]'` — JSON-able copy of every field.
- constants: `HONESTY_NOTE`, `SCORE_KINDS`, `STATS_SPLITS`

## `research_rl.utils.logging`

Consistent loggers for ResearchRL code, with an optional rank-0-only mode for multi-process runs.

- `get_logger(name: 'str', level: 'int | str' = 20, *, rank_zero_only: 'bool' = False, stream: 'TextIO | None' = None) -> 'logging.Logger'` — Return ``logging.getLogger(name)`` with exactly one ResearchRL handler attached.
- `get_rank() -> 'int'` — Global process rank: ``torch.distributed`` if initialised, else ``$RANK``, else 0.

## `research_rl.utils.profiling`

Timing, throughput and hardware/memory reports.

- **class `StepTimer`** — Records wall-clock durations of steps (and optionally tokens processed per step).
  - `count`
  - `last`
  - `start(self) -> 'None'`
  - `step(self, tokens: 'int | None' = None) -> 'Iterator[StepTimer]'`
  - `stop(self, tokens: 'int | None' = None) -> 'float'`
  - `summary(self) -> 'dict[str, float | int | None]'` — Aggregate statistics; ``tokens_per_second`` only when every step reported tokens.
  - `total`
- `gpu_memory_report(device: 'int | None' = None) -> 'dict[str, Any]'` — Measured CUDA memory statistics, or ``{"status": "NOT_TESTED", ...}`` without CUDA.
- `hardware_report() -> 'dict[str, Any]'` — What this process runs on. GPU entries are NOT_TESTED unless CUDA is available.
- `measure_throughput(fn: 'Callable[[], Any]', n_steps: 'int', warmup: 'int' = 1, tokens_per_step: 'int | None' = None, *, clock: 'Clock' = <built-in function perf_counter>, synchronize: 'Sync' = 'auto') -> 'dict[str, Any]'` — Call ``fn`` ``warmup`` times untimed, then ``n_steps`` times timed as one block.

## `research_rl.utils.smoke`

End-to-end smoke check: a real TRL trainer step with ResearchRL update rules installed.

- `trl_rules_smoke(output_dir: 'str | Path', steps: 'int' = 2, bf16: 'bool' = False) -> 'dict[str, Any]'` — Run ``steps`` SFT steps through TRL with MaskRule + P2PriorRule; return the checks.
