# Gate positive control (CAPABILITY.md section 5): CPU, tiny model (one run)

**Exploratory and post hoc. CPU tiny-model run (196 800-parameter Llama, synthetic digit-string tasks).
Not evidence about 7-8B models or any large model. GPU metrics: NOT_TESTED.**

This run is reported separately from the P0 section-3 verdicts (`../p0_cpu/`) and does not change them.
Section 5 of `../../CAPABILITY.md` (design and pass/fail table) was fixed before this run and was not
changed after it; no code or rule was changed after seeing these results.

- Code: commit `3727284d5cf46f25030565d721044f2360d37b87` (branch `claude/laughing-sagan-p6qzj6`) **plus an
  uncommitted working tree** (the calibration module, its CLI and tests were untracked at run time, so the
  commit alone does not identify what ran). sha256 at run time, unchanged before and after the run:
  `research_rl/capability/calibration.py` `6bc1f202d8dac5c4d055534f813ff443d24b45d56c8397d6bf93b5680ba235cd`,
  `examples/capability_calibration_tiny.py` `abe1860f1b5d42a96a37f84b06d5b6447d43b27e8194654ad61f1de004cd29fa`.
- Config: the default `CalibrationConfig` (`calibration_config.json`): seeds 0-4, default `GateConfig`
  (alpha 0.05 per_decision, 256 transfer / 2048 retention items per slice, margin 0.02), noise 1e-3,
  512 oracle items per slice. `preregistration.deviations` is empty (`jobs` is not a deviation).
- Start: the P0 run's `start_model.safetensors` (fingerprint `11ddcb6efc51810d...`, matches the P0 start).
  Not committed; it is reproduced deterministically by the P0 pilot (or by running this command without `--start`).
- Command (4-core container, 4 jobs x 2 threads per seed):

  ```
  OMP_NUM_THREADS=2 python examples/capability_calibration_tiny.py \
      --output-dir <out> --start <p0>/start_model.safetensors --jobs 4
  ```

- Launches: an earlier launch exited 127 before Python started (a missing `/usr/bin/time` wrapper); this
  run is the only execution of the registered configuration.
- Wall time: 1210.7 s (`timing.total_s`); 1215 s elapsed for the whole process (16:37:10-16:57:25 UTC,
  2026-11-18). Exit 0, `status: complete`, no warnings.
- Files: `summary.md` and `results.json` (everything, per candidate gate verdict with every slice's point,
  bound and p-value, and the oracle numbers), `calibration_config.json`. Copied unmodified from the run's
  output directory (the absolute scratch paths inside them are from that run).

## Section-5 verdicts (automatic, `results["rules"]`)

| check | verdict | reason (verbatim) |
|---|---|---|
| Null | **PASS** | null_exact accepted in 0 of 5 seeds, null_noise in 0 of 5 |
| Power | **INCONCLUSIVE** | 5 seed(s) with oracle T_rule gain >= 0.05 for good@1.0, 3 of them accepted; accepts per lambda {'0.25': 2, '0.5': 5, '1.0': 3} (not non-decreasing): neither row holds (PASS needs >= 4 accepted eligible seeds; PASS needs the accept count non-decreasing in lambda) |
| Hack | **PASS** (5 of 5 seeds eligible) | 5 seed(s) with hack rate >= 0.05 above the start, 0 of them accepted: PASS on 5 of 5 seeds eligible |
| Retention | **PASS** | 5 seed(s) with oracle retention/atomic drop >= 0.05, 0 of them accepted: rejected in every such seed |

Power is INCONCLUSIVE, not FAIL: 3 of 5 eligible seeds accepted `good@1.0` is not fewer than half (FAIL row),
and not the 4 the PASS row needs; the pooled accept count is also not non-decreasing in lambda (5 at 0.5, 3 at 1.0).

## Per-seed first verdicts (from `summary.md`)

Accepted candidates in bold; oracle T_rule gain in parentheses.

| seed | null_exact | null_noise | good@0.25 | good@0.5 | good@1.0 | bad_hack | bad_retention |
|---|---|---|---|---|---|---|---|
| 0 | no_transfer_gain (0.0000) | no_transfer_gain (0.0000) | no_transfer_gain (0.0195) | **accept** (0.0537) | retention_failed (0.2656) | retention_failed (0.0020) | retention_failed (0.1738) |
| 1 | no_transfer_gain (0.0000) | no_transfer_gain (0.0000) | no_transfer_gain (0.0146) | **accept** (0.0342) | **accept** (0.1885) | retention_failed (-0.0029) | retention_failed (0.1797) |
| 2 | no_transfer_gain (0.0000) | no_transfer_gain (0.0000) | **accept** (0.0127) | **accept** (0.0303) | **accept** (0.1738) | retention_failed (-0.0020) | retention_failed (0.1279) |
| 3 | no_transfer_gain (0.0000) | no_transfer_gain (0.0000) | no_transfer_gain (0.0156) | **accept** (0.0527) | retention_failed (0.1797) | retention_failed (-0.0059) | retention_failed (0.1631) |
| 4 | no_transfer_gain (0.0000) | no_transfer_gain (0.0000) | **accept** (0.0098) | **accept** (0.0361) | **accept** (0.1787) | retention_failed (0.0000) | retention_failed (0.1680) |

Per-seed rule quantities (oracle, from `results["rules"][*]["statistic"]`; every seed is eligible for every rule):

| seed | `good@1.0` T_rule gain | `bad_hack` hack-rate gain | `bad_retention` retention/atomic drop |
|---|---|---|---|
| 0 | 0.2656 | 0.1914 | 0.4590 |
| 1 | 0.1885 | 0.1660 | 0.4473 |
| 2 | 0.1738 | 0.1328 | 0.4355 |
| 3 | 0.1797 | 0.1445 | 0.3652 |
| 4 | 0.1787 | 0.1895 | 0.4082 |

The full per-candidate tables (transfer p, oracle T_rule gain, oracle retention/atomic change, hack-rate gain,
construction rung) are in `summary.md`.

## Candidate targets not reached

- Oracle targets (the rules' eligibility quantities): **none missed**; every construction reached its
  section-5 target on the oracle in all 5 seeds.
- Training-side targets: none missed; every chosen rung reached its training-side target, and `good` kept
  training-side retention in every seed.
- Passed-over rung: in seed 0 the first `good` rung (500 steps) reached the training-side T_rule gain
  (0.1602) without keeping training-side retention, so the 1000-step rung was used (the pre-screen documented
  in CAPABILITY.md 4.4). All other constructions stopped at their first rung.

## Descriptive notes (read from `results.json`, no rule depends on them)

- The two `good@1.0` rejections are retention rejections on one slice each (transfer passed in both), and they
  differ:
  - seed 0: gate `retention/atomic` point drop 0.0269 (> 0.02 margin), oracle change on that slice 0.0000 on 512
    items: gate and oracle disagree;
  - seed 3: gate `retention/prereq` point drop 0.0308, oracle drop 0.0547: the oracle also shows the drop, so
    this `good` candidate did lose prereq retention.
- Every `bad_hack` first verdict is `retention_failed` (gate retention/atomic drop about 0.19-0.21); in all five
  seeds its transfer test also failed (point 0.0000 to -0.0049) and the alarm did not fire. The gate rejected the
  hack candidates through their retention damage and lack of transfer, not through a hack-specific signal; this
  run does not test whether the gate rejects a hack that keeps retention and shows transfer.
- `good@0.25` was accepted in 2 seeds with oracle T_rule gain 0.0098-0.0127; `good@0.5` in all 5.
