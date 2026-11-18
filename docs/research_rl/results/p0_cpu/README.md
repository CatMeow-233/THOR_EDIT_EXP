# P0 capability pilot: CPU, tiny model (one run)

**CPU tiny-model pilot. Not evidence about 7-8B models. GPU metrics: NOT_TESTED.**

- Code: commit `d44fbd3` (branch `claude/laughing-sagan-p6qzj6`), default `PilotConfig`
  (48 arms = {grpo, es_full, es_informed, es_random} x {gate_off, gate_on} x {leak, clean} x seeds {0,1,2},
  48 updates per arm, 6 gate decisions per arm, 196 800-parameter Llama, SFT 4000 steps).
- Command: `python examples/capability_pilot_tiny.py --output-dir <dir> --jobs 4` on a 4-core container, 883.7 s wall.
- Files: `summary.md` (human-readable), `results.json` (everything), `pilot_config.json`, `decisions/*.jsonl`.
  `start_model.safetensors` (772 KB) is not committed; it is reproduced deterministically by the command.
- Section 3 of `../../CAPABILITY.md` (the kill criteria) was fixed before this run and was not changed after it.

## Section-3 verdicts (automatic)

| criterion | verdict | how to read it |
|---|---|---|
| Gate | INCONCLUSIVE | 0 of 288 blocks were oracle-improved (max oracle diff 0.0039 vs threshold 0.01) and the gate accepted 0 first verdicts. The false-reject rate is not evaluable and the false-accept rate (0) is 0 by construction. The leak-variant hack rate was about 0.14 with the gate on and off, with 0 alarms. **P0 says nothing either way about whether the gate prevents reward hacking.** |
| Subset | KILL | No separation between the informed and the density-matched random subset. This KILL is the vacuous kind its own definition warns about: the check below shows the ES arms ended where the start checkpoint is, so neither subset acquired anything. Per section 3: channel groups stay as the rollback unit only, until a testbed where ES learns at all re-tests it. |
| Operator | INCONCLUSIVE | GRPO gain interval contains 0; KILL was unreachable by configuration (ES spent 5% of its 10x-GRPO budget). Known before the run and warned in `summary.md`. |
| Low start | INCONCLUSIVE | Compares each arm's final greedy **pass@1** with the start's sampled **pass@16** on `t_rule/unseen_pairs` (a strict acquisition reading of "exceeds the start's pass@k"; a cross-module reviewer flagged it as not like-for-like, two skeptics judged it the documented design). No arm significantly exceeds it; not every arm is significantly below it. |

## Post-hoc check: did anything learn? (`start_vs_final_check.py`, output in `start_vs_final_check.txt`)

The start checkpoint scored on each seed's final-step (step 48) tracker items, beside every arm's final point on
the same items:

- ES arms (es_full / es_informed / es_random): A and T_rule equal to the start within one or two items
  (1/256 = 0.004 per item). **No learning.**
- GRPO arms: A (train/pairs) up by about 0.01-0.05 over the start in every seed (gate_off clean: 0.242 vs 0.203,
  0.219 vs 0.172, 0.164 vs 0.125), T_rule unchanged within noise. GRPO learns a little on the training pairs and
  does not transfer within 48 updates.
- The within-arm "A gain since step 0" figures in `summary.md` (`A_gain_*`) mostly reflect the tracker's per-step
  item draw, not weight changes; cross-arm paired differences (same items) are the meaningful comparisons.

## What follows

The testbed is acquisition-limited at this budget: the gate was never asked to tell an improvement from a hack,
so its discrimination is untested. The next step is an exploratory positive-control experiment with pass/fail
rules written down before it runs: null candidates (accept rate <= alpha), known-good candidates (SFT on unseen
pairs drawn under their own purpose, interpolated at several lambda for a power curve), and known-bad candidates
(trained on the leaky verifier only). It is reported separately from these section-3 verdicts.
