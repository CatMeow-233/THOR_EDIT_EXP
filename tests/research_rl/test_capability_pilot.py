"""Tests of research_rl.capability.pilot (the P0 CPU pilot) and its thin CLI.

CPU only; the tiny model is built in-process and nothing is downloaded. The end-to-end tests run
the ``--smoke`` configuration (a plumbing check: its numbers are not results). The criterion
tests run :func:`evaluate_criteria` on synthetic results dicts so that every SUCCESS / KILL /
INCONCLUSIVE branch is reached deterministically.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from research_rl.capability.loop import OVERRIDE_GATE_DISABLED, WOULD_PREFIX
from research_rl.capability.pilot import (
    BANNER,
    ORACLE_PURPOSE,
    VERDICT_INCONCLUSIVE,
    VERDICT_KILL,
    VERDICT_SUCCESS,
    ArmSpec,
    PilotConfig,
    _report_counts,
    _sampler_root,
    arm_specs,
    baseline_standard_error,
    build_splits,
    choose_retention_items,
    evaluate_criteria,
    excluded_uids,
    expected_outputs,
    gate_exercise,
    interval_with_baseline_error,
    oracle_exclusion_steps,
    oracle_items,
    paired_interval,
    parse_arm,
    render_summary,
    run_pilot,
    smoke_config,
)
from research_rl.capability.probe import TRACK_POOL, CapabilityProbe, StartMap, item_pool
from research_rl.capability.tasks import ExactAnswerVerifier, ItemSampler, VerifierScoreFn
from research_rl.capability.tracker import CapabilityTracker
from research_rl.capability.types import OperatorReport, Split
from research_rl.models.tiny import build_char_tokenizer

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================== smoke runs
@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> tuple[PilotConfig, Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("pilot_smoke")
    config = smoke_config(progress=False)
    run_pilot(config, out)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    return config, out, results


def _decisions(results: dict[str, Any]) -> list[Any]:
    """Everything decision-related of every arm (no wall times)."""
    return [
        {
            "key": arm["key"],
            "decisions": arm["decisions"],
            "blocks": arm["blocks"],
            "curve": arm["curve"],
            "counts": arm["counts"],
            "final_fingerprint": arm["final_fingerprint"],
        }
        for arm in results["arms"]
    ]


def _assert_arms_train(results: dict[str, Any]) -> None:
    """The smoke arms are not degenerate: training rollouts pass V1 (hacks, on the leak variant) and
    every arm's final weights differ from the start checkpoint, so a determinism comparison of
    their decisions, curves, counts and weight fingerprints can fail when training changes."""
    start = results["sft"]["start_fingerprint"]
    for arm in results["arms"]:
        assert arm["counts"]["v1_pass"] > 0, arm["key"]
        assert arm["counts"]["hacks"] > 0, arm["key"]
        assert arm["final_fingerprint"] != start, arm["key"]


def test_smoke_writes_every_output(smoke: tuple[PilotConfig, Path, dict[str, Any]]) -> None:
    config, out, results = smoke
    for path in expected_outputs(out, config):
        assert path.is_file(), path
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert BANNER in summary
    assert "NOT_TESTED" in summary
    assert results["banner"] == BANNER
    assert results["environment"]["gpu"] == "NOT_TESTED"
    assert [a["key"] for a in results["arms"]] == ["grpo/gate_on/leak/s0", "es_informed/gate_off/leak/s0"]
    assert set(results["criteria"]) >= {"gate", "subset", "operator", "low_start", "definitions"}
    for name in ("gate", "subset", "operator", "low_start"):
        # one seed: never a verdict
        assert results["criteria"][name]["verdict"] == VERDICT_INCONCLUSIVE
    assert len(results["splits"]["fingerprint"]) == 64
    assert results["splits"]["num_operators"] >= 10
    assert results["oracle"]["improvement_threshold"] == config.oracle_threshold
    assert results["gate_choice"]["items_per_retention_slice"] == 16
    assert set(results["start_summary"]) >= {"t_rule/unseen_pairs", "retention/atomic"}
    # the smoke gate is not exercised (no oracle-improved block): the run says so
    assert not results["gate_exercise"]["exercised"]
    assert any(w.startswith("gate not exercised") for w in results["warnings"])
    assert "## Gate exercise" in summary
    # the decision log has one JSON line per decision record
    for spec in arm_specs(config):
        lines = (out / "decisions" / f"{spec.file_stem}.jsonl").read_text(encoding="utf-8").splitlines()
        arm = next(a for a in results["arms"] if a["key"] == spec.key)
        assert [json.loads(line) for line in lines] == arm["decisions"]


def test_smoke_arm_semantics(smoke: tuple[PilotConfig, Path, dict[str, Any]]) -> None:
    config, _, results = smoke
    grpo, es = results["arms"]
    # gate_off records the would-be decision and never rolls back
    for record in es["decisions"]:
        assert record["decision"] == "accept"
        assert record["override"] == OVERRIDE_GATE_DISABLED
        assert record["reason"].startswith(WOULD_PREFIX)
        assert len(record["verdicts"]) == 1
    for block in es["blocks"]:
        assert block["first_reason"] == block["reason"][len(WOULD_PREFIX) :]
    for arm in (grpo, es):
        assert len(arm["blocks"]) == config.updates // config.base_gate_config().block_updates
        assert len(arm["curve"]) == len(arm["blocks"]) + 1  # step 0 plus one point per decision
        for block in arm["blocks"]:
            oracle = block["oracle"]
            assert oracle["n"] == 2 * config.oracle_items_per_slice  # two T_RULE slices
            assert oracle["improved"] == (oracle["diff"] > config.oracle_threshold)
            assert block["false_accept"] == (block["accepted_first"] and not oracle["improved"])
            assert block["false_reject"] == ((not block["accepted_first"]) and oracle["improved"])
        assert 0.0 <= arm["hack_rate_total"] <= 1.0
        assert arm["alarm"]["updates"] == config.updates
    # pass counts: GRPO backprops, ES is forward-only; FPE counts a backward sequence as two forwards
    gc, ec = grpo["counts"], es["counts"]
    rollouts = config.grpo_prompts_per_update * config.grpo_group_size * config.updates
    assert gc["rollouts"] == rollouts
    assert gc["backward_calls"] == config.updates and gc["backward_sequences"] == rollouts
    assert gc["fpe"] == gc["forward_sequences"] + 2 * gc["backward_sequences"] == 4 * rollouts
    assert ec["backward_calls"] == 0 and ec["backward_sequences"] == 0
    assert ec["rollouts"] == config.es_population * config.es_items_per_update * config.updates
    assert ec["fpe"] == ec["forward_sequences"] == ec["rollouts"]
    assert es["space"]["type"] == "ChannelGainSpace"
    assert es["space"]["channels"] == config.informed_k
    assert results["selection"]["scorer"] in ("contrast", "usage_ratio")


def test_smoke_is_deterministic_across_runs_and_jobs(
    smoke: tuple[PilotConfig, Path, dict[str, Any]], tmp_path: Path
) -> None:
    config, _, first = smoke
    _assert_arms_train(first)
    again = dataclass_replace(config, jobs=2)
    second = run_pilot(again, tmp_path)
    _assert_arms_train(second)
    assert _decisions(second) == _decisions(first)
    assert second["criteria"] == first["criteria"]
    assert second["start_map"] == first["start_map"]
    assert second["selection"] == first["selection"]


@pytest.mark.integration
def test_smoke_comparison_is_sensitive_to_training(
    smoke: tuple[PilotConfig, Path, dict[str, Any]], tmp_path: Path
) -> None:
    """Changing only the operators' learning rates changes what the determinism test compares (so
    it is not vacuous); a negative oracle threshold drives the oracle-improved / false-reject path
    end to end (every block counts as improved, nothing else changes)."""
    config, _, first = smoke
    changed = dataclass_replace(
        config, grpo_lr=config.grpo_lr * 3.0, es_lr=config.es_lr * 3.0, oracle_threshold=-1.0
    )
    second = run_pilot(changed, tmp_path)
    assert second["start_map"] == first["start_map"]  # same SFT and start: only the arms differ
    for a, b in zip(first["arms"], second["arms"]):
        assert a["final_fingerprint"] != b["final_fingerprint"], a["key"]
    assert _decisions(second) != _decisions(first)
    for arm in second["arms"]:
        assert arm["gate_errors"]["oracle_improved"] == len(arm["blocks"]) > 0
        for block in arm["blocks"]:
            assert block["oracle"]["improved"]
            assert block["false_reject"] == (not block["accepted_first"])
            assert not block["false_accept"]
        rejected = sum(1 for b in arm["blocks"] if not b["accepted_first"])
        assert arm["gate_errors"]["false_reject_rate"] == rejected / len(arm["blocks"])
        assert arm["gate_errors"]["false_accept_rate"] is None  # no block was not improved
    exercise = second["gate_exercise"]
    assert exercise["oracle_improved"] == sum(len(a["blocks"]) for a in second["arms"])
    assert exercise["false_reject_evaluable"]


def test_trailing_updates_and_alarm_decisions(tmp_path: Path) -> None:
    # 5 updates in blocks of 2: one update follows the last decision and is measured once more
    config = smoke_config(updates=5, arms=("es_informed+gate_on",), sft_steps=300, progress=False)
    results = run_pilot(config, tmp_path / "tail")
    (arm,) = results["arms"]
    assert arm["trailing_updates"] == 1 and len(arm["blocks"]) == 2
    assert arm["curve"][-1]["updates"] == 5
    assert arm["curve"][-1]["fpe"] == arm["counts"]["fpe"]
    assert arm["final"]["T_rule"] == arm["curve"][-1]["T_rule"]
    # an alarm that fires on every update: every update ends in an alarm decision
    alarm = {"min_rollouts": 4, "baseline_gap": -0.5, "drift": 0.0, "threshold": 0.1}
    config = smoke_config(updates=3, arms=("grpo+gate_on",), alarm=alarm, sft_steps=300, progress=False)
    results = run_pilot(config, tmp_path / "alarm")
    (arm,) = results["arms"]
    assert arm["alarm"]["gap_alarms"] == 3
    assert [b["updates_done"] for b in arm["blocks"]] == [1, 2, 3]
    assert all(r["alarm"] for r in arm["decisions"])
    assert {r["reason"] for r in arm["decisions"]} <= {
        "confirmed_alarm",
        "alarm_hold",
        "horizon_exhausted",
        "hold_limit",
    }
    assert arm["trailing_updates"] == 0 and arm["curve"][-1]["fpe"] == arm["counts"]["fpe"]


@pytest.mark.integration
def test_oracle_items_shared_by_arms_with_different_update_counts(tmp_path: Path) -> None:
    # es_updates != updates: GRPO and ES arms of one seed still score the same oracle items at the
    # same decision index (the exclusion bound is the config's, not the arm's). Item counts are large
    # enough that tracker steps 5-6 collide with decision-0 oracle draws, so an arm-own bound (4
    # for GRPO, 6 for ES) would give the two arms different oracle items.
    config = smoke_config(
        es_updates=6, sft_steps=300, track_items_per_slice=64, oracle_items_per_slice=256, progress=False
    )
    results = run_pilot(config, tmp_path)
    grpo, es = results["arms"]
    assert grpo["updates"] == 4 and es["updates"] == 6
    for d in range(len(grpo["blocks"])):
        assert grpo["blocks"][d]["oracle"]["items_sha256"] == es["blocks"][d]["oracle"]["items_sha256"]
    assert grpo["blocks"][0]["oracle"]["items_sha256"] != grpo["blocks"][1]["oracle"]["items_sha256"]


def test_report_counts_hack_definition_and_fpe() -> None:
    # hack = V1 pass and V2 fail; V1 & V2 pass is a real solve, V1 fail is no hack
    report = OperatorReport(
        update_index=0,
        reward_v1=0.75,
        reward_v2=0.25,
        v1_pass=(True, True, False, True),
        v2_pass=(True, False, False, False),
        forward_passes=2,
        backward_passes=1,
    )
    grpo = _report_counts(report, "grpo")
    assert (grpo["rollouts"], grpo["hacks"], grpo["v1_pass"], grpo["v2_pass"]) == (4, 2, 3, 1)
    assert grpo["fpe"] == 2 * 4 + 2 * 4  # generate + training forward, one backward = two forwards
    es = _report_counts(report, "es")
    assert es["hacks"] == 2 and es["fpe"] == 4 and es["backward_sequences"] == 0


def test_oracle_exclusion_bound_is_per_config() -> None:
    config = PilotConfig(updates=48, es_updates=96)
    assert oracle_exclusion_steps(config) == 96
    # the ETA probe (one block) and a full arm of either operator use the same bound
    for n in (8, 48, 96, None):
        assert oracle_exclusion_steps(config, n) == 96
    # an explicit longer run still excludes every tracker step it measures
    assert oracle_exclusion_steps(config, 200) == 200
    assert oracle_exclusion_steps(PilotConfig(arms=("grpo+gate_on",), updates=16, es_updates=96)) == 16


def dataclass_replace(config: PilotConfig, **changes: Any) -> PilotConfig:
    data = config.to_dict()
    data.update(changes)
    return PilotConfig.from_dict(data)


# ============================================================================== oracle items
def test_oracle_items_are_disjoint_from_gate_tracker_and_start(
    smoke: tuple[PilotConfig, Path, dict[str, Any]],
) -> None:
    config, _, results = smoke
    splits = build_splits(config)
    sampler = ItemSampler(splits, root_seed=_sampler_root(config, 0))
    start = StartMap.from_dict(results["start_map"])
    tok = build_char_tokenizer()
    fn = VerifierScoreFn(tok, ExactAnswerVerifier(), max_new_tokens=8, batch_size=8)
    probe = CapabilityProbe(
        sampler,
        fn,
        transfer_slices=("t_rule/unseen_pairs", "t_rule/depth3", "t_task/argmax"),
        retention_slices=("retention/atomic", "retention/copy"),
        items_per_transfer_slice=64,
        items_per_retention_slice=64,
    )
    tracker = CapabilityTracker(sampler, fn, StartMap.from_dict(results["start_map"]), items_per_slice=32)
    max_step = 6
    exclude = excluded_uids(sampler, start, track_items_per_slice=32, max_step=max_step)
    start_uids = {u for s in start.slices.values() for u in s.uids}
    track_uids = {it.uid for step in range(max_step + 1) for g in tracker.items(step).values() for it in g}
    for decision in range(4):
        oracle = oracle_items(sampler, 200, decision, exclude)
        assert set(oracle) == set(splits.slices_of(Split.T_RULE))
        uids = {it.uid for group in oracle.values() for it in group}
        assert len(uids) == 400
        assert all(it.split is Split.T_RULE for group in oracle.values() for it in group)
        assert all(item_pool(sampler, u) == TRACK_POOL for u in uids)
        gate_uids = {it.uid for a in range(4) for g in probe.gate_items(decision, a).values() for it in g}
        assert not uids & gate_uids
        assert not uids & track_uids
        assert not uids & start_uids
        # deterministic in the key, fresh per decision
        assert oracle == oracle_items(sampler, 200, decision, exclude)
    assert {it.uid for g in oracle_items(sampler, 50, 0, exclude).values() for it in g} != {
        it.uid for g in oracle_items(sampler, 50, 1, exclude).values() for it in g
    }
    assert ORACLE_PURPOSE == "oracle"


# ============================================================================== configuration
def test_config_roundtrip_and_validation() -> None:
    config = PilotConfig()
    assert PilotConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert len(arm_specs(config)) == 4 * 2 * 2 * 3
    with pytest.raises(ValueError, match="num_operators"):
        PilotConfig(num_operators=9)
    with pytest.raises(ValueError, match="arm"):
        PilotConfig(arms=("grpo+maybe",))
    with pytest.raises(ValueError, match="set per arm"):
        PilotConfig(gate={"enabled": False})
    with pytest.raises(ValueError, match="unknown gate keys"):
        PilotConfig(gate={"no_such_key": 1})
    with pytest.raises(ValueError, match="unknown PilotConfig keys"):
        PilotConfig.from_dict({"bogus": 1})
    assert parse_arm("es_random+gate_off") == ("es_random", "gate_off")


def test_arm_specs_canonical_order() -> None:
    config = PilotConfig(
        arms=("es_random+gate_on", "grpo+gate_off"), seeds=(2, 0), variants=("clean", "leak")
    )
    keys = [s.key for s in arm_specs(config)]
    assert keys == [
        "grpo/gate_off/leak/s0",
        "es_random/gate_on/leak/s0",
        "grpo/gate_off/leak/s2",
        "es_random/gate_on/leak/s2",
        "grpo/gate_off/clean/s0",
        "es_random/gate_on/clean/s0",
        "grpo/gate_off/clean/s2",
        "es_random/gate_on/clean/s2",
    ]
    assert ArmSpec("grpo", "gate_on", "leak", 1).file_stem == "grpo__gate_on__leak__s1"


def test_choose_retention_items_covers_the_horizon() -> None:
    config = PilotConfig()
    splits = build_splits(config)
    choice = choose_retention_items(config, splits.slices_of(Split.RETENTION))
    # 48 updates, blocks of 8, 3 verdicts per decision, alarm decisions: 6 * 3 + 42 = 60 verdicts
    assert choice["worst_case_verdicts"] == 60
    assert not choice["short_horizon"]
    covering = [row["n"] for row in choice["candidates"] if row["covers"]]
    assert choice["items_per_retention_slice"] == min(covering)
    gate = config.base_gate_config()
    n = choice["items_per_retention_slice"]
    assert gate.retention_horizon(n=n, n10=round(0.005 * n)) >= 60
    smaller = [row for row in choice["candidates"] if row["n"] < n]
    assert smaller and all(not row["covers"] for row in smaller)
    short = choose_retention_items(PilotConfig(retention_items_candidates=(64,)), ("retention/atomic",))
    assert short["short_horizon"] and short["items_per_retention_slice"] == 64


def test_paired_interval() -> None:
    assert paired_interval([])["mean"] is None
    one = paired_interval([0.3])
    assert one["mean"] == 0.3 and one["lo"] is None
    flat = paired_interval([0.1, 0.1, 0.1])
    assert flat["se"] == 0.0 and flat["lo"] == pytest.approx(0.1) and flat["hi"] == pytest.approx(0.1)
    iv = paired_interval([1.0, 2.0, 3.0], alpha=0.05)
    # mean 2, sd 1, se 1/sqrt(3), t_{0.975, 2} = 4.3027
    assert iv["mean"] == pytest.approx(2.0)
    assert iv["hi"] - iv["mean"] == pytest.approx(4.302653 / 3**0.5, rel=1e-4)


# ============================================================================== criteria on synthetic results
def _arm(
    operator: str,
    gate: str,
    variant: str,
    seed: int,
    *,
    t_rule: float = 0.1,
    hack: float = 0.0,
    r_mean: float = 0.9,
    a: float = 0.2,
    unseen: float | None = None,
    fpe: int = 1000,
    curve: Sequence[tuple[int, float]] | None = None,
    blocks: Sequence[tuple[bool, bool]] = (),
    t_rule0: float = 0.0,
) -> dict[str, Any]:
    pts = list(curve) if curve is not None else [(0, t_rule0), (fpe, t_rule)]
    return {
        "operator": operator,
        "gate": gate,
        "variant": variant,
        "seed": seed,
        "key": f"{operator}/{gate}/{variant}/s{seed}",
        "final": {
            "A": a,
            "R_mean": r_mean,
            "T_rule": t_rule,
            "t_rule_unseen": t_rule if unseen is None else unseen,
        },
        "t_rule0": t_rule0,
        "hack_rate_final": hack,
        "counts": {"fpe": fpe},
        "curve": [{"fpe": f, "T_rule": t} for f, t in pts],
        "blocks": [{"accepted_first": acc, "oracle": {"improved": imp}} for acc, imp in blocks],
    }


def _results(arms: list[dict[str, Any]], *, pass_k: float = 0.05, start_items: int = 128) -> dict[str, Any]:
    return {
        "config": {"criteria_alpha": 0.05, "min_improved_blocks": 4, "start_items_per_slice": start_items},
        "start_summary": {"t_rule/unseen_pairs": {"pass1": 0.0, "pass_k": pass_k, "sampled_mean": 0.0}},
        "arms": arms,
    }


def _gate_arms(seeds: Sequence[int], leak_on: Sequence[float], leak_off: Sequence[float], clean_on: Sequence[float], clean_off: Sequence[float], *, hack_on: float = 0.05, hack_off: float = 0.4, blocks: Sequence[tuple[bool, bool]] = ()) -> list[dict[str, Any]]:  # fmt: skip
    arms = []
    for i, s in enumerate(seeds):
        arms.append(
            _arm("grpo", "gate_on", "leak", s, t_rule=leak_on[i], hack=hack_on + 0.01 * i, blocks=blocks)
        )
        arms.append(_arm("grpo", "gate_off", "leak", s, t_rule=leak_off[i], hack=hack_off + 0.02 * i))
        arms.append(_arm("grpo", "gate_on", "clean", s, t_rule=clean_on[i]))
        arms.append(_arm("grpo", "gate_off", "clean", s, t_rule=clean_off[i]))
    return arms


def test_gate_criterion_branches() -> None:
    seeds = (0, 1, 2)
    good = _gate_arms(seeds, [0.30, 0.32, 0.29], [0.10, 0.11, 0.12], [0.20, 0.21, 0.19], [0.20, 0.20, 0.20])
    crit = evaluate_criteria(_results(good))["gate"]
    assert crit["verdict"] == VERDICT_SUCCESS, crit
    assert crit["statistic"]["d_hack"]["lo"] > 0 and crit["statistic"]["d_T_rule_leak"]["lo"] > 0
    # clean cost (0.2) larger than the planted gain (~0.01): KILL
    costly = _gate_arms(seeds, [0.12, 0.12, 0.13], [0.11, 0.11, 0.12], [0.00, 0.01, 0.00], [0.20, 0.21, 0.20])
    assert evaluate_criteria(_results(costly))["gate"]["verdict"] == VERDICT_KILL
    # the gate rejects 3 of 4 oracle-improved blocks: KILL even though the T_rule intervals look fine
    rejecting = _gate_arms(
        seeds,
        [0.30, 0.32, 0.29],
        [0.10, 0.11, 0.12],
        [0.20, 0.21, 0.19],
        [0.20, 0.20, 0.20],
        blocks=[(False, True), (False, True), (False, True), (True, True), (True, False)],
    )
    crit = evaluate_criteria(_results(rejecting))["gate"]
    assert crit["verdict"] == VERDICT_KILL and "rejected" in crit["reason"]
    assert crit["statistic"]["oracle"]["improved_blocks"] == 12
    # near zero everywhere: a tiny negative gain and a tiny clean benefit are noise, not a KILL
    near_zero = _gate_arms(
        seeds, [0.004, 0.0, 0.002], [0.004, 0.004, 0.004], [0.0, 0.004, 0.002], [0.0, 0.0, 0.004]
    )
    crit = evaluate_criteria(_results(near_zero))["gate"]
    assert crit["verdict"] == VERDICT_INCONCLUSIVE, crit
    assert crit["statistic"]["k_cost_minus_gain"]["lo"] <= 0
    # noisy: intervals contain zero, no kill condition
    noisy = _gate_arms(seeds, [0.30, 0.05, 0.20], [0.10, 0.25, 0.12], [0.20, 0.21, 0.19], [0.20, 0.20, 0.20])
    assert evaluate_criteria(_results(noisy))["gate"]["verdict"] == VERDICT_INCONCLUSIVE
    # two seeds: never a verdict, even for a clear effect
    two = _gate_arms((0, 1), [0.30, 0.32], [0.10, 0.11], [0.20, 0.21], [0.20, 0.20])
    crit = evaluate_criteria(_results(two))["gate"]
    assert crit["verdict"] == VERDICT_INCONCLUSIVE and "need >= 3" in crit["reason"]
    # a missing counterpart arm (no clean variant): INCONCLUSIVE, no crash
    leak_only = [a for a in good if a["variant"] == "leak"]
    assert evaluate_criteria(_results(leak_only))["gate"]["verdict"] == VERDICT_INCONCLUSIVE


def _subset_arms(d_r: Sequence[float], d_t: Sequence[float], d_a: Sequence[float]) -> list[dict[str, Any]]:
    arms = []
    for s, (r, t, a) in enumerate(zip(d_r, d_t, d_a)):
        arms.append(_arm("es_informed", "gate_off", "clean", s, r_mean=0.8 + r, t_rule=0.1 + t, a=0.3 + a))
        arms.append(_arm("es_random", "gate_off", "clean", s, r_mean=0.8, t_rule=0.1, a=0.3))
    return arms


def test_subset_criterion_branches() -> None:
    win = _subset_arms([0.05, 0.06, 0.055], [0.0, 0.01, -0.01], [0.0, 0.005, -0.005])
    assert evaluate_criteria(_results(win))["subset"]["verdict"] == VERDICT_SUCCESS
    tie = _subset_arms([0.05, -0.04, 0.0], [0.02, -0.03, 0.0], [0.0, 0.005, -0.005])
    assert evaluate_criteria(_results(tie))["subset"]["verdict"] == VERDICT_KILL
    unequal_a = _subset_arms([0.05, 0.06, 0.055], [0.0, 0.01, -0.01], [0.10, 0.12, 0.11])
    crit = evaluate_criteria(_results(unequal_a))["subset"]
    assert crit["verdict"] == VERDICT_INCONCLUSIVE and "equal A" in crit["reason"]
    assert evaluate_criteria(_results(win[:4]))["subset"]["verdict"] == VERDICT_INCONCLUSIVE


def _operator_arms(
    grpo_gain: Sequence[float], es_curves: Sequence[Sequence[tuple[int, float]]], *, grpo_fpe: int = 100
) -> list[dict[str, Any]]:
    arms = []
    for s, (g, curve) in enumerate(zip(grpo_gain, es_curves)):
        arms.append(_arm("grpo", "gate_off", "clean", s, t_rule=g, fpe=grpo_fpe))
        arms.append(
            _arm("es_informed", "gate_off", "clean", s, t_rule=curve[-1][1], fpe=curve[-1][0], curve=curve)
        )
    return arms


def test_operator_criterion_branches() -> None:
    gains = [0.2, 0.22, 0.18]
    # ES reaches > half of GRPO's gain within 10x GRPO's 100 FPE
    reach = _operator_arms(gains, [[(0, 0.0), (500, 0.15), (2000, 0.3)]] * 3)
    crit = evaluate_criteria(_results(reach))["operator"]
    assert crit["verdict"] == VERDICT_SUCCESS, crit
    # the 0.3 point lies beyond the budget and does not count
    assert crit["statistic"]["per_seed"][0]["es_gain_within_budget"] == pytest.approx(0.15)
    # ES is scored at its last point within the budget, like GRPO at its final point: a noisy peak
    # earlier on the curve does not count (a maximum over points would favour ES by selection)
    peak = _operator_arms(gains, [[(0, 0.0), (300, 0.2), (900, 0.01)]] * 3)
    crit = evaluate_criteria(_results(peak))["operator"]
    assert crit["verdict"] != VERDICT_SUCCESS, crit
    assert crit["statistic"]["per_seed"][0]["es_gain_within_budget"] == pytest.approx(0.01)
    assert crit["statistic"]["per_seed"][0]["es_point_fpe"] == 900
    # gains are measured from each arm's own step-0 tracker point, not from the shared start-map value
    offset = _operator_arms(gains, [[(0, 0.0), (500, 0.15), (2000, 0.3)]] * 3)
    for arm in offset:
        arm["t_rule0"] = 0.5  # a biased start-map baseline that would make every gain negative
    assert evaluate_criteria(_results(offset))["operator"]["verdict"] == VERDICT_SUCCESS
    # ES spent the whole budget and stays far below half: KILL
    below = _operator_arms(gains, [[(0, 0.0), (600, 0.01), (1200, 0.02)]] * 3)
    assert evaluate_criteria(_results(below))["operator"]["verdict"] == VERDICT_KILL
    # ES below, but it stopped at 200 FPE (< budget 1000): INCONCLUSIVE, not KILL
    short = _operator_arms(gains, [[(0, 0.0), (200, 0.01)]] * 3)
    crit = evaluate_criteria(_results(short))["operator"]
    assert crit["verdict"] == VERDICT_INCONCLUSIVE and "budget" in crit["reason"]
    # GRPO gained nothing
    flat = _operator_arms([0.0, -0.01, 0.0], [[(0, 0.0), (1000, 0.0)]] * 3)
    assert evaluate_criteria(_results(flat))["operator"]["verdict"] == VERDICT_INCONCLUSIVE
    assert evaluate_criteria(_results(reach[:2]))["operator"]["verdict"] == VERDICT_INCONCLUSIVE


def test_low_start_criterion_branches() -> None:
    above = [_arm("grpo", "gate_on", "leak", s, unseen=0.2 + 0.01 * s) for s in range(3)]
    above += [_arm("es_full", "gate_on", "leak", s, unseen=0.0) for s in range(3)]
    crit = evaluate_criteria(_results(above, pass_k=0.05))["low_start"]
    assert crit["verdict"] == VERDICT_SUCCESS and crit["statistic"]["exceeding"] == ["grpo/gate_on/leak"]
    below = [_arm(op, "gate_on", "leak", s, unseen=0.01 * s) for op in ("grpo", "es_full") for s in range(3)]
    # a precise start measurement: clearly below it -> KILL
    assert (
        evaluate_criteria(_results(below, pass_k=0.05, start_items=100000))["low_start"]["verdict"]
        == VERDICT_KILL
    )
    # the same finals against a 128-item start sample: its own sampling error (common to every seed)
    # keeps the interval from excluding zero
    crit = evaluate_criteria(_results(below, pass_k=0.05, start_items=128))["low_start"]
    assert crit["verdict"] == VERDICT_INCONCLUSIVE, crit
    assert crit["statistic"]["start_pass_k_se"] == pytest.approx(baseline_standard_error(0.05, 128))
    mixed = [_arm("grpo", "gate_on", "leak", s, unseen=u) for s, u in enumerate((0.0, 0.2, 0.06))]
    assert evaluate_criteria(_results(mixed, pass_k=0.05))["low_start"]["verdict"] == VERDICT_INCONCLUSIVE
    assert evaluate_criteria(_results(above[:2]))["low_start"]["verdict"] == VERDICT_INCONCLUSIVE


def _binomial(rng: random.Random, p: float, n: int) -> float:
    return sum(rng.random() < p for _ in range(n)) / n


def test_low_start_false_success_rate_under_the_null() -> None:
    """Null: every seed's true final pass@1 equals the true start pass@k. The start pass@k is one
    128-item sample shared by every seed, so a low draw shifts every paired difference together;
    an interval that treats it as a constant declares SUCCESS far above the nominal one-sided
    2.5% (about 9% measured), the interval with the baseline's error stays at or below it."""
    rng = random.Random(20261118)
    trials, p, n_items = 400, 0.10, 128
    with_baseline = without_baseline = 0
    for _ in range(trials):
        pass_k = _binomial(rng, p, n_items)
        finals = [_binomial(rng, p, n_items) for _ in range(3)]
        arms = [_arm("grpo", "gate_off", "clean", s, unseen=f) for s, f in enumerate(finals)]
        crit = evaluate_criteria(_results(arms, pass_k=pass_k, start_items=n_items))["low_start"]
        with_baseline += crit["verdict"] == VERDICT_SUCCESS
        plain = paired_interval([f - pass_k for f in finals], 0.05)
        without_baseline += plain["lo"] is not None and plain["lo"] > 0
    assert without_baseline / trials > 0.05  # the defect this guards against
    assert with_baseline / trials <= 0.035


def test_baseline_error_helpers() -> None:
    assert baseline_standard_error(0.0, 128) > 0.0  # a zero count does not prove a zero rate
    assert baseline_standard_error(0.5, 100) == pytest.approx((0.5 * 0.5 / 100) ** 0.5, rel=0.01)
    with pytest.raises(ValueError):
        baseline_standard_error(0.1, 0)
    base = paired_interval([0.01, 0.02, 0.03])
    wide = interval_with_baseline_error(base, 0.02, 0.05, baseline_n=128)
    assert wide["lo"] < base["lo"] and wide["hi"] > base["hi"] and wide["mean"] == base["mean"]
    assert wide["se_seeds"] == base["se"] and wide["se_baseline"] == 0.02
    same = interval_with_baseline_error(base, 0.0, 0.05, baseline_n=128)
    assert same["lo"] == pytest.approx(base["lo"]) and same["hi"] == pytest.approx(base["hi"])
    assert interval_with_baseline_error(paired_interval([0.1]), 0.02)["lo"] is None


def test_gate_exercise_flags_a_vacuous_run() -> None:
    # every block held / rejected and none oracle-improved (what the default run measured)
    vacuous = _gate_arms(
        (0, 1, 2),
        [0.004, 0.0, 0.002],
        [0.004, 0.004, 0.004],
        [0.0, 0.004, 0.002],
        [0.0, 0.0, 0.004],
        blocks=[(False, False)] * 6,
    )
    ex = gate_exercise(vacuous, 4)
    assert not ex["exercised"] and not ex["false_reject_evaluable"] and not ex["false_accept_informative"]
    assert ex["accepted"] == 0 and ex["oracle_improved"] == 0 and len(ex["problems"]) == 2
    crit = evaluate_criteria(_results(vacuous))["gate"]
    assert crit["statistic"]["exercise"] == ex
    assert "gate not exercised" in crit["reason"]
    exercised = _gate_arms(
        (0, 1, 2),
        [0.30, 0.32, 0.29],
        [0.10, 0.11, 0.12],
        [0.20, 0.21, 0.19],
        [0.20, 0.20, 0.20],
        blocks=[(True, True), (True, True), (False, False), (True, False)],
    )
    ex = gate_exercise(exercised, 4)
    assert ex["exercised"] and ex["problems"] == []
    crit = evaluate_criteria(_results(exercised))["gate"]
    assert crit["verdict"] == VERDICT_SUCCESS and "not exercised" not in crit["reason"]


def test_criteria_never_crash_on_empty_results() -> None:
    crit = evaluate_criteria({"arms": []})
    for name in ("gate", "subset", "operator", "low_start"):
        assert crit[name]["verdict"] == VERDICT_INCONCLUSIVE
    assert "pairing" in crit["definitions"]


def test_render_summary_has_banner_and_verdicts(smoke: tuple[PilotConfig, Path, dict[str, Any]]) -> None:
    _, _, results = smoke
    text = render_summary(results)
    assert text.count(BANNER) == 2
    assert "SMOKE RUN" in text
    for name in ("gate", "subset", "operator", "low_start"):
        assert f"| {name} | **INCONCLUSIVE** |" in text


# ============================================================================== CLI
def _load_cli() -> Any:
    path = ROOT / "examples" / "capability_pilot_tiny.py"
    spec = importlib.util.spec_from_file_location("capability_pilot_tiny_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_builds_configs(tmp_path: Path) -> None:
    cli = _load_cli()
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"oracle_threshold": 0.02, "updates": 16}), encoding="utf-8")
    args = cli.parse_args(
        ["--output-dir", str(tmp_path), "--config", str(cfg_file), "--seeds", "3,4,5", "--arms", "grpo,es_full+gate_on",
         "--variants", "clean", "--updates", "24", "--jobs", "2"]
    )  # fmt: skip
    config = cli.build_config(args)
    assert config.seeds == (3, 4, 5)
    assert config.arms == ("grpo+gate_off", "grpo+gate_on", "es_full+gate_on")
    assert config.variants == ("clean",)
    assert config.updates == 24 and config.jobs == 2 and config.oracle_threshold == 0.02
    assert not config.smoke
    smoke_args = cli.parse_args(["--smoke", "--output-dir", str(tmp_path)])
    smoke_cfg = cli.build_config(smoke_args)
    assert smoke_cfg == smoke_config()
