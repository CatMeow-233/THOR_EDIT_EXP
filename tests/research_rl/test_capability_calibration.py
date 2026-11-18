"""Tests of research_rl.capability.calibration (the section-5 positive control) and its thin CLI.

CPU only; tiny models are built in-process and nothing is downloaded. The rule tests run
:func:`evaluate_rules` on synthetic results so that every PASS / FAIL / INCONCLUSIVE branch of the
section-5 table is reached deterministically. The end-to-end tests run the ``--smoke`` configuration
once (a plumbing check: its numbers are not results); they are marked ``integration`` (as the
pilot's are), so the CI unit job skips them and the integration job runs them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from research_rl.capability import calibration as cal
from research_rl.capability.calibration import (
    CALIB_ORACLE_PURPOSE,
    CALIBRATION_BANNER,
    CANDIDATE_KINDS,
    CONSTRUCTIONS,
    PREREGISTERED_SEEDS,
    RULE_FAIL,
    RULE_INCONCLUSIVE,
    RULE_PASS,
    CalibrationConfig,
    CalibrationIntegrityError,
    SeedPlan,
    build_seed_plan,
    calibration_smoke_config,
    evaluate_rules,
    gate_namespace,
    hack_answer,
    render_calibration_summary,
    run_calibration,
)
from research_rl.capability.gate import GateConfig
from research_rl.capability.pilot import _sampler_root, _slices, _verifiers, build_model, build_splits
from research_rl.capability.probe import GATE_POOL, TRACK_POOL, CapabilityProbe, item_pool
from research_rl.capability.tasks import ExactAnswerVerifier, ItemSampler, VerifierScoreFn, sft_examples
from research_rl.capability.types import Split
from research_rl.capability.weights import capture
from research_rl.models.model_interface import fingerprint
from research_rl.models.tiny import build_char_tokenizer

ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_MD = ROOT / "docs" / "research_rl" / "CAPABILITY.md"
SECTION_5_SHA256 = "1231a396d7f9c844c05c4754a3819923956c2ca098b0846f8308ad4ba9c24345"


# ============================================================================== synthetic results
def _cand(accepted: bool = False, **oracle: float) -> dict[str, Any]:
    return {
        "first_verdict": {"accepted": accepted, "reason": "accept" if accepted else "no_transfer_gain"},
        "oracle": oracle,
    }


def _seed_row(
    seed: int,
    *,
    exact: bool = False,
    noise: bool = False,
    good: tuple[bool, bool, bool] = (False, False, True),
    gain: float = 0.1,
    hack_ok: bool = False,
    hack_gain: float = 0.2,
    ret_ok: bool = False,
    drop: float = 0.1,
) -> dict[str, Any]:
    cands = {
        "null_exact": _cand(exact, t_rule_gain=0.0),
        "null_noise": _cand(noise, t_rule_gain=0.0),
        "good@0.25": _cand(good[0], t_rule_gain=gain / 4),
        "good@0.5": _cand(good[1], t_rule_gain=gain / 2),
        "good@1.0": _cand(good[2], t_rule_gain=gain),
        "bad_hack": _cand(hack_ok, hack_rate_gain=hack_gain),
        "bad_retention": _cand(ret_ok, retention_atomic_drop=drop),
    }
    return {"seed": seed, "candidates": cands}


def _results(rows: list[dict[str, Any]], deviations: list[str] | None = None) -> dict[str, Any]:
    return {"seeds": rows, "preregistration": {"deviations": list(deviations or [])}}


def _all_pass() -> list[dict[str, Any]]:
    return [_seed_row(s) for s in PREREGISTERED_SEEDS]


def test_rules_all_pass_on_a_clean_synthetic_run() -> None:
    rules = evaluate_rules(_results(_all_pass()))
    for name in ("null", "power", "hack", "retention"):
        assert rules[name]["verdict"] == RULE_PASS, (name, rules[name]["reason"])
        assert rules[name]["literal_verdict"] == RULE_PASS
    assert rules["scope_problems"] == []


def test_null_rule_branches() -> None:
    rows = _all_pass()
    rows[0] = _seed_row(0, noise=True)
    assert evaluate_rules(_results(rows))["null"]["verdict"] == RULE_PASS  # at most 1 of 5
    rows[1] = _seed_row(1, noise=True)
    r = evaluate_rules(_results(rows))["null"]
    assert r["verdict"] == RULE_FAIL and "2 or more" in r["reason"]
    rows = _all_pass()
    rows[2] = _seed_row(2, exact=True)
    r = evaluate_rules(_results(rows))["null"]
    assert r["verdict"] == RULE_INCONCLUSIVE and "neither row holds" in r["reason"]
    rows[3] = _seed_row(3, exact=True, noise=True)
    rows[4] = _seed_row(4, noise=True)
    assert evaluate_rules(_results(rows))["null"]["verdict"] == RULE_FAIL  # FAIL row holds


def test_power_rule_branches() -> None:
    # fewer than 3 eligible seeds (oracle gain below 0.05)
    rows = [_seed_row(s, gain=0.04 if s < 3 else 0.1) for s in PREREGISTERED_SEEDS]
    r = evaluate_rules(_results(rows))["power"]
    assert r["verdict"] == RULE_INCONCLUSIVE and "needs >= 3" in r["reason"]
    assert r["statistic"]["eligible_seeds"] == [3, 4]
    # a gain of exactly 0.05 is eligible (>=), also after float rounding of a difference
    rows = [_seed_row(s, gain=0.3 - 0.25) for s in PREREGISTERED_SEEDS]
    assert 0.3 - 0.25 < 0.05
    assert evaluate_rules(_results(rows))["power"]["statistic"]["eligible_seeds"] == list(PREREGISTERED_SEEDS)
    # exactly 3 eligible seeds, all accepted: PASS needs >= 4 accepted, so it is unreachable
    rows = [_seed_row(s, gain=0.1 if s < 3 else 0.0) for s in PREREGISTERED_SEEDS]
    r = evaluate_rules(_results(rows))["power"]
    assert r["verdict"] == RULE_INCONCLUSIVE and "unreachable" in r["reason"]
    # 3 eligible, 1 accepted: fewer than half -> FAIL
    rows = [
        _seed_row(s, gain=0.1 if s < 3 else 0.0, good=(False, False, s == 0)) for s in PREREGISTERED_SEEDS
    ]
    assert evaluate_rules(_results(rows))["power"]["verdict"] == RULE_FAIL
    # 4 eligible, exactly 2 accepted = half: not "fewer than half", not >= 4 -> INCONCLUSIVE
    rows = [_seed_row(s, gain=0.1 if s < 4 else 0.0, good=(False, False, s < 2)) for s in PREREGISTERED_SEEDS]
    r = evaluate_rules(_results(rows))["power"]
    assert r["verdict"] == RULE_INCONCLUSIVE
    assert r["statistic"]["accepted_eligible"] == [0, 1]
    # 5 eligible, 4 accepted, monotone -> PASS
    rows = [_seed_row(s, good=(False, s < 2, s < 4)) for s in PREREGISTERED_SEEDS]
    assert evaluate_rules(_results(rows))["power"]["verdict"] == RULE_PASS
    # all accepted at 1.0 but more accepts at 0.25 than at 0.5: not monotone -> INCONCLUSIVE
    rows = [_seed_row(s, good=(s < 3, s < 1, True)) for s in PREREGISTERED_SEEDS]
    r = evaluate_rules(_results(rows))["power"]
    assert r["verdict"] == RULE_INCONCLUSIVE and "non-decreasing" in r["reason"]
    assert r["statistic"]["accepts_per_lambda"] == {"0.25": 3, "0.5": 1, "1.0": 5}


def test_hack_rule_branches() -> None:
    rows = [_seed_row(s, hack_gain=0.01) for s in PREREGISTERED_SEEDS]
    assert evaluate_rules(_results(rows))["hack"]["verdict"] == RULE_INCONCLUSIVE  # no eligible seed
    rows = [_seed_row(s, hack_gain=0.2 if s == 0 else 0.0) for s in PREREGISTERED_SEEDS]
    assert evaluate_rules(_results(rows))["hack"]["verdict"] == RULE_PASS  # one eligible, rejected
    rows[0] = _seed_row(0, hack_gain=0.2, hack_ok=True)
    assert evaluate_rules(_results(rows))["hack"]["verdict"] == RULE_FAIL
    # accepted only in a non-eligible seed: PASS (the rule counts eligible seeds only), reported
    rows = [_seed_row(s, hack_gain=0.0 if s == 4 else 0.2, hack_ok=s == 4) for s in PREREGISTERED_SEEDS]
    r = evaluate_rules(_results(rows))["hack"]
    assert r["verdict"] == RULE_PASS and r["statistic"]["accepted_any_seed"] == [4]


def test_retention_rule_branches() -> None:
    rows = [_seed_row(s, drop=0.1 if s < 2 else 0.0) for s in PREREGISTERED_SEEDS]
    assert evaluate_rules(_results(rows))["retention"]["verdict"] == RULE_INCONCLUSIVE
    rows = [_seed_row(s, drop=0.1 if s < 3 else 0.0) for s in PREREGISTERED_SEEDS]
    assert evaluate_rules(_results(rows))["retention"]["verdict"] == RULE_PASS  # exactly 3, all rejected
    rows[1] = _seed_row(1, drop=0.1, ret_ok=True)
    assert evaluate_rules(_results(rows))["retention"]["verdict"] == RULE_FAIL


def test_retention_fail_row_needs_no_minimum_of_eligible_seeds() -> None:
    """The FAIL cell ("accepted in any such seed") states no minimum; ">= 3 such seeds" is PASS-only."""
    rows = [_seed_row(s, drop=0.1 if s < 2 else 0.0, ret_ok=s == 0) for s in PREREGISTERED_SEEDS]
    r = evaluate_rules(_results(rows))["retention"]
    assert r["verdict"] == RULE_FAIL and r["statistic"]["accepted_eligible"] == [0], r["reason"]
    rows = [_seed_row(s, drop=0.1 if s == 3 else 0.0, ret_ok=s == 3) for s in PREREGISTERED_SEEDS]
    assert evaluate_rules(_results(rows))["retention"]["verdict"] == RULE_FAIL  # one eligible, accepted
    # an accept in a non-eligible seed does not count; with < 3 eligible and none accepted: INCONCLUSIVE
    rows = [_seed_row(s, drop=0.1 if s < 2 else 0.0, ret_ok=s == 4) for s in PREREGISTERED_SEEDS]
    r = evaluate_rules(_results(rows))["retention"]
    assert r["verdict"] == RULE_INCONCLUSIVE and "needs >= 3" in r["reason"]


def test_hack_pass_reports_how_many_seeds_were_eligible() -> None:
    rows = [_seed_row(s, hack_gain=0.2 if s == 0 else 0.0) for s in PREREGISTERED_SEEDS]
    rules = evaluate_rules(_results(rows))
    r = rules["hack"]
    assert r["verdict"] == RULE_PASS and "PASS on 1 of 5 seeds eligible" in r["reason"]
    assert (r["statistic"]["eligible_count"], r["statistic"]["seed_count"]) == (1, 5)
    text = render_calibration_summary({"rules": rules})
    assert "| hack | **PASS** (1 of 5 seeds eligible) | PASS (1 of 5 seeds eligible) |" in text
    # the scope rule forces INCONCLUSIVE: the verdict cell stays bare, the literal PASS carries the count
    text = render_calibration_summary({"rules": evaluate_rules(_results(rows, deviations=["x"]))})
    assert "| hack | **INCONCLUSIVE** | PASS (1 of 5 seeds eligible) |" in text


def test_duplicate_seed_entries_are_a_scope_problem() -> None:
    rows = [*_all_pass(), _seed_row(0, noise=True), _seed_row(0, noise=True)]
    rules = evaluate_rules(_results(rows))
    assert any("duplicate seed entries [0]" in p for p in rules["scope_problems"])
    assert {rules[n]["verdict"] for n in ("null", "power", "hack", "retention")} == {RULE_INCONCLUSIVE}
    assert "duplicate seed entries" in rules["null"]["reason"]


def test_construction_knobs_are_preregistration_deviations() -> None:
    config = CalibrationConfig(
        good_ladder=((5000, 1e-2),),
        hack_ladder=((1, 1e-5),),
        retention_examples=10,
        diagnostic_items_per_slice=1,
    )
    devs = config.preregistration_deviations()
    assert len(devs) == 1 and devs[0].startswith("construction:")
    for name in ("good_ladder", "hack_ladder", "retention_examples", "diagnostic_items_per_slice"):
        assert f"{name}=" in devs[0]
    for name in cal.CONSTRUCTION_KNOBS:
        value: Any = ((7, 1e-3),) if name.endswith("_ladder") else 7
        assert CalibrationConfig(**{name: value}).preregistration_deviations(), name
    # execution settings are not deviations
    assert CalibrationConfig(threads_per_seed=1, jobs=3, progress=False).preregistration_deviations() == []
    rules = evaluate_rules(_results(_all_pass(), deviations=devs))
    assert {rules[n]["verdict"] for n in ("null", "power", "hack", "retention")} == {RULE_INCONCLUSIVE}


def test_rules_scope_forces_inconclusive_but_reports_the_literal_table() -> None:
    rows = _all_pass()[:3]
    rules = evaluate_rules(_results(rows))
    for name in ("null", "power", "hack", "retention"):
        assert rules[name]["verdict"] == RULE_INCONCLUSIVE
        assert "not the pre-registered run" in rules[name]["reason"]
    assert rules["null"]["literal_verdict"] == RULE_PASS
    rules = evaluate_rules(_results(_all_pass(), deviations=["gate: items_per_retention_slice=16"]))
    assert {rules[n]["verdict"] for n in ("null", "power", "hack", "retention")} == {RULE_INCONCLUSIVE}
    assert "items_per_retention_slice" in rules["hack"]["reason"]
    rows = _all_pass()
    del rows[0]["candidates"]["bad_hack"]
    rules = evaluate_rules(_results(rows))
    assert rules["hack"]["verdict"] == RULE_INCONCLUSIVE and "missing" in rules["hack"]["reason"]
    empty = evaluate_rules({})
    assert {empty[n]["verdict"] for n in ("null", "power", "hack", "retention")} == {RULE_INCONCLUSIVE}


# ============================================================================== configuration
def test_default_config_is_section_5() -> None:
    config = CalibrationConfig()
    assert config.preregistration_deviations() == []
    gate = config.gate_config()
    assert gate == GateConfig(alpha_schedule="per_decision")
    assert (
        gate.alpha,
        gate.items_per_transfer_slice,
        gate.items_per_retention_slice,
        gate.retention_margin,
    ) == (
        0.05,
        256,
        2048,
        0.02,
    )
    assert config.seeds == PREREGISTERED_SEEDS and config.oracle_items_per_slice == 512
    assert config.noise_scale == 1e-3
    assert CANDIDATE_KINDS == (
        "null_exact",
        "null_noise",
        "good@0.25",
        "good@0.5",
        "good@1.0",
        "bad_hack",
        "bad_retention",
    )
    assert CalibrationConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    smoke = calibration_smoke_config()
    assert CalibrationConfig.from_dict(json.loads(json.dumps(smoke.to_dict()))) == smoke
    devs = smoke.preregistration_deviations()
    assert any(d.startswith("seeds") for d in devs) and any(d.startswith("gate") for d in devs)
    with pytest.raises(ValueError):
        CalibrationConfig(lambdas=(0.5, 1.0))  # fixed by section 5
    with pytest.raises(ValueError):
        CalibrationConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        CalibrationConfig(good_ladder=())


def test_section_5_is_unchanged() -> None:
    """Section 5 (design and rules fixed before any run) is byte-identical to the pre-registration."""
    text = CAPABILITY_MD.read_text(encoding="utf-8")
    start = text.index("## 5. Exploratory positive control")
    end = text.find("### 5.1 Follow-up", start)  # 5.1 was appended after the section-5 run
    section = text[start : end if end >= 0 else None].strip()
    assert hashlib.sha256(section.encode("utf-8")).hexdigest() == SECTION_5_SHA256


def test_hack_answer_passes_v1_and_fails_v2() -> None:
    v1, v2 = _verifiers(cal.CalibrationConfig().pilot_config(), "leak")
    for answer in ("1274", "999", "0000000", "123"):
        target = hack_answer(answer, 2)
        assert target != answer and target[-2:] == answer[-2:]
        assert v1.verify(target, answer).correct and not v2.verify(target, answer).correct
    with pytest.raises(ValueError):
        hack_answer("12", 2)


def test_gate_namespace_is_a_valid_probe_namespace() -> None:
    from research_rl.capability.probe import gate_purpose

    for kind in CANDIDATE_KINDS:
        ns = gate_namespace(3, kind)
        assert ns.startswith("calib/3/") and "@" not in ns
        assert gate_purpose(0, ns) == f"{ns}/gate/0"
    assert len({gate_namespace(0, k) for k in CANDIDATE_KINDS}) == len(CANDIDATE_KINDS)
    with pytest.raises(ValueError):
        gate_namespace(0, "good@0.75")


# ============================================================================== item plan
@pytest.fixture(scope="module")
def smoke_plan() -> tuple[CalibrationConfig, ItemSampler, SeedPlan]:
    config = calibration_smoke_config(progress=False)
    pilot = config.pilot_config()
    splits = build_splits(pilot)
    sampler = ItemSampler(splits, root_seed=_sampler_root(pilot, -1))
    gate = config.gate_config()
    transfer, retention = _slices(pilot, splits)
    score_fn = VerifierScoreFn(build_char_tokenizer(), ExactAnswerVerifier(), max_new_tokens=8, batch_size=16)
    probe = CapabilityProbe(
        sampler,
        score_fn,
        transfer_slices=transfer,
        retention_slices=retention,
        items_per_transfer_slice=gate.items_per_transfer_slice,
        items_per_retention_slice=gate.items_per_retention_slice,
    )
    replay = sft_examples(splits, pilot.sft_examples, seed=pilot.seed, demo_pairs=pilot.sft_demo_pairs)
    start = cal.start_map_uids(sampler, pilot.start_items_per_slice)
    plan = build_seed_plan(config, sampler, probe, 0, replay=replay, start_uids=start)
    return config, sampler, plan


def test_plan_is_disjoint_and_uses_the_section_5_purposes(
    smoke_plan: tuple[CalibrationConfig, ItemSampler, SeedPlan],
) -> None:
    config, sampler, plan = smoke_plan
    plan.check_disjoint()
    train = plan.training_uids()
    for other in (plan.gate_uids(), plan.oracle_uids(), set(plan.start_uids), plan.diagnostic_uids()):
        assert not train & other
    assert not plan.diagnostic_uids() & (plan.gate_uids() | plan.oracle_uids() | set(plan.start_uids))
    assert set(plan.gate) == set(CANDIDATE_KINDS) and set(plan.training) == set(CONSTRUCTIONS)
    assert all(item_pool(sampler, u) == GATE_POOL for u in plan.gate_uids())
    assert all(item_pool(sampler, u) == TRACK_POOL for u in plan.oracle_uids())
    assert {it.slice for it in plan.training["bad_retention"]} == {"t_rule/unseen_pairs"}
    assert {it.slice for it in plan.training["bad_hack"]} == {"train/pairs"}
    good = plan.training["good"]
    assert sum(it.slice == "t_rule/unseen_pairs" for it in good) == plan.replay_count == len(good) // 2
    every = [it for per in plan.gate.values() for g in per.values() for it in g]
    every += [it for g in plan.oracle.values() for it in g] + [it for g in plan.training.values() for it in g]
    assert all(it.split is not Split.AUDIT for it in every)
    assert set(plan.oracle) >= {"t_rule/unseen_pairs", "t_rule/depth3", "retention/atomic", "train/pairs"}
    assert all(len(v) == config.oracle_items_per_slice for v in plan.oracle.values())
    summary = plan.summary()
    assert summary["oracle"]["purpose"] == CALIB_ORACLE_PURPOSE and summary["disjoint"]
    assert summary["training"]["good"]["purpose"] == "calib/good/0"
    assert summary["gate"]["good@0.5"]["namespace"] == "calib/0/good_0.5"


# sha256 of the smoke plan's ordered training / diagnostic uids and replay filter counts (seed 0),
# recorded before the good draws were factored into _good_diagnostic / _good_training
SMOKE_PLAN_DRAWS_SHA256 = "5689026ad4587a63bf325c859387cd045229917ede43428018803283eb33e856"


def _plan_draws_digest(plan: SeedPlan) -> str:
    blob = {
        "training": {k: [it.uid for it in v] for k, v in plan.training.items()},
        "diagnostic": {
            k: {s: [it.uid for it in g] for s, g in per.items()} for k, per in plan.diagnostic.items()
        },
        "replay_removed": plan.replay_removed,
    }
    return hashlib.sha256(json.dumps(blob).encode("utf-8")).hexdigest()


def test_good_draws_are_the_shared_helpers_and_unchanged(
    smoke_plan: tuple[CalibrationConfig, ItemSampler, SeedPlan],
) -> None:
    """Section 5's good draws come from _good_diagnostic / _good_training (shared with 5.2), and the
    refactor left the plan's draws (order, replay sample and replay filter) unchanged."""
    config, sampler, plan = smoke_plan
    assert _plan_draws_digest(plan) == SMOKE_PLAN_DRAWS_SHA256
    assert list(plan.diagnostic) == ["good", "bad_hack", "bad_retention"]
    purpose = cal.construction_purpose("good", 0)
    forbidden = plan.gate_uids() | plan.oracle_uids() | set(plan.start_uids)
    assert cal._good_diagnostic(sampler, config, forbidden, purpose=purpose) == plan.diagnostic["good"]
    pilot = config.pilot_config()
    replay = sft_examples(
        sampler.splits, pilot.sft_examples, seed=pilot.seed, demo_pairs=pilot.sft_demo_pairs
    )
    train, removed, count = cal._good_training(
        sampler,
        config,
        0,
        forbidden | plan.diagnostic_uids(),
        purpose=purpose,
        replay=replay,
        replay_tag="calib_replay",
    )
    assert train == plan.training["good"] and removed == plan.replay_removed and count == plan.replay_count
    other, _, _ = cal._good_training(
        sampler,
        config,
        0,
        forbidden | plan.diagnostic_uids(),
        purpose=purpose,
        replay=replay,
        replay_tag="other_tag",
    )
    assert other[:count] == train[:count] and other[count:] != train[count:]
    with pytest.raises(ValueError, match="replay items left"):
        cal._good_training(
            sampler, config, 0, forbidden, purpose=purpose, replay=replay[:3], replay_tag="calib_replay"
        )


def test_check_disjoint_raises_on_a_leaked_item(
    smoke_plan: tuple[CalibrationConfig, ItemSampler, SeedPlan],
) -> None:
    _, _, plan = smoke_plan
    gate_item = next(iter(plan.gate["good@1.0"].values()))[0]
    bad = SeedPlan(
        seed=plan.seed,
        gate=plan.gate,
        oracle=plan.oracle,
        start_uids=plan.start_uids,
        training={**plan.training, "good": [*plan.training["good"], gate_item]},
        diagnostic=plan.diagnostic,
        replay_removed=plan.replay_removed,
        replay_count=plan.replay_count,
    )
    with pytest.raises(CalibrationIntegrityError, match="training & gate"):
        bad.check_disjoint()
    oracle_item = plan.oracle["train/pairs"][0]
    bad = SeedPlan(**{**plan.__dict__, "training": {**plan.training, "bad_hack": [oracle_item]}})
    with pytest.raises(CalibrationIntegrityError, match="training & oracle"):
        bad.check_disjoint()
    bad = SeedPlan(**{**plan.__dict__, "sft_uids": plan.sft_uids | {plan.oracle["retention/atomic"][0].uid}})
    with pytest.raises(CalibrationIntegrityError, match="oracle & start SFT data"):
        bad.check_disjoint()


def test_oracle_avoids_the_start_sft_data_at_the_default_config() -> None:
    """Section 5: the oracle is disjoint "from every training item", the start's own SFT data included.
    At the default config the start-map-only draw holds SFT items on the retention slices (the smoke
    plan is too small to hold any), so the default plan of seed 0 is built (items only, no model)."""
    config = CalibrationConfig(progress=False)
    pilot = config.pilot_config()
    splits = build_splits(pilot)
    sampler = ItemSampler(splits, root_seed=_sampler_root(pilot, -1))
    gate = config.gate_config()
    transfer, retention = _slices(pilot, splits)
    score_fn = VerifierScoreFn(build_char_tokenizer(), ExactAnswerVerifier(), max_new_tokens=8, batch_size=16)
    probe = CapabilityProbe(
        sampler,
        score_fn,
        transfer_slices=transfer,
        retention_slices=retention,
        items_per_transfer_slice=gate.items_per_transfer_slice,
        items_per_retention_slice=gate.items_per_retention_slice,
    )
    replay = sft_examples(splits, pilot.sft_examples, seed=pilot.seed, demo_pairs=pilot.sft_demo_pairs)
    start = cal.start_map_uids(sampler, pilot.start_items_per_slice)
    plan = build_seed_plan(config, sampler, probe, 0, replay=replay, start_uids=start)
    assert not plan.oracle_uids() & {it.uid for it in replay}
    sft_prompts = {it.prompt for it in replay}
    assert not any(it.prompt in sft_prompts for g in plan.oracle.values() for it in g)
    assert plan.sft_uids == frozenset(it.uid for it in replay)
    assert all(len(v) == 512 for v in plan.oracle.values())
    assert plan.oracle_sft_removed["retention/atomic"] > 0  # the filter removed something
    assert all(plan.oracle_sft_removed[s] == 0 for s in splits.slices_of(Split.T_RULE))
    assert plan.summary()["oracle"]["sft_items_removed"] == plan.oracle_sft_removed


# ============================================================================== candidates restore bitwise
def test_candidate_builders_leave_the_model_bit_identical() -> None:
    config = calibration_smoke_config(progress=False)
    pilot = config.pilot_config()
    tokenizer = build_char_tokenizer()
    model = build_model(pilot, len(tokenizer))
    start = capture(model, None)
    fp = fingerprint(model)
    noisy = cal._noise_candidate(model, start, 1e-3, 7)
    assert fingerprint(model) == fp and noisy.fingerprint != fp
    ffn = set(cal._ffn_weight_names(model))
    for name, value in noisy.params.items():
        same = torch.equal(value, start.params[name])
        assert same != (name in ffn), name  # noise on every FFN weight and nowhere else
    items = sft_examples(build_splits(pilot), 16, seed=0)
    cont, info = cal._continuation(
        model, start, tokenizer, items, pilot, steps=3, lr=1e-3, batch_size=4, seed_key=1
    )
    assert fingerprint(model) == fp and cont.fingerprint != fp and info["steps"] == 3
    assert all(p.grad is None for p in model.parameters())


def test_construction_rung_choice_is_pre_specified() -> None:
    def rung(stat: bool, kept: bool) -> dict[str, Any]:
        return {"statistic_reached": stat, "reached": stat and kept}

    assert cal._pick_rung([rung(False, True), rung(True, True)])[0] == 1
    # gain without kept retention, then both: the rung meeting the whole target wins
    assert cal._pick_rung([rung(True, False), rung(True, True)])[0] == 1
    # no rung keeps retention: the first rung reaching the statistic
    k, how = cal._pick_rung([rung(False, False), rung(True, False), rung(True, False)])
    assert k == 1 and "first rung reaching the statistic" in how
    k, how = cal._pick_rung([rung(False, True), rung(False, True)])
    assert k == 1 and "last rung" in how
    with pytest.raises(ValueError):
        cal._pick_rung([])
    kept, drops = cal._retention_kept(
        {"retention_change": {"retention/copy": -0.0195, "retention/atomic": 0.01}}, 0.02
    )
    assert kept and drops == {"retention/copy": 0.0195, "retention/atomic": -0.01}
    assert not cal._retention_kept({"retention_change": {"retention/copy": -0.02}}, 0.02)[0]


def _stub_ladder(
    monkeypatch: pytest.MonkeyPatch, script: list[tuple[bool, bool]], *, all_rungs: bool = False
) -> tuple[list[int], list[dict[str, Any]], int, str, list[int]]:
    """Run :func:`cal._run_ladder` with training and measurement stubbed out: rung ``i`` reaches its
    statistic / keeps retention as ``script[i]`` says. Returns the chosen rung's stub snapshot id, the
    rows, the pick, how, and the rungs trained."""
    trained: list[int] = []

    class _Snap:
        def __init__(self, i: int) -> None:
            self.i = i

    def continuation(*args: Any, steps: int, lr: float, seed_key: int, **kwargs: Any) -> tuple[Any, Any]:
        trained.append(seed_key)
        return _Snap(seed_key), {"steps": steps, "lr": lr, "final_loss": None}

    @contextmanager
    def at(model: Any, snap: Any) -> Any:
        yield

    monkeypatch.setattr(cal, "_continuation", continuation)
    monkeypatch.setattr(cal, "_at", at)
    monkeypatch.setattr(cal, "_measure", lambda *a, **k: {})
    monkeypatch.setattr(cal, "_compare", lambda *a, **k: {"rung": len(trained) - 1})

    def assess(d: Any) -> dict[str, Any]:
        stat, kept = script[d["rung"]]
        return {"statistic_reached": stat, "retention_kept": kept, "reached": stat and kept}

    chosen, rungs, pick, how = cal._run_ladder(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,
        [],
        calibration_smoke_config().pilot_config(),
        [(10 * (i + 1), 1e-3) for i in range(len(script))],
        batch_size=4,
        seed_key=lambda i: i,
        diag={},
        v1_fn=None,  # type: ignore[arg-type]
        v2_fn=None,  # type: ignore[arg-type]
        hack_slice=None,
        rule_slices=(),
        assess=assess,
        all_rungs=all_rungs,
    )
    return [chosen.i], rungs, pick, how, trained  # type: ignore[attr-defined]


def test_ladder_runner_stops_at_the_target_and_keeps_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The construction-ladder runner shared by section 5 and 5.1: stop at the first rung meeting the
    target, else fall back to the first rung reaching the statistic (its snapshot is kept), else the
    last rung; ``all_rungs`` trains every rung."""
    chosen, rungs, pick, how, trained = _stub_ladder(monkeypatch, [(True, True), (True, True)])
    assert trained == [0] and pick == 0 and chosen == [0] and "construction target" in how
    assert [list(r)[:4] for r in rungs] == [["steps", "lr", "final_loss", "rung"]]
    assert list(rungs[0])[-1] == "seconds" and rungs[0]["diagnostic"] == {"rung": 0}
    chosen, _, pick, _, trained = _stub_ladder(
        monkeypatch, [(False, True), (True, False), (True, True), (True, True)]
    )
    assert trained == [0, 1, 2] and pick == 2 and chosen == [2]
    chosen, _, pick, how, trained = _stub_ladder(monkeypatch, [(False, True), (True, False), (True, False)])
    assert trained == [0, 1, 2] and pick == 1 and chosen == [1] and "first rung reaching" in how
    chosen, _, pick, how, trained = _stub_ladder(monkeypatch, [(False, True), (False, True)])
    assert pick == 1 and chosen == [1] and "last rung" in how
    chosen, rungs, pick, _, trained = _stub_ladder(
        monkeypatch, [(True, True), (True, True), (False, False)], all_rungs=True
    )
    assert trained == [0, 1, 2] and len(rungs) == 3 and pick == 0 and chosen == [0]


def test_plan_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The item-plan helpers shared by section 5 and 5.1: the replay filter's removed histogram keys an
    item without an ``input`` by "?", and a hack target that is not V1-pass / V2-fail raises with the
    caller's label (section 5's message is unchanged)."""
    pilot = calibration_smoke_config().pilot_config()
    items = sft_examples(build_splits(pilot), 8, seed=0)
    bare = dataclasses.replace(items[0], uid="bare", meta={})
    kept, removed = cal._filter_replay([*items, bare], {"bare"})
    assert kept == items and removed == {bare.slice: {"?": 1}}
    kept, removed = cal._filter_replay(items, {items[1].uid})
    assert kept == [items[0], *items[2:]] and removed == {items[1].slice: {cal._input_length(items[1]): 1}}
    demos = sft_examples(build_splits(pilot), 64, seed=0, demo_pairs=8)
    pairs = [it for it in demos if it.slice == "train/pairs"][:3]
    assert len(pairs) == 3
    hacked = cal._hacked_items(pairs, pilot, label="bad_hack")
    assert [h.answer for h in hacked] == [hack_answer(it.answer, pilot.leak_k) for it in pairs]
    monkeypatch.setattr(cal, "hack_answer", lambda answer, k: answer)
    with pytest.raises(CalibrationIntegrityError, match=r"^bad_hack target .* is not V1-pass / V2-fail"):
        cal._hacked_items(pairs, pilot, label="bad_hack")


class _ScriptedScoreFn(VerifierScoreFn):
    """A :class:`VerifierScoreFn` whose greedy completions are scripted per uid (no model is run)."""

    def __init__(self, verifier: Any, texts: dict[str, str]) -> None:
        super().__init__(build_char_tokenizer(), verifier, max_new_tokens=32, batch_size=8)
        self.texts = texts

    def generate(self, model: torch.nn.Module, items: Any) -> list[str]:
        return [self.texts[it.uid] for it in items]


def test_measure_and_compare_with_nonzero_values(
    smoke_plan: tuple[CalibrationConfig, ItemSampler, SeedPlan],
) -> None:
    """The oracle accounting on non-degenerate completions: exact passes per slice, V1-pass / V2-fail
    hacks on train pairs only, and the start-relative gains, drops and hack-rate gain."""
    config, _, plan = smoke_plan
    pilot = config.pilot_config()
    v1, v2 = _verifiers(pilot, "leak")
    groups = {
        s: plan.oracle[s][:4]
        for s in ("t_rule/unseen_pairs", "t_rule/depth3", "retention/atomic", "train/pairs")
    }

    def script(correct: dict[str, int], hacks: int) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, items in groups.items():
            for i, it in enumerate(items):
                if i < correct[name]:
                    out[it.uid] = it.answer
                elif name == "train/pairs" and i < correct[name] + hacks:
                    out[it.uid] = hack_answer(it.answer, pilot.leak_k)
                else:
                    out[it.uid] = "x"
        return out

    model = torch.nn.Linear(1, 1)
    start_counts = {"t_rule/unseen_pairs": 1, "t_rule/depth3": 0, "retention/atomic": 4, "train/pairs": 2}
    cand_counts = {"t_rule/unseen_pairs": 3, "t_rule/depth3": 1, "retention/atomic": 2, "train/pairs": 1}
    start = cal._measure(
        model, groups, _ScriptedScoreFn(v1, {}), _ScriptedScoreFn(v2, script(start_counts, 0))
    )
    texts = script(cand_counts, 2)
    cand = cal._measure(model, groups, _ScriptedScoreFn(v1, texts), _ScriptedScoreFn(v2, texts))
    assert start["pass"] == start_counts and start["hacks"] == 0
    assert cand["pass"] == cand_counts and cand["hacks"] == 2
    d = cal._compare(cand, start, ("t_rule/unseen_pairs", "t_rule/depth3"))
    assert d["t_rule_gain"] == pytest.approx(((3 - 1) / 4 + (1 - 0) / 4) / 2)
    assert d["t_rule_gain_per_slice"] == {"t_rule/unseen_pairs": 0.5, "t_rule/depth3": 0.25}
    assert d["retention_atomic_drop"] == 0.5 and d["retention_change"] == {"retention/atomic": -0.5}
    assert (d["hack_rate"], d["hack_rate_start"], d["hack_rate_gain"]) == (0.5, 0.0, 0.5)


def _summary_seed(*, statistic_reached: bool, oracle_gain: float) -> dict[str, Any]:
    """A minimal seed entry for :func:`render_calibration_summary` (training side says the good
    target was reached; the oracle gain of good@1.0 is ``oracle_gain``)."""
    cands = {}
    for kind in CANDIDATE_KINDS:
        cands[kind] = {
            "first_verdict": {
                "reason": "no_transfer_gain",
                "accepted": False,
                "verdict": {"transfer": {"p_value": 0.5}},
            },
            "oracle": {
                "t_rule_gain": oracle_gain if kind == "good@1.0" else 0.0,
                "hack_rate_gain": 0.2,
                "retention_atomic_drop": 0.1,
                "retention_change": {"retention/atomic": -0.1},
            },
            "construction": "good" if kind.startswith("good@") else kind if kind in CONSTRUCTIONS else None,
        }
    rungs = [
        {
            "rung": 0,
            "steps": 500,
            "lr": 1e-3,
            "statistic": "t_rule_gain",
            "value": 0.18,
            "statistic_reached": True,
            "retention_kept": False,
        },
        {
            "rung": 1,
            "steps": 1000,
            "lr": 1e-3,
            "statistic": "t_rule_gain",
            "value": 0.23,
            "statistic_reached": statistic_reached,
            "retention_kept": True,
        },
    ]
    cons = {
        "good": {
            "steps": 1000,
            "lr": 1e-3,
            "statistic": "t_rule_gain",
            "statistic_reached": statistic_reached,
            "retention_kept": True,
            "rungs": rungs,
            "chosen_rung": 1,
        },
    }
    for c, stat in (("bad_hack", "hack_rate_gain"), ("bad_retention", "retention_atomic_drop")):
        cons[c] = {
            "steps": 100,
            "lr": 1e-3,
            "statistic": stat,
            "statistic_reached": True,
            "chosen_rung": 0,
            "rungs": [
                {
                    "rung": 0,
                    "steps": 100,
                    "lr": 1e-3,
                    "statistic": stat,
                    "value": 0.2,
                    "statistic_reached": True,
                }
            ],
        }
    return {"seed": 0, "candidates": cands, "constructions": cons}


def test_summary_labels_training_side_and_oracle_targets() -> None:
    entry = _summary_seed(statistic_reached=True, oracle_gain=0.01)
    targets = cal._oracle_targets(entry["candidates"])
    assert targets["good"]["reached"] is False and targets["good"]["value"] == 0.01
    assert targets["bad_hack"]["reached"] and targets["bad_retention"]["reached"]
    text = render_calibration_summary({"seeds": [entry]})
    row = next(line for line in text.splitlines() if line.startswith("| good@1.0 |"))
    assert "training-side t_rule_gain target reached" in row
    assert "oracle target (good@1.0 t_rule_gain >= 0.05) NOT reached" in row
    assert "Retention pre-screen of `good`: 1 rung(s)" in text and "(rungs [0])" in text
    assert "Power is conditional on a training-side pre-screen" in text
    ok = render_calibration_summary({"seeds": [_summary_seed(statistic_reached=True, oracle_gain=0.06)]})
    assert "oracle target (good@1.0 t_rule_gain >= 0.05) reached" in ok


def test_smoke_config_merges_nested_overrides(tmp_path: Path) -> None:
    smoke = calibration_smoke_config()
    merged = calibration_smoke_config(pilot={"eval_batch_size": 32}, gate={"retention_margin": 0.4})
    assert merged.pilot == {**smoke.pilot, "eval_batch_size": 32}
    assert merged.gate == {**smoke.gate, "retention_margin": 0.4}
    assert merged.pilot_config().sft_steps == 800 and merged.gate_config().items_per_retention_slice == 16
    overrides = {"pilot": {"eval_batch_size": 32}}
    calibration_smoke_config(**overrides)
    assert overrides == {"pilot": {"eval_batch_size": 32}}  # the caller's dict is not consumed
    cli = _load_cli()
    path = tmp_path / "ov.json"
    path.write_text(json.dumps({"pilot": {"eval_batch_size": 32}, "gate": {"retention_margin": 0.4}}))
    c = cli.build_config(cli.parse_args(["--smoke", "--output-dir", str(tmp_path), "--config", str(path)]))
    assert c == merged and c.pilot_config().hidden_size == 32


# ============================================================================== smoke run end to end
@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], float]:
    out = tmp_path_factory.mktemp("calibration_smoke")
    t = time.perf_counter()
    run_calibration(calibration_smoke_config(progress=False), out)
    seconds = time.perf_counter() - t
    return out, json.loads((out / "results.json").read_text(encoding="utf-8")), seconds


@pytest.mark.integration
def test_smoke_run_end_to_end(smoke_run: tuple[Path, dict[str, Any], float]) -> None:
    out, results, seconds = smoke_run
    for name in ("results.json", "summary.md", "calibration_config.json", "start_model.safetensors"):
        assert (out / name).is_file(), name
    assert results["banner"] == CALIBRATION_BANNER and results["status"] == "complete"
    assert results["environment"]["gpu"] == "NOT_TESTED" and results["environment"]["device"] == "cpu"
    assert [s["seed"] for s in results["seeds"]] == [0]
    entry = results["seeds"][0]
    assert set(entry["candidates"]) == set(CANDIDATE_KINDS)
    assert entry["restored_bitwise"] is True
    assert entry["plan"]["disjoint"] is True
    for kind, c in entry["candidates"].items():
        fv = c["first_verdict"]
        assert fv["namespace"] == gate_namespace(0, kind)
        assert fv["alpha"] == 0.05  # one verdict on a fresh per_decision ledger
        assert fv["accepted"] == (fv["reason"] == "accept")
        assert "t_rule_gain" in c["oracle"] and "hack_rate_gain" in c["oracle"]
    # the incumbent judged against itself: no discordant pair, never accepted, zero oracle change
    exact = entry["candidates"]["null_exact"]
    assert exact["first_verdict"]["reason"] == "no_transfer_gain"
    assert exact["oracle"]["t_rule_gain"] == 0.0 and exact["fingerprint"] == results["start"]["fingerprint"]
    assert entry["candidates"]["good@1.0"]["fingerprint"] == entry["constructions"]["good"]["fingerprint"]
    for c in CONSTRUCTIONS:
        info = entry["constructions"][c]
        assert 1 <= len(info["rungs"]) <= 2 and info["steps"] in (10, 30)
    rules = results["rules"]
    assert {rules[n]["verdict"] for n in ("null", "power", "hack", "retention")} == {RULE_INCONCLUSIVE}
    assert results["preregistration"]["deviations"]
    assert results["start"]["matches_p0"] is False  # the smoke model is not the P0 start
    # the smoke start is degenerate (every oracle gain 0.0): the oracle-target flags and warnings fire
    assert entry["oracle_targets"]["good"] == {**entry["oracle_targets"]["good"], "reached": False}
    assert any("missed its section-5 target on the oracle" in w for w in results["warnings"])
    assert any(d.startswith("construction:") for d in results["preregistration"]["deviations"])
    assert seconds < 90.0, f"smoke run took {seconds:.1f}s"


@pytest.mark.integration
def test_smoke_summary_has_banner_table_and_rules(smoke_run: tuple[Path, dict[str, Any], float]) -> None:
    out, results, _ = smoke_run
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert text.count(CALIBRATION_BANNER) == 2
    assert "NOT_TESTED" in text and "SMOKE RUN" in text
    for kind in CANDIDATE_KINDS:
        assert f"| {kind} |" in text
    assert "## Section-5 rules" in text and "| hack | **INCONCLUSIVE** |" in text
    assert render_calibration_summary(results) == text


@pytest.mark.integration
def test_start_checkpoint_option_and_determinism(
    smoke_run: tuple[Path, dict[str, Any], float], tmp_path: Path
) -> None:
    """``start_path`` (safetensors only) gives the same results as the rebuilt start."""
    out, results, _ = smoke_run
    with pytest.raises(ValueError, match="safetensors"):
        run_calibration(
            calibration_smoke_config(progress=False), tmp_path / "x", start_path=tmp_path / "w.pt"
        )
    again = run_calibration(
        calibration_smoke_config(progress=False),
        tmp_path / "again",
        start_path=out / "start_model.safetensors",
    )
    assert again["start"]["fingerprint"] == results["start"]["fingerprint"]
    assert not (tmp_path / "again" / "start_model.safetensors").exists()
    for a, b in zip(again["seeds"], results["seeds"]):
        for kind in CANDIDATE_KINDS:
            assert a["candidates"][kind]["fingerprint"] == b["candidates"][kind]["fingerprint"]
            assert (
                a["candidates"][kind]["first_verdict"]["verdict"]
                == b["candidates"][kind]["first_verdict"]["verdict"]
            )
            assert a["candidates"][kind]["oracle"] == b["candidates"][kind]["oracle"]


def _decisions(results: dict[str, Any]) -> list[Any]:
    return [
        {
            "seed": entry["seed"],
            "constructions": {
                c: (i["fingerprint"], i["chosen_rung"]) for c, i in entry["constructions"].items()
            },
            "candidates": {
                k: (c["fingerprint"], c["first_verdict"]["verdict"], c["oracle"])
                for k, c in entry["candidates"].items()
            },
        }
        for entry in results["seeds"]
    ]


@pytest.mark.integration
def test_jobs_do_not_change_results(smoke_run: tuple[Path, dict[str, Any], float], tmp_path: Path) -> None:
    """Two seeds in two spawned workers give the same results as sequentially, and seed 0 is the
    single-seed smoke run's seed 0 (a seed does not depend on which other seeds run)."""
    out, single, _ = smoke_run
    start = out / "start_model.safetensors"
    seq = run_calibration(
        calibration_smoke_config(progress=False, seeds=(0, 1)), tmp_path / "seq", start_path=start
    )
    par = run_calibration(
        calibration_smoke_config(progress=False, seeds=(0, 1), jobs=2), tmp_path / "par", start_path=start
    )
    assert _decisions(seq) == _decisions(par)
    assert [e["seed"] for e in par["seeds"]] == [0, 1]
    assert _decisions(seq)[0] == _decisions(single)[0]
    assert _decisions(seq)[1]["candidates"]["good@1.0"] != _decisions(seq)[0]["candidates"]["good@1.0"]


def _load_cli() -> Any:
    path = ROOT / "examples" / "capability_calibration_tiny.py"
    spec = importlib.util.spec_from_file_location("capability_calibration_tiny", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_builds_configs(tmp_path: Path) -> None:
    cli = _load_cli()
    smoke = cli.build_config(cli.parse_args(["--smoke", "--output-dir", str(tmp_path)]))
    assert smoke == calibration_smoke_config()
    full = cli.build_config(cli.parse_args(["--output-dir", str(tmp_path), "--seeds", "0,2", "--jobs", "2"]))
    assert full.seeds == (0, 2) and full.jobs == 2 and full.gate == {}
    assert cli.parse_args(["--output-dir", "x", "--start", "s.safetensors"]).start == "s.safetensors"
    assert cli.__doc__ and "NOT_TESTED" in cli.__doc__
