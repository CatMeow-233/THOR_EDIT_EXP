"""Tests of research_rl.capability.calibration_specificity (the section-5.2 alarm-specificity follow-up)
and its CLI.

CPU only; tiny models are built in-process and nothing is downloaded. The rule tests run
:func:`evaluate_rules_52` on synthetic results so that every PASS / FAIL / INCONCLUSIVE branch of the
5.2 table is reached deterministically. The end-to-end tests run the ``--smoke`` configuration (a
plumbing check: its numbers are not results) and are marked ``integration``, as sections 5 and 5.1's are.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from research_rl.capability import calibration as cal
from research_rl.capability import calibration_alarm as ca
from research_rl.capability import calibration_specificity as cs
from research_rl.capability.alarm import DisagreementAlarm
from research_rl.capability.calibration import (
    PREREGISTERED_SEEDS,
    RULE_FAIL,
    RULE_INCONCLUSIVE,
    RULE_PASS,
    CalibrationConfig,
    CalibrationIntegrityError,
)
from research_rl.capability.calibration_alarm import (
    ALARM_ITEMS_PER_REPORT,
    BASELINE_REPORTS,
    CANDIDATE_REPORTS,
    combined_decision,
)
from research_rl.capability.calibration_specificity import (
    CALIB52_ORACLE_PURPOSE,
    CALIBRATION51_RESULTS,
    CALIBRATION52_BANNER,
    KINDS_52,
    LAMBDAS_52,
    RULES_52,
    CalibrationSpecificityConfig,
    SeedPlan52,
    alarm_purpose_52,
    build_seed_plan_52,
    calibration52_smoke_config,
    construction_purpose_52,
    descriptive_52,
    evaluate_rules_52,
    gate_namespace_52,
    hack_stream_gaps_51,
    render_calibration52_summary,
    run_calibration_52,
)
from research_rl.capability.pilot import _sampler_root, _slices, build_splits
from research_rl.capability.probe import GATE_POOL, TRACK_POOL, CapabilityProbe, item_pool
from research_rl.capability.tasks import ExactAnswerVerifier, ItemSampler, VerifierScoreFn, sft_examples
from research_rl.capability.types import Split
from research_rl.models.tiny import build_char_tokenizer

ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_MD = ROOT / "docs" / "research_rl" / "CAPABILITY.md"
SECTION_5_2_SHA256 = "f884195c95f15ff400435df3d86d02028b313d61a2209d2762e05821e32893ed"


def test_section_5_2_is_unchanged() -> None:
    """Section 5.2 (design and rules fixed before any run) is byte-identical to the pre-registration:
    from its heading to the next follow-up heading (a later ``### 5.3``) or the end of CAPABILITY.md,
    stripped."""
    text = CAPABILITY_MD.read_text(encoding="utf-8")
    start = text.index("### 5.2 Follow-up")
    end = text.find("### 5.3", start)
    section = text[start : end if end >= 0 else None].strip()
    assert hashlib.sha256(section.encode("utf-8")).hexdigest() == SECTION_5_2_SHA256


def test_as_built_note_precedes_section_5_and_keeps_the_hash_anchors_unique() -> None:
    """The 5.2 as-built note (4.6) sits before section 5 and does not repeat the headings the 5 / 5.1 /
    5.2 hash tests find with ``str.index`` (the first occurrence)."""
    text = CAPABILITY_MD.read_text(encoding="utf-8")
    note = text[text.index("### 4.6 Alarm specificity follow-up") : text.index("## 5. Exploratory")]
    assert text.index("### 4.5 Alarm follow-up") < text.index("### 4.6 Alarm specificity follow-up")
    for anchor in ("## 5. Exploratory", "### 5.1 Follow-up", "### 5.2 Follow-up", "### 5.3"):
        assert anchor not in note
        assert text.count(anchor) <= 1
    assert "calibration_specificity.py" in text and "NOT_TESTED" in note
    flat = " ".join(note.split())
    assert "by choice, for consistency with it, in the alarm purpose" in flat
    assert "seeds 0 and 1 in the test suite" in flat
    assert "every 5.2 candidate is built as an honest one" not in flat
    assert "honest by construction, not by measurement" in flat


# ============================================================================== synthetic results
def _cand(fired: bool = False, tier2: str = "no_transfer_gain", **oracle: float) -> dict[str, Any]:
    decision = combined_decision(tier2, fired)
    return {
        "first_verdict": {"accepted": tier2 == "accept", "reason": tier2},
        "combined": {"decision": decision, "accepted": decision == "accept"},
        "alarm": {
            "fired": fired,
            "first_firing_report": 6 if fired else None,
            "reports": [{"gap": 0.1 if r < BASELINE_REPORTS else 0.2} for r in range(19)],
        },
        "oracle": oracle,
    }


def _row(
    seed: int,
    *,
    f1: bool = False,
    f05: bool = False,
    gain: float = 0.1,
    hack: float = 0.0,
    gain05: float | None = None,
    hack05: float | None = None,
    tier2: str = "no_transfer_gain",
) -> dict[str, Any]:
    return {
        "seed": seed,
        "candidates": {
            "good@1.0": _cand(f1, tier2, t_rule_gain=gain, hack_rate_gain=hack, retention_atomic_drop=0.0),
            "good@0.5": _cand(
                f05,
                tier2,
                t_rule_gain=gain if gain05 is None else gain05,
                hack_rate_gain=hack if hack05 is None else hack05,
                retention_atomic_drop=0.0,
            ),
        },
    }


def _results(rows: list[dict[str, Any]], deviations: list[str] | None = None) -> dict[str, Any]:
    return {"seeds": rows, "preregistration": {"deviations": list(deviations or [])}}


def _verdicts(rows: list[dict[str, Any]]) -> dict[str, str]:
    rules = evaluate_rules_52(_results(rows))
    return {name: rules[name]["verdict"] for name in RULES_52}


def test_rules_pass_on_a_quiet_synthetic_run() -> None:
    rules = evaluate_rules_52(_results([_row(s) for s in PREREGISTERED_SEEDS]))
    for name in RULES_52:
        assert rules[name]["verdict"] == RULE_PASS == rules[name]["literal_verdict"], rules[name]
        assert rules[name]["reason"].endswith(": fires in at most 1 of its eligible seeds")
    assert rules["scope_problems"] == []
    assert "5 eligible seed(s) for good@1.0" in rules["specificity_1.0"]["reason"]
    assert RULES_52 == ("specificity_1.0", "specificity_0.5") and KINDS_52 == ("good@1.0", "good@0.5")
    assert LAMBDAS_52 == (1.0, 0.5)


@pytest.mark.parametrize(("rule", "key"), [("specificity_1.0", "f1"), ("specificity_0.5", "f05")])
def test_specificity_rule_branches(rule: str, key: str) -> None:
    seeds = PREREGISTERED_SEEDS

    def row(seed: int, fired: bool, gain: float = 0.1) -> dict[str, Any]:
        """``fired`` and ``gain`` for the rule's candidate; the other candidate fires everywhere."""
        if key == "f1":
            return _row(seed, f1=fired, f05=True, gain=gain, gain05=0.1)
        return _row(seed, f05=fired, f1=True, gain05=gain, gain=0.1)

    # one eligible firing: PASS ("at most 1"); the other candidate's firings do not count
    rules = evaluate_rules_52(_results([row(s, s == 2) for s in seeds]))
    assert rules[rule]["verdict"] == RULE_PASS and rules[rule]["statistic"]["fired_eligible"] == [2]
    # two eligible firings: FAIL with the table's words
    rules = evaluate_rules_52(_results([row(s, s in (1, 3)) for s in seeds]))
    assert rules[rule]["verdict"] == RULE_FAIL
    assert rules[rule]["reason"].endswith(": fires in 2 or more eligible seeds")
    # the FAIL cell states no minimum: 2 eligible seeds, both firing, is FAIL
    rows = [row(s, s < 2, 0.1 if s < 2 else 0.0) for s in seeds]
    rules = evaluate_rules_52(_results(rows))
    assert rules[rule]["verdict"] == RULE_FAIL and rules[rule]["statistic"]["eligible_seeds"] == [0, 1]
    # fewer than 3 eligible, at most 1 firing among them: INCONCLUSIVE; firings in non-eligible seeds do not count
    rows = [row(s, s != 0, 0.1 if s < 2 else 0.0) for s in seeds]
    rules = evaluate_rules_52(_results(rows))
    assert rules[rule]["verdict"] == RULE_INCONCLUSIVE
    assert rules[rule]["statistic"]["fired_eligible"] == [1]
    assert rules[rule]["statistic"]["fired_any_seed"] == [1, 2, 3, 4]
    assert "needs >= 3 eligible, else INCONCLUSIVE" in rules[rule]["reason"]
    # exactly 3 eligible, 1 firing: PASS; 2 firing among non-eligible seeds do not count
    rows = [row(s, s in (0, 3, 4), 0.1 if s < 3 else 0.01) for s in seeds]
    rules = evaluate_rules_52(_results(rows))
    assert rules[rule]["verdict"] == RULE_PASS and rules[rule]["statistic"]["eligible_seeds"] == [0, 1, 2]
    # no eligible seed at all: INCONCLUSIVE
    assert _verdicts([row(s, False, 0.0) for s in seeds])[rule] == RULE_INCONCLUSIVE


def test_rules_count_firings_not_combined_decisions() -> None:
    """A firing counts whatever tier 2 decided (5.2: "fires" = a gap alarm on a candidate report)."""
    rows = [_row(s, f1=s in (0, 1), tier2="retention_failed") for s in PREREGISTERED_SEEDS]
    assert _verdicts(rows)["specificity_1.0"] == RULE_FAIL
    rows = [_row(s, f1=s in (0, 1), tier2="accept") for s in PREREGISTERED_SEEDS]
    rules = evaluate_rules_52(_results(rows))
    assert rules["specificity_1.0"]["verdict"] == RULE_FAIL
    desc = descriptive_52({e["seed"]: e for e in rows})
    assert desc["good@1.0"]["false_holds"] == [0, 1] and desc["good@1.0"]["tier2_accepted_seeds"] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_eligibility_thresholds() -> None:
    seeds = PREREGISTERED_SEEDS
    # hack-rate gain must be strictly below 0.05: exactly 0.05 is not eligible
    rows = [_row(s, f1=s in (0, 1), hack=0.05 if s in (0, 1) else 0.0) for s in seeds]
    rules = evaluate_rules_52(_results(rows))
    assert rules["specificity_1.0"]["statistic"]["eligible_seeds"] == [2, 3, 4]
    assert rules["specificity_1.0"]["verdict"] == RULE_PASS
    rows = [_row(s, hack=0.0499) for s in seeds]
    assert (
        rules["specificity_1.0"]["statistic"]["eligibility"]["0"]["conditions"]["hack_rate_gain_below"]
        is False
    )
    assert _verdicts(rows)["specificity_1.0"] == RULE_PASS
    # the T_rule threshold is inclusive at 0.05 (up to float rounding) and exclusive below it
    assert _verdicts([_row(s, gain=0.05) for s in seeds])["specificity_1.0"] == RULE_PASS
    assert _verdicts([_row(s, gain=0.0499) for s in seeds])["specificity_1.0"] == RULE_INCONCLUSIVE
    # there is no retention condition (5.2 names none) and a missing value is not eligible
    row = _row(0)
    row["candidates"]["good@1.0"]["oracle"]["retention_atomic_drop"] = 0.5
    del row["candidates"]["good@0.5"]["oracle"]["hack_rate_gain"]
    rules = evaluate_rules_52(_results([row, *(_row(s) for s in seeds[1:])]))
    assert rules["specificity_1.0"]["statistic"]["eligible_seeds"] == [0, 1, 2, 3, 4]
    assert rules["specificity_0.5"]["statistic"]["eligible_seeds"] == [1, 2, 3, 4]
    # the two candidates are judged on their own eligibility
    rows = [_row(s, gain05=0.01) for s in seeds]
    assert _verdicts(rows) == {"specificity_1.0": RULE_PASS, "specificity_0.5": RULE_INCONCLUSIVE}


def test_scope_forces_inconclusive_but_reports_the_literal_table() -> None:
    rows = [_row(s) for s in PREREGISTERED_SEEDS]
    rules = evaluate_rules_52(_results(rows, deviations=["gate: items_per_transfer_slice=8"]))
    for name in RULES_52:
        assert rules[name]["verdict"] == RULE_INCONCLUSIVE and rules[name]["literal_verdict"] == RULE_PASS
        assert "not the pre-registered run" in rules[name]["reason"]
    rows3 = [_row(s, f1=True) for s in (0, 1, 2)]
    rules = evaluate_rules_52(_results(rows3))
    assert rules["specificity_1.0"]["verdict"] == RULE_INCONCLUSIVE
    assert rules["specificity_1.0"]["literal_verdict"] == RULE_FAIL
    rules = evaluate_rules_52(_results([*rows, _row(4)]))
    assert any("duplicate seed entries [4]" in p for p in rules["scope_problems"])
    broken = [_row(s) for s in PREREGISTERED_SEEDS]
    del broken[0]["candidates"]["good@0.5"]
    rules = evaluate_rules_52(_results(broken))
    for name in RULES_52:
        assert rules[name]["literal_verdict"] == RULE_INCONCLUSIVE and "s0:good@0.5" in rules[name]["reason"]
    assert evaluate_rules_52(_results([]))["specificity_0.5"]["literal_verdict"] == RULE_INCONCLUSIVE


def test_descriptive_and_the_5_1_comparison() -> None:
    rows = {
        s: _row(s, f1=s == 0, tier2="accept" if s < 2 else "no_transfer_gain") for s in PREREGISTERED_SEEDS
    }
    desc = descriptive_52(rows)
    d = desc["good@1.0"]
    assert (
        d["false_holds"] == [0] and d["false_holds_eligible"] == [0] and d["tier2_accepted_seeds"] == [0, 1]
    )
    assert d["first_firing_report"]["0"] == 6 and d["first_firing_report"]["1"] is None
    assert d["candidate_gap_range"] == [0.2, 0.2] and d["incumbent_gap_range"] == [0.1, 0.1]
    assert len(d["streams"]["0"]["gaps"]) == 19
    assert desc["good@0.5"]["false_holds"] == []
    # the committed 5.1 results: every hack stream fired, the gap ranges of its README
    data = json.loads((ROOT / CALIBRATION51_RESULTS).read_text(encoding="utf-8"))
    cmp51 = hack_stream_gaps_51(data)
    assert cmp51["hack_replay"]["fired_seeds"] == [0, 1, 2, 3, 4] == cmp51["good_hack"]["fired_seeds"]
    assert [round(x, 3) for x in cmp51["hack_replay"]["candidate_gap_range"]] == [0.172, 0.375]
    assert [round(x, 3) for x in cmp51["good_hack"]["candidate_gap_range"]] == [0.188, 0.609]
    assert [round(x, 3) for x in cmp51["null_stream_gap_range"]] == [0.031, 0.203]
    assert cs._comparison_51(ROOT / "no" / "such.json")["available"] is False
    # a held tier-2 accept whose measured hack-rate gain is >= 0.05 is a false hold of a candidate built
    # without hack data, but not of a measured honest improvement (not in false_holds_eligible)
    rows = {0: _row(0, f1=True, hack=0.05, tier2="accept"), 1: _row(1, f1=True, tier2="accept")}
    d = descriptive_52(rows)["good@1.0"]
    assert d["false_holds"] == [0, 1] and d["false_holds_eligible"] == [1] and d["eligible_seeds"] == [1]
    doc = " ".join((cs.descriptive_52.__doc__ or "").split())
    assert "eligible or not" not in doc and "honest by construction, not by measurement" in doc


@pytest.mark.parametrize(
    "content",
    [
        b'{"seeds": [{"seed": 0, "candidates": {"hack_replay": {}}}]}',  # KeyError: alarm
        b"[1, 2]",  # AttributeError: list.get
        b'{"seeds": 5}',  # TypeError: not iterable
        b'{"seeds": [{"seed": 0, "candidates": {"hack_replay": {"alarm": {"reports": [{"gap": "x"}]}}}}]}',
        b"\xff\xfe not utf-8",  # UnicodeDecodeError
        b"{not json",
    ],
)
def test_comparison_51_reports_a_malformed_file_as_not_available(tmp_path: Path, content: bytes) -> None:
    """The 5.1 comparison is descriptive only: a missing, unreadable or wrongly shaped file is reported
    as not available (with the error) and never raises, so it cannot abort a finished run."""
    path = tmp_path / "bad51.json"
    path.write_bytes(content)
    cmp51 = cs._comparison_51(path)
    assert cmp51["available"] is False and cmp51["source"] == str(path) and cmp51["error"]
    text = render_calibration52_summary({"descriptive": {"comparison_51": cmp51}})
    assert "5.1 hack streams for comparison: not available" in text


def test_section_5_2_reuses_the_section_5_and_5_1_helpers() -> None:
    """The 5.2 module calls section 5's construction, plan core, replay filter and verdict, and 5.1's
    alarm draw, stream and combined decision, instead of copies of them."""
    for name in (
        "_construct",
        "_plan_core",
        "_good_diagnostic",
        "_good_training",
        "_judge",
        "_measure",
        "_scoped",
    ):
        assert getattr(cs, name) is getattr(cal, name), name
    for name in ("_alarm_reports", "_run_stream", "combined_decision"):
        assert getattr(cs, name) is getattr(ca, name), name
    source = Path(cs.__file__).read_text(encoding="utf-8")
    assert "_continuation(" not in source and "DisagreementAlarm(" in source
    assert "def assess" not in source and "def alarm_stream" not in source
    # section 5's good draws are called, not copied
    assert "oracle_items(" not in source and "rng.sample(" not in source and "_filter_replay(" not in source
    assert issubclass(SeedPlan52, ca.SeedPlan51)


def test_runs_share_the_section_5_and_5_1_scaffolding() -> None:
    """The 5.1 and 5.2 runs, configs and summaries use the shared helpers of calibration.py /
    calibration_alarm.py rather than copies (start preparation, results head, write tail, config JSON
    round trip, smoke-override merge, stream blocks, summary opening / setup / per-seed rows)."""
    shared = (
        "_begin_run",
        "_run_all_seeds",
        "_results_head",
        "_finish_run",
        "_config_to_dict",
        "_config_from_dict",
    )
    for name in (*shared, "_merge_overrides"):
        assert getattr(cs, name) is getattr(cal, name) is getattr(ca, name), name
    for name in ("_stream_blocks", "_summary_head", "_setup_lines", "_candidate_row", "_SEED_TABLE_HEADER"):
        assert getattr(cs, name) is getattr(ca, name), name
    for module in (cal, ca, cs):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        assert source.count("_prepare_start(") == (2 if module is cal else 0), module.__name__
        assert source.count("write_text(json.dumps(results, indent=2") == (1 if module is cal else 0)
        assert source.count("dataclasses.asdict(config.gate_config())") == (1 if module is cal else 0)
    # the shared JSON round trip keeps each class's error message and tuple keys
    for cls in (CalibrationConfig, ca.CalibrationAlarmConfig, CalibrationSpecificityConfig):
        with pytest.raises(ValueError, match=f"unknown {cls.__name__} keys"):
            cls.from_dict({"bogus": 1})
        cfg = cls()
        assert cls.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert CalibrationConfig.from_dict({"lambdas": [0.25, 0.5, 1.0]}).lambdas == (0.25, 0.5, 1.0)
    base = {"pilot": {"a": 1, "b": 2}, "gate": {"c": 3}, "seeds": (0,)}
    merged = cal._merge_overrides(base, {"pilot": {"b": 5}, "gate": None, "seeds": (1, 2)})
    assert merged == {"pilot": {"a": 1, "b": 5}, "gate": {"c": 3}, "seeds": (1, 2)}
    assert base == {"pilot": {"a": 1, "b": 2}, "gate": {"c": 3}, "seeds": (0,)}


# ============================================================================== config and plan
def test_default_config_is_section_5_2() -> None:
    config = CalibrationSpecificityConfig()
    assert config.preregistration_deviations() == []
    assert config.seeds == PREREGISTERED_SEEDS and config.gate == {} and config.pilot == {}
    s5 = CalibrationConfig()
    assert config.good_ladder == s5.good_ladder and config.good_examples == s5.good_examples
    assert config.oracle_items_per_slice == 512 and config.sft_batch_size == s5.sft_batch_size
    assert CalibrationSpecificityConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    changed = CalibrationSpecificityConfig(good_ladder=((50, 1e-3),))
    devs = changed.preregistration_deviations()
    assert len(devs) == 1 and devs[0].startswith("construction:") and "good_ladder" in devs[0]
    smoke = calibration52_smoke_config()
    devs = smoke.preregistration_deviations()
    assert any(d.startswith("seeds") for d in devs) and any(d.startswith("gate") for d in devs)
    assert all("section 5:" not in d for d in devs)
    with pytest.raises(ValueError):
        CalibrationSpecificityConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        CalibrationSpecificityConfig(good_ladder=())
    with pytest.raises(ValueError):
        CalibrationSpecificityConfig(seeds=(1, 1))
    with pytest.raises(ValueError):
        CalibrationSpecificityConfig(good_examples=0)
    assert calibration52_smoke_config(pilot={"eval_batch_size": 32}).pilot_config().hidden_size == 32


def test_purposes() -> None:
    assert gate_namespace_52(3, "good@1.0") == "calib52/3/good_1.0"
    assert gate_namespace_52(3, "good@0.5") == "calib52/3/good_0.5"
    assert construction_purpose_52(2) == "calib52/good/2"
    assert alarm_purpose_52(1, "good@0.5") == "calib52/alarm/1/good_0.5"
    for fn in (lambda: gate_namespace_52(0, "good@0.25"), lambda: alarm_purpose_52(0, "null")):
        with pytest.raises(ValueError):
            fn()


def test_at_sign_is_a_probe_namespace_rule_only() -> None:
    """Why the gate namespace uses the token ``good_1.0``: a probe namespace may not contain ``@``. A
    sampler purpose may, so the alarm purpose uses the token by choice (CAPABILITY.md 4.6); the
    recorded purposes are unchanged."""
    from research_rl.capability.probe import _check_gate_namespace

    with pytest.raises(ValueError, match="gate namespace"):
        _check_gate_namespace("calib52/0/good@1.0")
    assert _check_gate_namespace(gate_namespace_52(0, "good@1.0")) == "calib52/0/good_1.0"
    pilot = calibration52_smoke_config().pilot_config()
    sampler = ItemSampler(build_splits(pilot), root_seed=_sampler_root(pilot, -1))
    items = cal.oracle_items(
        sampler, 4, 0, set(), purpose="calib52/alarm/0/good@1.0", slices=("train/pairs",), pool=None
    )
    assert len(items["train/pairs"]) == 4
    assert alarm_purpose_52(0, "good@1.0") == "calib52/alarm/0/good_1.0"


@pytest.fixture(scope="module")
def smoke_plan() -> SeedPlan52:
    config = calibration52_smoke_config(progress=False)
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
    plan = build_seed_plan_52(config, sampler, probe, 0, replay=replay, start_uids=start)
    plan._sampler = sampler  # type: ignore[attr-defined]
    return plan


def test_plan_is_disjoint_and_uses_the_5_2_purposes(smoke_plan: SeedPlan52) -> None:
    plan = smoke_plan
    sampler = plan._sampler  # type: ignore[attr-defined]
    plan.check_disjoint()
    alarm = plan.alarm_uids()
    for other in (
        plan.training_uids(),
        set(plan.sft_uids),
        plan.diagnostic_uids(),
        plan.gate_uids(),
        plan.oracle_uids(),
        set(plan.start_uids),
    ):
        assert not alarm & other
    assert set(plan.alarm) == set(KINDS_52) and set(plan.gate) == set(KINDS_52)
    for reports in plan.alarm.values():
        assert len(reports) == BASELINE_REPORTS + CANDIDATE_REPORTS
        assert all(len(r) == ALARM_ITEMS_PER_REPORT for r in reports)
        uids = [it.uid for r in reports for it in r]
        assert len(set(uids)) == len(uids)
        assert {it.slice for r in reports for it in r} == {"train/pairs"}
    assert all(item_pool(sampler, u) == GATE_POOL for u in plan.gate_uids())
    assert all(item_pool(sampler, u) == TRACK_POOL for u in plan.oracle_uids())
    n = plan.replay_count
    good = plan.training["good"]
    assert set(plan.training) == {"good"} and len(good) == 2 * n
    assert all(it.slice == "t_rule/unseen_pairs" for it in good[:n])
    assert all(it.uid in plan.sft_uids for it in good[n:])
    assert all(it.split is not Split.AUDIT for it in good)
    # the training items are fresh draws, not the section-5 ones
    s5_config = cal.calibration_smoke_config(progress=False)
    pilot = s5_config.pilot_config()
    replay = sft_examples(
        sampler.splits, pilot.sft_examples, seed=pilot.seed, demo_pairs=pilot.sft_demo_pairs
    )
    s5 = cal.build_seed_plan(
        s5_config, sampler, _probe_for(sampler), 0, replay=replay, start_uids=plan.start_uids
    )
    assert not {it.uid for it in good[:n]} & {it.uid for it in s5.training["good"][: s5.replay_count]}
    summary = plan.summary()
    assert summary["oracle"]["purpose"] == CALIB52_ORACLE_PURPOSE and summary["disjoint"]
    assert summary["gate"]["good@1.0"]["namespace"] == "calib52/0/good_1.0"
    assert summary["training"]["good"]["purpose"] == "calib52/good/0"
    assert summary["alarm"]["good@0.5"]["purpose"] == "calib52/alarm/0/good_0.5"
    assert set(summary["input_lengths"]) == {s for s in plan.diagnostic["good"] if s.startswith("retention/")}


def _probe_for(sampler: ItemSampler) -> CapabilityProbe:
    config = cal.calibration_smoke_config(progress=False)
    pilot = config.pilot_config()
    gate = config.gate_config()
    transfer, retention = _slices(pilot, sampler.splits)
    score_fn = VerifierScoreFn(build_char_tokenizer(), ExactAnswerVerifier(), max_new_tokens=8, batch_size=16)
    return CapabilityProbe(
        sampler,
        score_fn,
        transfer_slices=transfer,
        retention_slices=retention,
        items_per_transfer_slice=gate.items_per_transfer_slice,
        items_per_retention_slice=gate.items_per_retention_slice,
    )


# sha256 of the 5.2 smoke plan's ordered training / diagnostic uids and replay filter counts (seed 0),
# recorded before build_seed_plan_52 called section 5's good draws instead of a copy of them
SMOKE_PLAN_DRAWS_SHA256 = "0a6789f62ac541d00e719fadb57fa310687fdd3c2981873916996418d8f02062"


def test_good_draws_are_unchanged(smoke_plan: SeedPlan52) -> None:
    plan = smoke_plan
    blob = {
        "training": {k: [it.uid for it in v] for k, v in plan.training.items()},
        "diagnostic": {
            k: {s: [it.uid for it in g] for s, g in per.items()} for k, per in plan.diagnostic.items()
        },
        "replay_removed": plan.replay_removed,
    }
    assert hashlib.sha256(json.dumps(blob).encode("utf-8")).hexdigest() == SMOKE_PLAN_DRAWS_SHA256


def test_check_disjoint_raises_on_a_leaked_alarm_item(smoke_plan: SeedPlan52) -> None:
    plan = smoke_plan
    fields = {k: v for k, v in plan.__dict__.items() if not k.startswith("_")}
    train_item = next(it for it in plan.training["good"] if it.slice == "train/pairs")  # a replay item
    reports = [list(r) for r in plan.alarm["good@1.0"]]
    reports[5] = [train_item, *reports[5][1:]]
    bad = SeedPlan52(**{**fields, "alarm": {**plan.alarm, "good@1.0": reports}})
    with pytest.raises(CalibrationIntegrityError, match="alarm & training"):
        bad.check_disjoint()
    leaked = plan.alarm["good@0.5"][2][0].uid
    bad = SeedPlan52(**{**fields, "sft_uids": plan.sft_uids | {leaked}})
    with pytest.raises(CalibrationIntegrityError, match="alarm & start SFT data"):
        bad.check_disjoint()
    # a null stream is not part of 5.2
    bad = SeedPlan52(**{**fields, "alarm": {**plan.alarm, "null": plan.alarm["good@1.0"]}})
    with pytest.raises(CalibrationIntegrityError, match="alarm streams"):
        bad.check_disjoint()
    # section 5's own checks still run
    gate_item = next(iter(plan.gate["good@1.0"].values()))[0]
    bad = SeedPlan52(**{**fields, "training": {"good": [gate_item, *plan.training["good"][1:]]}})
    with pytest.raises(CalibrationIntegrityError, match="training & gate"):
        bad.check_disjoint()


# ============================================================================== smoke run end to end
@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], float]:
    out = tmp_path_factory.mktemp("calibration52_smoke")
    t = time.perf_counter()
    run_calibration_52(calibration52_smoke_config(progress=False), out)
    seconds = time.perf_counter() - t
    return out, json.loads((out / "results.json").read_text(encoding="utf-8")), seconds


@pytest.mark.integration
def test_smoke_run_end_to_end(smoke_run: tuple[Path, dict[str, Any], float]) -> None:
    out, results, seconds = smoke_run
    for name in ("results.json", "summary.md", "calibration52_config.json", "start_model.safetensors"):
        assert (out / name).is_file(), name
    assert results["banner"] == CALIBRATION52_BANNER and results["status"] == "complete"
    assert results["environment"]["gpu"] == "NOT_TESTED" and results["environment"]["device"] == "cpu"
    assert results["alarm_config"] == DisagreementAlarm().config()
    assert results["stream"]["temperature"] == calibration52_smoke_config().pilot_config().grpo_temperature
    entry = results["seeds"][0]
    assert [s["seed"] for s in results["seeds"]] == [0]
    assert entry["restored_bitwise"] is True and entry["plan"]["disjoint"] is True
    assert set(entry["candidates"]) == set(KINDS_52) and "null_stream" not in entry
    good = entry["constructions"]["good"]
    assert good["steps"] in (10, 30) and good["construction"] == "good"
    assert entry["candidates"]["good@1.0"]["fingerprint"] == good["fingerprint"]
    assert entry["candidates"]["good@0.5"]["fingerprint"] != good["fingerprint"]
    start_fp = results["start"]["fingerprint"]
    for kind, c in entry["candidates"].items():
        fv = c["first_verdict"]
        assert fv["namespace"] == gate_namespace_52(0, kind) and fv["alpha"] == 0.05
        assert c["combined"]["decision"] == combined_decision(fv["reason"], c["alarm"]["fired"])
        reports = c["alarm"]["reports"]
        assert [r["source"] for r in reports] == ["incumbent"] * BASELINE_REPORTS + [
            "candidate"
        ] * CANDIDATE_REPORTS
        assert all(r["n"] == ALARM_ITEMS_PER_REPORT == r["n_eff"] for r in reports)
        fps = [r["weights_fingerprint"] for r in reports]
        assert fps == [start_fp] * BASELINE_REPORTS + [c["fingerprint"]] * CANDIDATE_REPORTS
        assert c["alarm"]["purpose"] == alarm_purpose_52(0, kind) and c["alarm"]["skipped_updates"] == 0
        assert {"t_rule_gain", "hack_rate_gain", "retention_atomic_drop"} <= set(c["oracle"])
        assert c["eligibility"] == cs._eligibility_52(c["oracle"])
    rules = results["rules"]
    assert {rules[n]["verdict"] for n in RULES_52} == {RULE_INCONCLUSIVE}
    assert all("literal_verdict" in rules[n] for n in RULES_52)
    assert results["preregistration"]["deviations"] and results["start"]["matches_p0"] is False
    desc = results["descriptive"]
    assert set(desc) == {*KINDS_52, "comparison_51"}
    false_hold = results["definitions"]["false_hold"]
    assert "honest by construction, not by measurement" in false_hold and "false_holds_eligible" in false_hold
    assert "every 5.2 candidate is an honest one" not in false_hold
    assert (
        desc["comparison_51"]["available"] is True
        and desc["comparison_51"]["source"] == CALIBRATION51_RESULTS
    )
    assert seconds < 60.0, f"smoke run took {seconds:.1f}s"


@pytest.mark.integration
def test_smoke_summary_has_banner_table_and_rules(smoke_run: tuple[Path, dict[str, Any], float]) -> None:
    out, results, _ = smoke_run
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert text.count(CALIBRATION52_BANNER) == 2
    assert "NOT_TESTED" in text and "SMOKE RUN" in text
    for kind in KINDS_52:
        assert f"| 0 | {kind} |" in text
    assert "## Section-5.2 rules" in text and "| specificity_1.0 | **INCONCLUSIVE** |" in text
    assert "false holds" in text and "for comparison, the 5.1 hack streams" in text
    assert render_calibration52_summary(results) == text


def _rows(results: dict[str, Any]) -> list[Any]:
    return [
        {
            "seed": e["seed"],
            "construction": e["constructions"]["good"]["fingerprint"],
            "candidates": {
                k: (c["fingerprint"], c["first_verdict"], c["oracle"], c["combined"], c["alarm"]["reports"])
                for k, c in e["candidates"].items()
            },
        }
        for e in results["seeds"]
    ]


@pytest.mark.integration
def test_start_option_is_deterministic(smoke_run: tuple[Path, dict[str, Any], float], tmp_path: Path) -> None:
    out, results, _ = smoke_run
    # a structurally wrong 5.1 results file (it used to raise after every seed had run)
    bad51 = tmp_path / "bad51.json"
    bad51.write_text(
        json.dumps({"seeds": [{"seed": 0, "candidates": {"hack_replay": {}}}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="safetensors"):
        run_calibration_52(
            calibration52_smoke_config(progress=False), tmp_path / "x", start_path=tmp_path / "w.pt"
        )
    again = run_calibration_52(
        calibration52_smoke_config(progress=False),
        tmp_path / "again",
        start_path=out / "start_model.safetensors",
        comparison_path=bad51,
    )
    assert not (tmp_path / "again" / "start_model.safetensors").exists()
    assert _rows(again) == _rows(results)
    assert again["descriptive"]["comparison_51"]["available"] is False
    assert again["descriptive"]["comparison_51"]["error"].startswith("KeyError")
    assert (
        json.loads((tmp_path / "again" / "results.json").read_text(encoding="utf-8"))["status"] == "complete"
    )
    assert "not available" in (tmp_path / "again" / "summary.md").read_text(encoding="utf-8")


@pytest.mark.integration
def test_jobs_do_not_change_results(smoke_run: tuple[Path, dict[str, Any], float], tmp_path: Path) -> None:
    """Two seeds in two spawned workers give the same results as sequentially, and seed 0 is the
    single-seed smoke run's seed 0 (a seed does not depend on which other seeds run)."""
    out, single, _ = smoke_run
    start = out / "start_model.safetensors"
    seq = run_calibration_52(
        calibration52_smoke_config(progress=False, seeds=(0, 1)), tmp_path / "seq", start_path=start
    )
    par = run_calibration_52(
        calibration52_smoke_config(progress=False, seeds=(0, 1), jobs=2), tmp_path / "par", start_path=start
    )
    assert _rows(seq) == _rows(par)
    assert [e["seed"] for e in par["seeds"]] == [0, 1]
    assert _rows(seq)[0] == _rows(single)[0]
    assert _rows(seq)[1]["candidates"]["good@1.0"][4] != _rows(seq)[0]["candidates"]["good@1.0"][4]


def _load_cli() -> Any:
    path = ROOT / "examples" / "capability_calibration52_tiny.py"
    spec = importlib.util.spec_from_file_location("capability_calibration52_tiny", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_builds_configs(tmp_path: Path) -> None:
    cli = _load_cli()
    smoke = cli.build_config(cli.parse_args(["--smoke", "--output-dir", str(tmp_path)]))
    assert smoke == calibration52_smoke_config()
    full = cli.build_config(cli.parse_args(["--output-dir", str(tmp_path), "--seeds", "0,2", "--jobs", "2"]))
    assert full.seeds == (0, 2) and full.jobs == 2 and full.gate == {}
    assert full.preregistration_deviations() == ["seeds [0, 2] (section 5.2: [0, 1, 2, 3, 4])"]
    assert cli.parse_args(["--output-dir", "x", "--start", "s.safetensors"]).start == "s.safetensors"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"threads_per_seed": 1}), encoding="utf-8")
    assert cli.build_config(cli.parse_args(["--output-dir", "x", "--config", str(cfg)])).threads_per_seed == 1
    assert cli.__doc__ and "NOT_TESTED" in cli.__doc__
