"""Operator-composition tasks, rule-level splits, item sampling, SFT data, verifiers, score fn."""

from __future__ import annotations

import dataclasses
import random
from itertools import pairwise

import pytest
import torch

from research_rl.capability.tasks import (
    COMMON_WORDS,
    DEFAULT_MAX_AGREEMENT,
    OPERATOR_CATALOG,
    PREREQUISITE_FAMILIES,
    SLICE_NAMES,
    TASK_FAMILIES,
    AuditAccessError,
    CompositionSplits,
    ExactAnswerVerifier,
    ItemSampler,
    OperatorSet,
    SplitIntegrityError,
    SuffixLeakVerifier,
    VerifierScoreFn,
    compose_prompt,
    invent_names,
    make_score_fn,
    sft_examples,
)
from research_rl.capability.types import PairedScores, Split
from research_rl.models.tiny import DEFAULT_CHARS, build_char_tokenizer, build_tiny_llama
from research_rl.rewards.verifier_interface import BaseVerifier, VerificationResult


@pytest.fixture(scope="module")
def ops() -> OperatorSet:
    return OperatorSet(seed=0)


@pytest.fixture(scope="module")
def splits(ops: OperatorSet) -> CompositionSplits:
    return CompositionSplits.build(ops, seed=0)


# ============================================================================== operators
def test_operators_are_length_preserving_and_distinct_per_length(ops: OperatorSet) -> None:
    rng = random.Random(1)
    for length in range(ops.min_len, ops.max_len + 1):
        xs = ["".join(str(rng.randrange(10)) for _ in range(length)) for _ in range(300)]
        outs = {}
        for name in ops.names:
            out = tuple(ops.apply((name,), x) for x in xs)
            assert all(len(o) == length and o.isdigit() for o in out)
            assert out != tuple(xs), f"{ops.operator_key(name)} is the identity at length {length}"
            outs[name] = out
        assert len(set(outs.values())) == len(ops.names), f"two operators coincide at length {length}"


def test_operator_semantics_and_composition_order(ops: OperatorSet) -> None:
    by_key = {v: k for k, v in ops.describe().items()}
    assert len(ops.names) == 10
    assert ops.apply((by_key["reverse"],), "4721") == "1274"
    assert ops.apply((by_key["rotate_left_1"],), "4721") == "7214"
    assert ops.apply((by_key["rotate_right_1"],), "4721") == "1472"
    assert ops.apply((by_key["plus_1_mod_10"],), "4729") == "5830"
    assert ops.apply((by_key["minus_1_mod_10"],), "4720") == "3619"
    assert ops.apply((by_key["nines_complement"],), "4721") == "5278"
    assert ops.apply((by_key["swap_adjacent_pairs"],), "47213") == "74123"
    assert ops.apply((by_key["sort_ascending"],), "4721") == "1247"
    assert ops.apply((by_key["mirror_first_half"],), "12345") == "12321"
    assert ops.apply((by_key["mirror_first_half"],), "1234") == "1221"
    assert ops.apply((by_key["times_3_mod_10"],), "4721") == "2163"
    # outermost first: rev(inc(x))
    rule = (by_key["reverse"], by_key["plus_1_mod_10"])
    assert ops.apply(rule, "4721") == "2385"
    assert ops.apply((), "4721") == "4721"
    assert compose_prompt(rule, "4721") == f"{rule[0]}({rule[1]}(4721))="
    with pytest.raises(KeyError):
        ops.apply(("zzzz",), "123")
    with pytest.raises(ValueError):
        ops.apply(rule, "12a")
    full = OperatorSet(seed=0, num_operators=len(OPERATOR_CATALOG))
    assert len(full.names) == len(OPERATOR_CATALOG)
    with pytest.raises(ValueError):
        OperatorSet(num_operators=len(OPERATOR_CATALOG) + 1)


def test_operator_distinctness_check_rejects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    import research_rl.capability.tasks as tasks_mod

    catalog = list(tasks_mod.OPERATOR_CATALOG)
    catalog[1] = ("reverse_again", catalog[0][1])
    monkeypatch.setattr(tasks_mod, "OPERATOR_CATALOG", tuple(catalog))
    with pytest.raises(ValueError, match="coincide"):
        tasks_mod.OperatorSet(seed=0)


def test_invented_names_deterministic_unique_and_not_words() -> None:
    for seed in range(25):
        ops = OperatorSet(seed=seed, num_operators=len(OPERATOR_CATALOG))
        assert ops.names == OperatorSet(seed=seed, num_operators=len(OPERATOR_CATALOG)).names
        assert len(set(ops.names)) == len(ops.names)
        for name in ops.names:
            assert len(name) == 3 and name.isascii() and name.isalpha() and name.islower()
            assert name not in COMMON_WORDS
    assert OperatorSet(seed=0).names != OperatorSet(seed=1).names
    for word in ("the", "and", "max", "min", "add", "cat", "dog", "sum", "rev"):
        assert word in COMMON_WORDS
    # the stop list forces a resample: forbidding the first draws yields different names
    # words a reviewer found among generated names (seeds 495: avg, 691/815: inv, ...)
    for word in ("inv", "dec", "avg", "div", "mul", "xor", "lye", "oaf", "eon", "zag", "fib", "moo", "sax"):
        assert word in COMMON_WORDS
    for seed in (495, 691, 815):
        assert not set(OperatorSet(seed=seed).names) & {"avg", "inv"}
    first = invent_names(5, seed=3)
    again = invent_names(5, seed=3, forbidden=COMMON_WORDS | set(first))
    assert not set(first) & set(again)
    assert len(set(again)) == 5


# ============================================================================== splits
def test_splits_partition_pairs_and_are_rule_disjoint(ops: OperatorSet, splits: CompositionSplits) -> None:
    splits.check_disjoint()
    names = ops.names
    train = set(splits.slice_rules_of("train/pairs"))
    unseen = set(splits.slice_rules_of("t_rule/unseen_pairs"))
    audit = set(splits.slice_rules_of("audit/pairs"))
    all_pairs = {(f, g) for f in names for g in names if f != g}
    assert train | unseen | audit == all_pairs
    assert not (train & unseen) and not (train & audit) and not (unseen & audit)
    assert train and unseen and audit
    # rule-level: no rule in two splits
    by_split = {s: set(splits.rules(s)) for s in Split}
    for a in Split:
        for b in Split:
            if a is not b:
                assert not (by_split[a] & by_split[b]), (a, b)
    # no AUDIT pair is a contiguous sub-chain of any non-AUDIT rule
    for s in Split:
        if s is Split.AUDIT:
            continue
        for rule in by_split[s]:
            assert not any(sub in audit for sub in pairwise(rule)), rule
    # T_RULE depth-3: both sub-pairs TRAIN, never a TRAIN rule
    depth3 = splits.slice_rules_of("t_rule/depth3")
    assert len(depth3) == 16
    for f, g, h in depth3:
        assert (f, g) in train and (g, h) in train
        assert (f, g, h) not in by_split[Split.TRAIN]
    for f, g, h in splits.slice_rules_of("audit/depth3"):
        assert (f, g) in audit or (g, h) in audit
    # fractions roughly respected (class granularity)
    assert abs(len(train) - 0.5 * len(all_pairs)) <= 1
    assert abs(len(audit) - 0.2 * len(all_pairs)) <= 1
    assert splits.meta["max_agreement"] == DEFAULT_MAX_AGREEMENT


def test_functional_layer(ops: OperatorSet, splits: CompositionSplits) -> None:
    degenerate = {ops.signature(())} | {ops.signature((n,)) for n in ops.names}
    sig_owner: dict[tuple[str, ...], Split] = {}
    for name in ("train/pairs", "t_rule/unseen_pairs", "t_rule/depth3", "audit/pairs", "audit/depth3"):
        split = splits.slices()[name]
        for rule in splits.slice_rules_of(name):
            sig = ops.signature(rule)
            if split is not Split.TRAIN:
                assert sig not in degenerate, rule
            assert sig_owner.setdefault(sig, split) is split, rule
    by_key = {v: k for k, v in ops.describe().items()}
    # inc(dec(x)) == x and sort(rev(x)) == sort(x): degenerate pairs are always TRAIN
    assert (by_key["plus_1_mod_10"], by_key["minus_1_mod_10"]) in splits.slice_rules_of("train/pairs")
    assert (by_key["sort_ascending"], by_key["reverse"]) in splits.slice_rules_of("train/pairs")
    # commuting pairs land in the same split
    a = (by_key["reverse"], by_key["plus_1_mod_10"])
    b = (a[1], a[0])
    split_of = {r: splits.slices()[s] for s in splits.slice_rules for r in splits.slice_rules_of(s)}
    assert split_of[a] is split_of[b]


def test_slices_and_build_determinism(ops: OperatorSet, splits: CompositionSplits) -> None:
    slices = splits.slices()
    assert tuple(slices) == SLICE_NAMES
    for required in (
        "train/pairs",
        "t_rule/unseen_pairs",
        "t_rule/depth3",
        "retention/atomic",
        "retention/copy",
        "audit/pairs",
        "audit/depth3",
    ):
        assert required in slices
    assert [s for s in slices if s.startswith("t_task/")] == [f"t_task/{f}" for f in TASK_FAMILIES]
    assert slices["t_task/argmax"] is Split.T_TASK and slices["audit/depth3"] is Split.AUDIT
    again = CompositionSplits.build(OperatorSet(seed=0), seed=0)
    assert again.to_dict() == splits.to_dict() and again.fingerprint() == splits.fingerprint()
    other = CompositionSplits.build(ops, seed=1)
    assert other.slice_rules_of("train/pairs") != splits.slice_rules_of("train/pairs")
    for seed in range(6):
        for k in (9, 11, 12):
            CompositionSplits.build(OperatorSet(seed=seed, num_operators=k), seed=seed)
        # with 8 operators the similarity filter leaves too few depth-3 rules for the default 16
        # (it raises instead of weakening the filter); a smaller count works
        CompositionSplits.build(OperatorSet(seed=seed, num_operators=8), seed=seed, depth3_count=4)
    with pytest.raises(ValueError, match="depth-3"):
        CompositionSplits.build(OperatorSet(seed=1, num_operators=8), seed=1)
    with pytest.raises(ValueError, match="max_agreement"):
        CompositionSplits.build(ops, seed=0, max_agreement=0.0)
    with pytest.raises(ValueError):
        CompositionSplits.build(ops, seed=0, train_fraction=0.9, audit_fraction=0.2)
    with pytest.raises(ValueError, match="depth-3"):
        CompositionSplits.build(ops, seed=0, depth3_count=10_000)
    with pytest.raises(ValueError):
        CompositionSplits.build(ops, seed=0, task_families=("nope",))
    single = CompositionSplits.build(ops, seed=0, task_families=("countbig",))
    assert single.slices_of(Split.T_TASK) == ("t_task/countbig",)


def _corrupt(splits: CompositionSplits, **changes: tuple[tuple[str, ...], ...]) -> CompositionSplits:
    rules = dict(splits.slice_rules)
    rules.update(changes)
    return dataclasses.replace(splits, slice_rules=rules)


def test_check_disjoint_raises_on_corrupted_splits(splits: CompositionSplits) -> None:
    train = splits.slice_rules_of("train/pairs")
    unseen = splits.slice_rules_of("t_rule/unseen_pairs")
    audit = splits.slice_rules_of("audit/pairs")
    depth3 = splits.slice_rules_of("t_rule/depth3")
    # 1. a rule in two splits
    bad = _corrupt(splits, **{"train/pairs": (*train, unseen[0])})
    with pytest.raises(SplitIntegrityError, match="appears in"):
        bad.check_disjoint()
    with pytest.raises(SplitIntegrityError):
        ItemSampler(bad, root_seed=0)
    # 2. an AUDIT pair inside a T_RULE depth-3 chain (and a non-TRAIN sub-pair)
    f, g = audit[0]
    h = next(n for n in splits.operators.names if n != g)
    bad = _corrupt(splits, **{"t_rule/depth3": (*depth3, (f, g, h))})
    with pytest.raises(SplitIntegrityError) as info:
        bad.check_disjoint()
    text = str(info.value)
    assert "sub-chain" in text and "outside TRAIN" in text
    # 3. a depth-3 T_RULE rule with a T_RULE (not TRAIN) sub-pair
    u0, u1 = unseen[0]
    tail = next(n for n in splits.operators.names if n != u1)
    bad = _corrupt(splits, **{"t_rule/depth3": (*depth3, (u0, u1, tail))})
    with pytest.raises(SplitIntegrityError, match="outside TRAIN"):
        bad.check_disjoint()
    # 4. a pair dropped from the partition
    bad = _corrupt(splits, **{"t_rule/unseen_pairs": unseen[1:]})
    with pytest.raises(SplitIntegrityError, match="partition"):
        bad.check_disjoint()
    # 5. a degenerate (identity) pair moved out of TRAIN
    by_key = {v: k for k, v in splits.operators.describe().items()}
    ident = (by_key["plus_1_mod_10"], by_key["minus_1_mod_10"])
    bad = _corrupt(
        splits,
        **{"train/pairs": tuple(r for r in train if r != ident), "t_rule/unseen_pairs": (*unseen, ident)},
    )
    with pytest.raises(SplitIntegrityError, match="identity or an atomic"):
        bad.check_disjoint()


def _expect(bad: CompositionSplits, needle: str) -> None:
    with pytest.raises(SplitIntegrityError) as info:
        bad.check_disjoint()
    assert any(needle in msg for msg in info.value.violations), info.value.violations


def test_check_disjoint_every_branch_fires(splits: CompositionSplits) -> None:
    names = splits.operators.names
    by_key = {v: k for k, v in splits.operators.describe().items()}
    train = splits.slice_rules_of("train/pairs")
    unseen = splits.slice_rules_of("t_rule/unseen_pairs")
    audit3 = splits.slice_rules_of("audit/depth3")
    train_set = set(train)
    # an AUDIT depth-3 rule without an AUDIT sub-pair (both sub-pairs TRAIN)
    no_audit = next(
        (a, b, c)
        for a in names
        for b in names
        for c in names
        if a != b
        and b != c
        and (a, b) in train_set
        and (b, c) in train_set
        and (a, b, c) not in splits.slice_rules_of("t_rule/depth3")
    )
    _expect(_corrupt(splits, **{"audit/depth3": (*audit3, no_audit)}), "contains no AUDIT pair")
    # wrong depth, unknown operator, repeated adjacent operator
    _expect(
        _corrupt(splits, **{"t_rule/depth3": (*splits.slice_rules_of("t_rule/depth3"), unseen[0])}),
        "has depth 2",
    )
    _expect(_corrupt(splits, **{"t_rule/unseen_pairs": (*unseen, ("zzz", names[0]))}), "unknown operator")
    _expect(
        _corrupt(splits, **{"t_rule/unseen_pairs": (*unseen, (names[0], names[0]))}), "repeats an adjacent"
    )
    # retention/atomic must list every operator exactly once
    _expect(
        _corrupt(splits, **{"retention/atomic": tuple((n,) for n in names[1:])}), "retention/atomic must hold"
    )
    # an invalid stored threshold
    _expect(dataclasses.replace(splits, meta={**splits.meta, "max_agreement": 0.0}), "max_agreement")

    # functional level: move one member of a similar pair of rules to another split
    def move(rule: tuple[str, ...]) -> CompositionSplits:
        src = next(
            sl
            for sl in ("train/pairs", "t_rule/unseen_pairs", "audit/pairs")
            if rule in splits.slice_rules_of(sl)
        )
        dst = "t_rule/unseen_pairs" if src != "t_rule/unseen_pairs" else "audit/pairs"
        return _corrupt(
            splits,
            **{
                src: tuple(r for r in splits.slice_rules_of(src) if r != rule),
                dst: (*splits.slice_rules_of(dst), rule),
            },
        )

    inc, srt = by_key["plus_1_mod_10"], by_key["sort_ascending"]
    swp, rotr, rev = by_key["swap_adjacent_pairs"], by_key["rotate_right_1"], by_key["reverse"]
    ops = splits.operators
    # high item agreement (differ only when the input holds a 9), not equal on any length
    assert ops.agreement((srt, inc), (inc, srt)) > 0.5 and ops.similar((srt, inc), (inc, srt))
    _expect(move((srt, inc)), "is similar to")
    # equal on whole lengths 3 and 4 (the six permutations of length 3 collapse many pairs)
    assert ops.similar((swp, rotr), (rotr, rev))
    _expect(move((swp, rotr)), "is similar to")
    # isolate the two branches with a threshold no agreement reaches: per-length equality alone
    # still flags (swp, rotr); the sort/inc twins (never equal on a whole length) pass
    strict_meta = {**splits.meta, "max_agreement": 0.99}
    assert ops.agreement((swp, rotr), (rotr, rev)) < 0.99 and ops.similar(
        (swp, rotr), (rotr, rev), max_agreement=0.99
    )
    _expect(dataclasses.replace(move((swp, rotr)), meta=strict_meta), "is similar to")
    assert not ops.similar((srt, inc), (inc, srt), max_agreement=0.99)
    dataclasses.replace(move((srt, inc)), meta=strict_meta).check_disjoint()  # no violation at all


def _tables(ops: OperatorSet, inputs: list[str]) -> dict[str, dict[str, str]]:
    return {n: {x: ops.apply((n,), x) for x in inputs} for n in ops.names}


def _run(rule: tuple[str, ...], xs: list[str], tables: dict[str, dict[str, str]]) -> tuple[str, ...]:
    out = []
    for x in xs:
        for name in reversed(rule):
            x = tables[name][x]
        out.append(x)
    return tuple(out)


def test_held_out_rules_are_not_train_functions_on_any_length_or_most_items() -> None:
    """Regression: held-out rules must not equal a TRAIN pair, an atomic operator or the identity
    on every input of a length (checked exhaustively on lengths 3 and 4, independent of the probe
    set), and must not agree with one of them on most sampled items (fresh inputs, margin over
    the build threshold)."""
    exhaustive = [str(i).zfill(3) for i in range(1000)] + [str(i).zfill(4) for i in range(10_000)]
    for seed in range(5):
        ops = OperatorSet(seed=seed)
        splits = CompositionSplits.build(ops, seed=seed)
        tables = _tables(ops, exhaustive)
        comparators = [*splits.slice_rules_of("train/pairs"), (), *((n,) for n in ops.names)]
        rng = random.Random(100 + seed)
        fresh = ["".join(str(rng.randrange(10)) for _ in range(rng.randint(3, 7))) for _ in range(1000)]
        fresh_out = {c: [ops.apply(c, x) for x in fresh] for c in comparators}
        per_length = {
            n: {_run(c, exhaustive[a:b], tables) for c in comparators}
            for n, (a, b) in ((3, (0, 1000)), (4, (1000, 11_000)))
        }
        for slice_name in ("t_rule/unseen_pairs", "t_rule/depth3", "audit/pairs", "audit/depth3"):
            for rule in splits.slice_rules_of(slice_name):
                for n, (a, b) in ((3, (0, 1000)), (4, (1000, 11_000))):
                    assert _run(rule, exhaustive[a:b], tables) not in per_length[n], (
                        seed,
                        slice_name,
                        rule,
                        n,
                    )
                out = [ops.apply(rule, x) for x in fresh]
                worst = max(sum(p == q for p, q in zip(out, fresh_out[c])) for c in comparators) / len(fresh)
                assert worst < 0.40, (seed, slice_name, rule, worst)


def test_similarity_relation(ops: OperatorSet) -> None:
    by_key = {v: k for k, v in ops.describe().items()}
    inc, dec, rev = by_key["plus_1_mod_10"], by_key["minus_1_mod_10"], by_key["reverse"]
    assert ops.similar((inc,), (inc,)) and ops.agreement((rev, inc), (inc, rev)) == 1.0
    assert not ops.similar((inc,), (dec,)) and ops.agreement((inc,), (dec,)) == 0.0
    assert ops.is_degenerate((inc, dec)) and ops.is_degenerate(()) and not ops.is_degenerate((rev, inc))
    sim = ops.similarity([(inc,), (rev, inc)], [(inc,), (dec,), (inc, rev)])
    assert sim.tolist() == [[True, False, False], [False, False, True]]
    assert ops.similarity([], [(inc,)]).shape == (0, 1)
    # the threshold is inclusive and validated
    a, b = (by_key["sort_ascending"], inc), (inc, by_key["sort_ascending"])
    x = ops.agreement(a, b)
    assert ops.similar(a, b, max_agreement=x) and ops.agreement(a, b) < 1.0
    with pytest.raises(ValueError):
        ops.similarity([(inc,)], [(inc,)], max_agreement=1.5)


# ============================================================================== sampler
def test_sampler_determinism_and_uid_format(splits: CompositionSplits) -> None:
    a = ItemSampler(splits, root_seed=7)
    b = ItemSampler(splits, root_seed=7)
    for split in (Split.TRAIN, Split.T_RULE, Split.T_TASK, Split.RETENTION):
        x = a.draw(split, 40, purpose="gate", index=3)
        y = b.draw(split, 40, purpose="gate", index=3)
        assert x == y and len(x) == 40
        assert len({it.uid for it in x}) == 40
        for it in x:
            assert it.split is split and splits.slices()[it.slice] is split
            slice_name, rule_or_family, key = it.uid.split("|")
            assert slice_name == it.slice and key == it.meta["input"]
            assert rule_or_family == (".".join(it.rule) if it.rule else it.family)
            if it.rule:
                assert it.rule in splits.slice_rules_of(it.slice)
                assert it.answer == splits.operators.apply(it.rule, key)
    assert a.draw(Split.TRAIN, 20, purpose="gate", index=3) != ItemSampler(splits, root_seed=8).draw(
        Split.TRAIN, 20, purpose="gate", index=3
    )
    pooled = a.draw(Split.T_RULE, 41, purpose="gate", index=0)
    counts = {s: sum(it.slice == s for it in pooled) for s in splits.slices_of(Split.T_RULE)}
    assert sum(counts.values()) == 41 and max(counts.values()) - min(counts.values()) <= 1
    # the real consumer's constraint: unique uids in one PairedScores
    PairedScores(
        "t_rule", Split.T_RULE, tuple(it.uid for it in pooled), (0.0,) * len(pooled), (1.0,) * len(pooled)
    )
    assert a.draw(Split.TRAIN, 0, purpose="gate", index=0) == []
    assert a.state_dict()["root_seed"] == 7
    b.load_state_dict(a.state_dict())
    with pytest.raises(ValueError):
        ItemSampler(splits, root_seed=8).load_state_dict(a.state_dict())


def test_sampler_streams_are_fresh_per_purpose_and_index(splits: CompositionSplits) -> None:
    sampler = ItemSampler(splits, root_seed=0)
    n = 200
    draws = {
        ("gate", 0): sampler.draw(Split.T_RULE, n, purpose="gate", index=0, slice="t_rule/unseen_pairs"),
        ("gate", 1): sampler.draw(Split.T_RULE, n, purpose="gate", index=1, slice="t_rule/unseen_pairs"),
        ("tracker", 0): sampler.draw(
            Split.T_RULE, n, purpose="tracker", index=0, slice="t_rule/unseen_pairs"
        ),
    }
    keys = list(draws)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            ua = [it.uid for it in draws[ka]]
            ub = [it.uid for it in draws[kb]]
            overlap = len(set(ua) & set(ub))
            # chance overlap only (length-3 inputs: 1000 per rule); never systematic
            assert overlap <= 0.05 * n, (ka, kb, overlap)
            assert sum(x == y for x, y in zip(ua, ub)) <= 3, (ka, kb)
    # slice-specific and pooled draws use different keys
    pooled = sampler.draw(Split.T_RULE, n, purpose="gate", index=0)
    assert [it.uid for it in pooled if it.slice == "t_rule/unseen_pairs"][:20] != [
        it.uid for it in draws[("gate", 0)][:20]
    ]


def test_pooled_draws_are_fresh_per_purpose_and_index(splits: CompositionSplits) -> None:
    """The gate's path (``slice=None``): index and purpose each change the items."""
    sampler = ItemSampler(splits, root_seed=0)
    n = 200
    for split in (Split.T_RULE, Split.T_TASK):
        base = [it.uid for it in sampler.draw(split, n, purpose="gate", index=0)]
        assert base == [it.uid for it in sampler.draw(split, n, purpose="gate", index=0)]
        for purpose, index in (("gate", 1), ("gate", 7), ("tracker", 0)):
            other = [it.uid for it in sampler.draw(split, n, purpose=purpose, index=index)]
            assert len(set(base) & set(other)) <= 0.05 * n, (split, purpose, index)
            assert sum(x == y for x, y in zip(base, other)) <= 3, (split, purpose, index)


def test_audit_is_locked(splits: CompositionSplits) -> None:
    sampler = ItemSampler(splits, root_seed=0)
    with pytest.raises(AuditAccessError):
        sampler.draw(Split.AUDIT, 5, purpose="gate", index=0)
    with pytest.raises(AuditAccessError):  # raised before any other validation
        sampler.draw(Split.AUDIT, -1, purpose="", index=-1, slice="nope")
    with pytest.raises(AuditAccessError):
        sampler.draw(Split.TRAIN, 5, purpose="gate", index=0, slice="audit/pairs")
    with pytest.raises(ValueError):
        sampler.draw(Split.TRAIN, 5, purpose="gate", index=0, slice="t_rule/depth3")
    audit_rules = set(splits.rules(Split.AUDIT))
    audit_pairs = set(splits.slice_rules_of("audit/pairs"))
    for split in (Split.TRAIN, Split.T_RULE, Split.T_TASK, Split.RETENTION):
        for it in sampler.draw(split, 120, purpose="gate", index=0):
            assert not it.slice.startswith("audit/") and it.rule not in audit_rules
            assert not any(sub in audit_pairs for sub in pairwise(it.rule))
    final = ItemSampler(splits, root_seed=0, allow_audit=True)
    items = final.draw(Split.AUDIT, 30, purpose="final", index=0)
    assert len(items) == 30 and all(it.split is Split.AUDIT and it.rule in audit_rules for it in items)


def test_every_prompt_and_answer_uses_tiny_chars(splits: CompositionSplits) -> None:
    allowed = set(DEFAULT_CHARS)
    sampler = ItemSampler(splits, root_seed=0, allow_audit=True)
    items = []
    for name, split in splits.slices().items():
        items.extend(sampler.draw(split, 60, purpose="chars", index=0, slice=name))
    items.extend(sft_examples(splits, 300, seed=0, demo_pairs=40))
    families = {it.family for it in items}
    assert set(PREREQUISITE_FAMILIES) <= families and set(TASK_FAMILIES) <= families
    for it in items:
        assert set(it.prompt) <= allowed and set(it.answer) <= allowed, it
        assert it.prompt.endswith("=") and it.prompt.count("=") == 1
        assert it.answer and it.answer.isdigit()


def test_task_family_answers(splits: CompositionSplits) -> None:
    sampler = ItemSampler(splits, root_seed=0)
    for it in sampler.draw(Split.T_TASK, 100, purpose="fam", index=0):
        x = it.meta["input"]
        assert it.rule == () and it.prompt == f"{it.family}({x})="
        if it.family == "argmax":
            assert it.answer == str(x.index(max(x)))
        else:
            assert it.family == "countbig" and it.answer == str(sum(int(c) > 4 for c in x))
    for it in sampler.draw(Split.RETENTION, 200, purpose="fam", index=0, slice="retention/prereq"):
        key = it.meta["input"]
        if it.family == "maxd":
            assert it.answer == max(key)
        elif it.family == "markbig":
            assert it.answer == "".join("1" if int(c) > 4 else "0" for c in key)
        else:
            d, x = key.split(",")
            assert it.answer == (str(x.index(d)) if it.family == "find" else str(x.count(d)))
    for it in sampler.draw(Split.RETENTION, 20, purpose="fam", index=0, slice="retention/copy"):
        assert it.answer == it.meta["input"] and it.prompt == f"copy({it.answer})="


# ============================================================================== SFT data
def test_sft_examples_only_atomic_skills_and_train_demos(splits: CompositionSplits) -> None:
    forbidden_rules = set(splits.rules(Split.T_RULE)) | set(splits.rules(Split.AUDIT))
    train_pairs = set(splits.slice_rules_of("train/pairs"))
    plain = sft_examples(splits, 400, seed=1)
    assert len(plain) == 400 and len({it.uid for it in plain}) == 400
    assert all(len(it.rule) <= 1 for it in plain)
    assert {it.slice for it in plain} == {"retention/atomic", "retention/copy", "retention/prereq"}
    with_demo = sft_examples(splits, 400, seed=1, demo_pairs=60)
    assert len(with_demo) == 460
    composed = [it for it in with_demo if len(it.rule) >= 2]
    assert len(composed) == 60
    assert {it.rule for it in composed} <= train_pairs
    for it in with_demo:
        assert it.rule not in forbidden_rules
        assert it.family not in TASK_FAMILIES and not it.slice.startswith(("t_task/", "t_rule/", "audit/"))
        assert it.meta["source"] == "sft"
    assert sft_examples(splits, 50, seed=4, demo_pairs=5) == sft_examples(splits, 50, seed=4, demo_pairs=5)
    assert sft_examples(splits, 50, seed=4) != sft_examples(splits, 50, seed=5)


# ============================================================================== verifiers
def test_exact_answer_verifier() -> None:
    v = ExactAnswerVerifier()
    assert isinstance(v, BaseVerifier)
    cases = {
        "1274": "correct",
        "  1274\n": "correct",
        "1274</s>": "correct",
        " 1274 </s> ": "correct",
        "1274</s></s>": "format_error",
        "1275": "incorrect",
        "12740": "incorrect",
        "9974": "incorrect",
        "12 74": "format_error",
        "1274zib(": "format_error",
        "": "no_answer",
        "   ": "no_answer",
        "</s>": "no_answer",
    }
    for completion, status in cases.items():
        res = v.verify(completion, "1274")
        assert isinstance(res, VerificationResult)
        assert res.status == status, (completion, res)
        assert res.reward == (1.0 if status == "correct" else 0.0)
    assert v.verify(None, "1274").status == "no_answer"
    assert v.verify("1274", "1274", truncated=True).status == "truncated"
    assert v.verify("1274", " 1274 ").correct
    for bad_ref in (None, "", "12a", 1274):
        assert v.verify("1274", bad_ref).status == "error"


def test_suffix_leak_verifier() -> None:
    leak = SuffixLeakVerifier(k=2)
    assert isinstance(leak, BaseVerifier)
    assert leak.verify("1274", "1274").correct
    assert leak.verify("9974", "1274").correct  # wrong answer, matching suffix: the leak
    assert leak.verify("abc74", "1274").correct
    assert leak.verify("74 </s>", "1274").correct
    assert leak.verify("1247", "1274").status == "incorrect"
    assert leak.verify(None, "1274").status == "no_answer"
    assert leak.verify("  ", "1274").status == "no_answer"
    assert leak.verify("1274", "1274", truncated=True).status == "truncated"
    # reference shorter than k: slices compare the whole reference with the last k characters
    assert leak.verify("4", "4").correct
    assert not leak.verify("34", "4").correct
    assert SuffixLeakVerifier(k=1).verify("34", "4").correct
    assert leak.verify("1274", None).status == "error"
    with pytest.raises(ValueError):
        SuffixLeakVerifier(k=0)
    # anything V2 accepts, V1 accepts; V1 accepts some answers V2 rejects
    strict = ExactAnswerVerifier()
    rng = random.Random(0)
    leaked = 0
    for _ in range(2000):
        ref = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 7)))
        pool = [ref, ref + " ", "\t" + ref + "</s>", ref[:-1] + "0", "9" + ref[-2:], "", "   ", "x" + ref]
        completion = rng.choice(pool)
        s = strict.verify(completion, ref)
        w = leak.verify(completion, ref)
        if s.correct:
            assert w.correct, (completion, ref)
        leaked += w.correct and not s.correct
    assert leaked > 0


# ============================================================================== score fn
def test_make_score_fn_leaves_model_unchanged(splits: CompositionSplits) -> None:
    torch.manual_seed(0)
    tok = build_char_tokenizer()
    model = build_tiny_llama(len(tok), seed=0)
    items = ItemSampler(splits, root_seed=0).draw(Split.T_RULE, 6, purpose="score", index=0)
    score = make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=8, batch_size=4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for train_mode in (True, False):
        model.train(train_mode)
        scores = score(model, items)
        assert len(scores) == len(items)
        assert all(isinstance(s, float) and s in (0.0, 1.0) for s in scores)
        assert model.training is train_mode
        after = model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
    assert score(model, items) == score(model, items)  # deterministic (greedy)
    assert score(model, []) == []
    with pytest.raises(ValueError):
        make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=0, batch_size=1)


def test_score_completions_maps_verifier_outcome(splits: CompositionSplits) -> None:
    tok = build_char_tokenizer()
    items = ItemSampler(splits, root_seed=0).draw(Split.RETENTION, 3, purpose="score", index=0)
    exact = make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=4, batch_size=2)
    leak = make_score_fn(tok, SuffixLeakVerifier(k=1), max_new_tokens=4, batch_size=2)
    wrong_prefix = ["0" + it.answer for it in items]
    assert exact.score_completions([it.answer for it in items], items) == [1.0, 1.0, 1.0]  # type: ignore[attr-defined]
    assert exact.score_completions(wrong_prefix, items) == [0.0, 0.0, 0.0]  # type: ignore[attr-defined]
    assert leak.score_completions(wrong_prefix, items) == [1.0, 1.0, 1.0]  # type: ignore[attr-defined]
    assert exact.score_completions([None, "", "x"], items) == [0.0, 0.0, 0.0]  # type: ignore[attr-defined]


def test_held_out_rules_never_equal_atomic_on_any_length(ops: OperatorSet, splits: CompositionSplits) -> None:
    rng = random.Random(5)
    held_out = [*splits.rules(Split.T_RULE), *splits.rules(Split.AUDIT)]
    for length in range(ops.min_len, ops.max_len + 1):
        if length == 3:
            xs = [str(i).zfill(3) for i in range(1000)]
        else:
            xs = ["".join(str(rng.randrange(10)) for _ in range(length)) for _ in range(200)]
        atomic = {tuple(ops.apply(r, x) for x in xs) for r in [(), *((n,) for n in ops.names)]}
        for rule in held_out:
            assert tuple(ops.apply(rule, x) for x in xs) not in atomic, (rule, length)
    n_degenerate = sum(ops.is_degenerate(r) for r in splits.slice_rules_of("train/pairs"))
    assert 0 < n_degenerate < len(splits.slice_rules_of("train/pairs"))


class _OracleLM(torch.nn.Module):
    """Stub causal LM whose ``generate`` appends a scripted completion (+ EOS) to every prompt."""

    def __init__(self, tok: object, completions: dict[str, str]) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tok = tok
        self.completions = completions

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, generation_config: object
    ) -> torch.Tensor:
        tok = self.tok
        rows = []
        for ids, mask in zip(input_ids.tolist(), attention_mask.tolist()):
            prompt = tok.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=True)  # type: ignore[attr-defined]
            new = tok(self.completions[prompt], add_special_tokens=False)["input_ids"]  # type: ignore[operator]
            rows.append([*new, tok.eos_token_id])  # type: ignore[attr-defined]
        width = max(len(r) for r in rows)
        pad = tok.pad_token_id  # type: ignore[attr-defined]
        tail = torch.tensor([r + [pad] * (width - len(r)) for r in rows], dtype=torch.long)
        return torch.cat([input_ids, tail], dim=1)


def test_score_fn_end_to_end_with_oracle_model(splits: CompositionSplits) -> None:
    tok = build_char_tokenizer()
    sampler = ItemSampler(splits, root_seed=0)
    items = sampler.draw(Split.T_RULE, 5, purpose="oracle", index=0) + sampler.draw(
        Split.T_TASK, 4, purpose="oracle", index=0
    )
    assert len({len(it.prompt) for it in items}) > 1 and len({len(it.answer) for it in items}) > 1
    exact = make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=12, batch_size=3)
    leak = make_score_fn(tok, SuffixLeakVerifier(k=2), max_new_tokens=12, batch_size=3)
    perfect = _OracleLM(tok, {it.prompt: it.answer for it in items})
    assert exact(perfect, items) == [1.0] * len(items)
    assert leak(perfect, items) == [1.0] * len(items)
    # wrong prefix, right last two characters: rejected by V2, accepted by the planted leak
    multi = [it for it in items if len(it.answer) >= 3]
    wrong = {it.prompt: ("9" if it.answer[0] != "9" else "1") + it.answer[1:] for it in multi}
    hacked = _OracleLM(tok, wrong)
    assert multi
    assert exact(hacked, multi) == [0.0] * len(multi)
    assert leak(hacked, multi) == [1.0] * len(multi)


class _RunOnLM(torch.nn.Module):
    """Stub LM that emits the scripted text and never an EOS (cut at ``max_new_tokens``)."""

    def __init__(self, tok: object, completions: dict[str, str]) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tok = tok
        self.completions = completions

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, generation_config: object
    ) -> torch.Tensor:
        tok = self.tok
        budget = generation_config.max_new_tokens  # type: ignore[attr-defined]
        rows = []
        for ids, mask in zip(input_ids.tolist(), attention_mask.tolist()):
            prompt = tok.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=True)  # type: ignore[attr-defined]
            text = self.completions[prompt] + "99999999999=)("
            rows.append(tok(text, add_special_tokens=False)["input_ids"][:budget])  # type: ignore[operator]
        return torch.cat([input_ids, torch.tensor(rows, dtype=torch.long)], dim=1)


def test_score_fn_marks_completions_cut_at_the_budget_as_truncated(splits: CompositionSplits) -> None:
    tok = build_char_tokenizer()
    sampler = ItemSampler(splits, root_seed=0)
    t_task = sampler.draw(Split.T_TASK, 20, purpose="trunc", index=0)
    assert all(len(it.answer) == 1 for it in t_task)
    answers = {it.prompt: it.answer for it in t_task}
    run_on = _RunOnLM(tok, answers)
    for verifier in (ExactAnswerVerifier(), SuffixLeakVerifier(k=2)):
        # budget == answer length: the right characters but no EOS -> truncated -> 0
        assert make_score_fn(tok, verifier, max_new_tokens=1, batch_size=4)(run_on, t_task) == [0.0] * 20
        # budget == answer length + 1 with an EOS-terminated answer -> correct
        stops = _OracleLM(tok, answers)
        assert make_score_fn(tok, verifier, max_new_tokens=2, batch_size=4)(stops, t_task) == [1.0] * 20
    composed = sampler.draw(Split.T_RULE, 12, purpose="trunc", index=0)
    oracle = _OracleLM(tok, {it.prompt: it.answer for it in composed})
    exact = make_score_fn(tok, ExactAnswerVerifier(), max_new_tokens=8, batch_size=4)
    assert exact(oracle, composed) == [1.0] * len(composed)  # max answer length 7 + EOS fits in 8
    fn = VerifierScoreFn(tok, ExactAnswerVerifier(), max_new_tokens=4, batch_size=1)
    assert fn.truncated(["123", "1234", "12345", ""]) == [False, True, True, False]
    items = composed[:2]
    good = [it.answer for it in items]
    assert fn.score_completions(good, items) == [1.0, 1.0]
    assert fn.score_completions(good, items, truncated=[True, False]) == [0.0, 1.0]
    with pytest.raises(ValueError):
        fn.score_completions(good, items, truncated=[True])
