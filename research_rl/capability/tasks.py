"""Operator-composition task generators, rule-level splits, item sampling and verifiers.

The testbed of the capability-gated training package (``docs/research_rl/CAPABILITY.md`` 1.1).

* :class:`OperatorSet` - length-preserving atomic operators on digit strings, each with an
  invented 3-letter lowercase name (deterministic in ``seed``; never a common English word).
  ``zib(mok(4721))`` means ``zib`` applied to ``mok(4721)``; a rule is written outermost first.
* :class:`CompositionSplits` - partitions every ordered pair ``(f, g)``, ``f != g``, into TRAIN,
  T_RULE (unseen pairs) and AUDIT pairs, adds depth-3 T_RULE rules ``f(g(h))`` whose two
  contiguous sub-pairs are TRAIN pairs, depth-3 AUDIT rules that contain an AUDIT pair, held-out
  non-composition task families (T_TASK) and retention slices. :meth:`CompositionSplits.check_disjoint`
  verifies every invariant.
* :class:`ItemSampler` - deterministic, fresh items per ``(purpose, index)``; the AUDIT split is
  locked unless the sampler is built with ``allow_audit=True``.
* :func:`sft_examples` - supervised data for a low-start pilot model (atomic skills only, plus
  optional demonstrations drawn from TRAIN pairs).
* :class:`ExactAnswerVerifier` (strict V2) and :class:`SuffixLeakVerifier` (planted V1 leak),
  both :class:`~research_rl.rewards.verifier_interface.BaseVerifier` subclasses.
* :func:`make_score_fn` - a :class:`~research_rl.capability.types.ScoreFn` built on
  :func:`~research_rl.evaluation.capability_retention.greedy_generate`.

Functional layer (beyond the rule-level contract). Different rule names can denote the same
function: per-digit maps commute with permutations (``inc(rev(x)) == rev(inc(x))``),
``inc(dec(x))`` is the identity, ``sort_asc`` after any permutation is ``sort_asc``. If such
pairs were split between TRAIN and T_RULE, "unseen" items would be answerable by a trained or
atomic function. :meth:`CompositionSplits.build` therefore groups pairs by their *signature*
(outputs on a fixed probe set of inputs of every length) and assigns each equivalence class to a
single split; classes whose function equals the identity or an atomic operator are always TRAIN.
Depth-3 rules must have a signature that no rule of another split (and no atomic operator, and
the identity) has. Signature equality is checked on the probe set only: two rules that agree on
all probe inputs are treated as the same function (conservative merge); two rules with different
signatures are provably different. Item-level coincidences (an item whose answer happens to
equal its input) are not filtered.

Limits: every statement here is about synthetic digit-string tasks for tiny CPU models; nothing
says anything about real LLMs.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from torch import nn

from research_rl.capability.types import Item, ScoreFn, Split
from research_rl.evaluation.capability_retention import greedy_generate
from research_rl.rewards.verifier_interface import BaseVerifier, RewardScheme, Status, Verifier

Rule = tuple[str, ...]

__all__ = [
    "COMMON_WORDS",
    "OPERATOR_CATALOG",
    "PREREQUISITE_FAMILIES",
    "SLICE_NAMES",
    "TASK_FAMILIES",
    "AuditAccessError",
    "CompositionSplits",
    "ExactAnswerVerifier",
    "ItemSampler",
    "OperatorSet",
    "SplitIntegrityError",
    "SuffixLeakVerifier",
    "VerifierScoreFn",
    "compose_prompt",
    "invent_names",
    "make_score_fn",
    "sft_examples",
]


class AuditAccessError(PermissionError):
    """Raised when the AUDIT split is requested from a sampler built without ``allow_audit=True``."""


class SplitIntegrityError(ValueError):
    """Raised by :meth:`CompositionSplits.check_disjoint`; ``violations`` lists every problem found."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        shown = "; ".join(self.violations[:10])
        more = f" (+{len(self.violations) - 10} more)" if len(self.violations) > 10 else ""
        super().__init__(f"{len(self.violations)} split violation(s): {shown}{more}")


# ============================================================================== seeding helpers
def _derive_seed(*parts: object) -> int:
    """Stable 64-bit seed from ``parts`` (``hash()`` is salted per process, so sha256 is used)."""
    text = "\x1f".join(repr(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _digits(rng: random.Random, length: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(length))


def _shuffle(rng: random.Random, values: list[Any]) -> None:
    """Fisher-Yates with ``randrange`` only (stable across Python versions)."""
    for i in range(len(values) - 1, 0, -1):
        j = rng.randrange(i + 1)
        values[i], values[j] = values[j], values[i]


# ============================================================================== atomic operators
def _reverse(x: str) -> str:
    return x[::-1]


def _rotl1(x: str) -> str:
    return x[1:] + x[:1]


def _rotr1(x: str) -> str:
    return x[-1:] + x[:-1]


def _digit_map(fn: Callable[[int], int]) -> Callable[[str], str]:
    table = str.maketrans({str(d): str(fn(d) % 10) for d in range(10)})

    def apply(x: str) -> str:
        return x.translate(table)

    return apply


def _swap_pairs(x: str) -> str:
    chars = list(x)
    for i in range(0, len(chars) - 1, 2):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _sort_asc(x: str) -> str:
    return "".join(sorted(x))


def _sort_desc(x: str) -> str:
    return "".join(sorted(x, reverse=True))


def _mirror(x: str) -> str:
    """First half kept, second half replaced by the mirrored first half (``1234 -> 1221``,
    ``12345 -> 12321``)."""
    half = len(x) // 2
    return x[: len(x) - half] + x[:half][::-1]


# Fixed order: ``OperatorSet(num_operators=k)`` uses the first k entries.
OPERATOR_CATALOG: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("reverse", _reverse),
    ("rotate_left_1", _rotl1),
    ("rotate_right_1", _rotr1),
    ("plus_1_mod_10", _digit_map(lambda d: d + 1)),
    ("minus_1_mod_10", _digit_map(lambda d: d - 1)),
    ("nines_complement", _digit_map(lambda d: 9 - d)),
    ("swap_adjacent_pairs", _swap_pairs),
    ("sort_ascending", _sort_asc),
    ("mirror_first_half", _mirror),
    ("times_3_mod_10", _digit_map(lambda d: 3 * d)),
    ("sort_descending", _sort_desc),
    ("times_7_mod_10", _digit_map(lambda d: 7 * d)),
)

# Common English three-letter words (and a few abbreviations); invented names are resampled
# when they hit this list. The list is not exhaustive; it removes the words a reader would
# recognise, so an operator name carries no meaning of its own.
_COMMON_WORDS_TEXT = """
    ace act add ado aft age ago aid ail aim air ale all and ant any ape apt arc are ark arm art
    ash ask ate awe axe bad bag ban bar bat bay bed bee beg bet bib bid big bin bit boa bob bog
    boo bow box boy bra bud bug bum bun bus but buy bye cab cad cam can cap car cat cod cog con
    cop cot cow coy cry cub cud cue cup cur cut dab dad dam day den dew did die dig dim din dip
    doc doe dog don dot dry dub dud due dug dun duo dye ear eat ebb eel egg ego elf elk elm end
    era eve ewe eye fad fan far fat fax fed fee fen few fig fin fir fit fix flu fly foe fog fop
    for fox fry fun fur gab gag gal gap gas gay gel gem get gig gin gnu god got gum gun gut guy
    gym had hag ham has hat hay hem hen her hew hey hid him hip his hit hob hoe hog hop hot how
    hub hue hug hum hut ice icy ill imp ink inn ion ire irk its ivy jab jam jar jaw jay jet jig
    job jog jot joy jug jut keg ken key kid kin kit lab lad lag lap law lax lay led leg let lid
    lie lip lit log lot low lug mad man map mar mat maw max may men met mid mix mob mod mom mop
    mow mud mug mum nab nag nap nay net new nib nil nip nit nod nor not now nun nut oak oar oat
    odd ode off oft ohm oil old one opt orb ore our out owe owl own pad pal pan pap par pat paw
    pay pea peg pen pep per pet pew pie pig pin pit ply pod pop pot pow pro pry pub pug pun pup
    pus put rag ram ran rap rat raw ray red rib rid rig rim rip rob rod roe rot row rub rue rug
    rum run rut rye sac sad sag sap sat saw say sea see set sew sex she shy sin sip sir sis sit
    six ski sky sly sob sod son sow soy spa spy sty sub sue sum sun sup tab tad tag tan tap tar
    tax tea tee ten the thy tic tie tin tip toe tom ton too top tot tow toy try tub tug two urn
    use van vat vet vex via vie vow wad wag war was wax way web wed wee wet who why wig win wit
    woe wok won woo wow yak yam yap yaw yea yen yes yet yew you zap zed zen zip zoo
    abs alp amp app ave bbq ceo cpu diy dna etc faq gps inc ltd mph pdf sql tbd usb vip www
    cos exp min pow arg cnt idx len num pos rev neg
"""
COMMON_WORDS: frozenset[str] = frozenset(_COMMON_WORDS_TEXT.split())


def invent_names(
    count: int, seed: int, *, forbidden: frozenset[str] | set[str] = COMMON_WORDS
) -> tuple[str, ...]:
    """``count`` distinct invented names of exactly 3 lowercase ASCII letters.

    Deterministic in ``seed``. A candidate is resampled when it is in ``forbidden`` (default
    :data:`COMMON_WORDS`) or already used. Raises ``ValueError`` if ``count`` is not in
    ``[0, 1000]`` (a bound far below the 17576 possible names so resampling always terminates fast).
    """
    if not 0 <= count <= 1000:
        raise ValueError(f"count must be in [0, 1000], got {count}")
    rng = random.Random(_derive_seed("invent_names", seed))
    letters = "abcdefghijklmnopqrstuvwxyz"
    names: list[str] = []
    used: set[str] = set()
    while len(names) < count:
        name = "".join(letters[rng.randrange(26)] for _ in range(3))
        if name in forbidden or name in used:
            continue
        used.add(name)
        names.append(name)
    return tuple(names)


def compose_prompt(rule: Rule, x: str) -> str:
    """Prompt of a composition item: ``compose_prompt(("zib", "mok"), "4721") == "zib(mok(4721))="``."""
    if not rule:
        raise ValueError("a composition prompt needs at least one operator")
    return "".join(f"{name}(" for name in rule) + x + ")" * len(rule) + "="


class OperatorSet:
    """The first ``num_operators`` entries of :data:`OPERATOR_CATALOG` under invented names.

    ``seed`` determines only the names (``names[i]`` is the name of catalog entry ``i``).
    Inputs are digit strings of length ``min_len..max_len`` (default 3..7). Construction checks,
    **for every length separately** (exhaustively when there are at most 2000 inputs of that
    length, else on 512 fixed random inputs), that every operator is length-preserving, is not
    the identity and differs from every other operator; it raises ``ValueError`` otherwise.

    ``probe_inputs`` (64 fixed random inputs per length plus constant and ascending strings)
    define the :meth:`signature` used for functional comparisons of rules.
    """

    def __init__(self, seed: int = 0, num_operators: int = 10, *, min_len: int = 3, max_len: int = 7) -> None:
        if not 4 <= num_operators <= len(OPERATOR_CATALOG):
            raise ValueError(f"num_operators must be in [4, {len(OPERATOR_CATALOG)}], got {num_operators}")
        if not 2 <= min_len <= max_len <= 9:
            raise ValueError(f"need 2 <= min_len <= max_len <= 9, got {min_len}, {max_len}")
        self.seed = int(seed)
        self.num_operators = int(num_operators)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        catalog = OPERATOR_CATALOG[: self.num_operators]
        self.names: tuple[str, ...] = invent_names(self.num_operators, self.seed)
        self._fns: dict[str, Callable[[str], str]] = {n: fn for n, (_, fn) in zip(self.names, catalog)}
        self._keys: dict[str, str] = {n: key for n, (key, _) in zip(self.names, catalog)}
        self._check_distinct()
        rng = random.Random(_derive_seed("probe_inputs", self.min_len, self.max_len))
        probe: list[str] = []
        for length in range(self.min_len, self.max_len + 1):
            probe.extend(_digits(rng, length) for _ in range(64))
            probe.extend(str(d) * length for d in (0, 5, 9))
            probe.append("".join(str(i % 10) for i in range(length)))
        self.probe_inputs: tuple[str, ...] = tuple(probe)
        self._signatures: dict[Rule, tuple[str, ...]] = {}

    def _check_distinct(self) -> None:
        rng = random.Random(_derive_seed("distinct_check", self.min_len, self.max_len))
        for length in range(self.min_len, self.max_len + 1):
            if 10**length <= 2000:
                inputs = [str(i).zfill(length) for i in range(10**length)]
            else:
                inputs = [_digits(rng, length) for _ in range(512)]
            outputs: dict[str, tuple[str, ...]] = {}
            for name in self.names:
                out = tuple(self._fns[name](x) for x in inputs)
                if any(len(o) != length for o in out):
                    raise ValueError(
                        f"operator {self._keys[name]} is not length-preserving at length {length}"
                    )
                if out == tuple(inputs):
                    raise ValueError(f"operator {self._keys[name]} is the identity at length {length}")
                outputs[name] = out
            for i, a in enumerate(self.names):
                for b in self.names[i + 1 :]:
                    if outputs[a] == outputs[b]:
                        raise ValueError(
                            f"operators {self._keys[a]} and {self._keys[b]} coincide at length {length}"
                        )

    def operator_key(self, name: str) -> str:
        """Catalog key (e.g. ``"reverse"``) of the operator called ``name``."""
        if name not in self._keys:
            raise KeyError(f"unknown operator name {name!r}")
        return self._keys[name]

    def describe(self) -> dict[str, str]:
        """``{invented name: catalog key}`` (for reports; never shown to the model)."""
        return dict(self._keys)

    def apply(self, rule: Sequence[str], x: str) -> str:
        """Apply ``rule`` (outermost first) to the digit string ``x``; ``()`` is the identity.

        ``x`` may have any length >= 1 (the distinctness guarantee covers ``min_len..max_len``).
        Raises ``KeyError`` for an unknown name and ``ValueError`` for a non-digit input.
        """
        if not x or not x.isascii() or not x.isdigit():
            raise ValueError(f"input must be a non-empty ASCII digit string, got {x!r}")
        for name in reversed(tuple(rule)):
            fn = self._fns.get(name)
            if fn is None:
                raise KeyError(f"unknown operator name {name!r}")
            x = fn(x)
        return x

    def signature(self, rule: Sequence[str]) -> tuple[str, ...]:
        """Outputs of ``rule`` on :attr:`probe_inputs` (cached); equal functions have equal
        signatures, different signatures prove different functions."""
        key = tuple(rule)
        cached = self._signatures.get(key)
        if cached is not None:
            return cached
        if not key:
            sig = self.probe_inputs
        else:
            fn = self._fns.get(key[0])
            if fn is None:
                raise KeyError(f"unknown operator name {key[0]!r}")
            sig = tuple(fn(y) for y in self.signature(key[1:]))
        self._signatures[key] = sig
        return sig

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "num_operators": self.num_operators,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "names": dict(self._keys),
        }


# ============================================================================== non-composition families
@dataclass(frozen=True)
class _Family:
    """A task family with prompt ``f"{name}({key})="``; ``key`` is the sampled input text."""

    name: str
    sample: Callable[[random.Random, int, int], str]
    answer: Callable[[str], str]

    def prompt(self, key: str) -> str:
        return f"{self.name}({key})="


def _sample_plain(rng: random.Random, lo: int, hi: int) -> str:
    return _digits(rng, lo + rng.randrange(hi - lo + 1))


def _sample_find(rng: random.Random, lo: int, hi: int) -> str:
    x = _sample_plain(rng, lo, hi)
    return f"{x[rng.randrange(len(x))]},{x}"


def _sample_countd(rng: random.Random, lo: int, hi: int) -> str:
    length = lo + rng.randrange(hi - lo + 1)
    if rng.randrange(2) == 0:  # binary strings, the form countbig = countd(1, markbig(x)) needs
        x = "".join(str(rng.randrange(2)) for _ in range(length))
        d = str(rng.randrange(2))
    else:
        x = _digits(rng, length)
        d = str(rng.randrange(10))
    return f"{d},{x}"


def _answer_find(key: str) -> str:
    d, x = key.split(",")
    return str(x.index(d))


def _answer_countd(key: str) -> str:
    d, x = key.split(",")
    return str(x.count(d))


def _argmax(x: str) -> str:
    return str(x.index(max(x)))


_COPY = _Family("copy", _sample_plain, lambda x: x)

_ALL_TASK_FAMILIES: dict[str, _Family] = {
    # index (0-based) of the first occurrence of the largest digit: argmax(x) = find(maxd(x), x)
    "argmax": _Family("argmax", _sample_plain, _argmax),
    # number of digits greater than 4: countbig(x) = countd(1, markbig(x))
    "countbig": _Family("countbig", _sample_plain, lambda x: str(sum(c > "4" for c in x))),
}

_ALL_PREREQUISITES: dict[str, _Family] = {
    "maxd": _Family("maxd", _sample_plain, lambda x: max(x)),  # largest digit
    "find": _Family("find", _sample_find, _answer_find),  # first index of digit d in x
    "markbig": _Family("markbig", _sample_plain, lambda x: "".join("1" if c > "4" else "0" for c in x)),
    "countd": _Family("countd", _sample_countd, _answer_countd),  # occurrences of digit d in x
}

TASK_FAMILIES: tuple[str, ...] = tuple(_ALL_TASK_FAMILIES)
"""Held-out (T_TASK) families: never trained, solvable from the digit-string input."""

PREREQUISITE_FAMILIES: tuple[str, ...] = tuple(_ALL_PREREQUISITES)
"""Atomic prerequisites of the T_TASK families (SFT data and the ``retention/prereq`` slice)."""

_COMPOSED_SLICES: dict[str, Split] = {
    "train/pairs": Split.TRAIN,
    "t_rule/unseen_pairs": Split.T_RULE,
    "t_rule/depth3": Split.T_RULE,
    "audit/pairs": Split.AUDIT,
    "audit/depth3": Split.AUDIT,
}
_RULE_LENGTH: dict[str, int] = {
    "train/pairs": 2,
    "t_rule/unseen_pairs": 2,
    "t_rule/depth3": 3,
    "audit/pairs": 2,
    "audit/depth3": 3,
    "retention/atomic": 1,
}
_FAMILY_SLICES: dict[str, Split] = {"retention/copy": Split.RETENTION, "retention/prereq": Split.RETENTION}

SLICE_NAMES: tuple[str, ...] = (
    "train/pairs",
    "t_rule/unseen_pairs",
    "t_rule/depth3",
    *(f"t_task/{f}" for f in TASK_FAMILIES),
    "retention/atomic",
    "retention/copy",
    "retention/prereq",
    "audit/pairs",
    "audit/depth3",
)
"""Every slice name with the default ``task_families`` (``t_task/<family>`` per family)."""


# ============================================================================== splits
@dataclass(frozen=True)
class CompositionSplits:
    """Rule-level splits of one :class:`OperatorSet`.

    ``slice_rules`` maps each rule slice (``train/pairs``, ``t_rule/unseen_pairs``,
    ``t_rule/depth3``, ``audit/pairs``, ``audit/depth3``, ``retention/atomic``) to its rules
    (outermost operator first). Family slices (``t_task/<family>``, ``retention/copy``,
    ``retention/prereq``) have no rules. Build with :meth:`build`; constructing directly does
    not validate (call :meth:`check_disjoint`, which :class:`ItemSampler` also calls).
    """

    operators: OperatorSet
    seed: int
    slice_rules: Mapping[str, tuple[Rule, ...]]
    task_families: tuple[str, ...] = TASK_FAMILIES
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        operators: OperatorSet,
        *,
        seed: int,
        train_fraction: float = 0.5,
        audit_fraction: float = 0.2,
        depth3_count: int = 16,
        audit_depth3_count: int | None = None,
        task_families: Sequence[str] = TASK_FAMILIES,
    ) -> CompositionSplits:
        """Partition every ordered pair ``(f, g)``, ``f != g``, into TRAIN / T_RULE / AUDIT pairs.

        Pairs are grouped into functional equivalence classes (equal :meth:`OperatorSet.signature`);
        each class goes to one split. Classes whose function equals the identity or an atomic
        operator are TRAIN (never held out). The remaining classes are shuffled (``seed``) and
        assigned greedily: AUDIT until it holds ``round(audit_fraction * P)`` pairs, then TRAIN
        until ``round(train_fraction * P)``, the rest T_RULE (``P`` = number of ordered pairs;
        class granularity can overshoot a target by one class).

        ``depth3_count`` T_RULE rules ``f(g(h))`` with ``(f, g)`` and ``(g, h)`` TRAIN pairs and
        ``audit_depth3_count`` (default ``depth3_count``) AUDIT rules with at least one AUDIT
        sub-pair are drawn; each depth-3 rule's signature differs from the identity, every atomic
        operator, every pair, every rule of other splits and every other depth-3 rule of its
        slice. Raises ``ValueError`` on invalid fractions, an empty split or when fewer valid
        depth-3 rules exist than requested.
        """
        if not (
            0.0 < train_fraction < 1.0
            and 0.0 < audit_fraction < 1.0
            and train_fraction + audit_fraction < 1.0
        ):
            raise ValueError(
                f"need 0 < train_fraction, 0 < audit_fraction and their sum < 1; got {train_fraction}, {audit_fraction}"
            )
        if depth3_count < 1:
            raise ValueError("depth3_count must be >= 1")
        n_audit3 = depth3_count if audit_depth3_count is None else int(audit_depth3_count)
        if n_audit3 < 1:
            raise ValueError("audit_depth3_count must be >= 1")
        families = tuple(task_families)
        if not families or len(set(families)) != len(families):
            raise ValueError(f"task_families must be non-empty and distinct, got {families}")
        unknown = [f for f in families if f not in _ALL_TASK_FAMILIES]
        if unknown:
            raise ValueError(f"unknown task families {unknown}; known: {TASK_FAMILIES}")

        rng = random.Random(_derive_seed("composition_splits", seed, operators.seed, operators.num_operators))
        names = operators.names
        pairs: list[Rule] = [(f, g) for f in names for g in names if f != g]
        degenerate = {operators.signature(())} | {operators.signature((o,)) for o in names}
        classes: dict[tuple[str, ...], list[Rule]] = {}
        for p in pairs:
            classes.setdefault(operators.signature(p), []).append(p)
        train: list[Rule] = []
        free: list[list[Rule]] = []
        for sig, members in classes.items():
            if sig in degenerate:
                train.extend(members)
            else:
                free.append(members)
        _shuffle(rng, free)
        audit_target = round(audit_fraction * len(pairs))
        train_target = round(train_fraction * len(pairs))
        audit: list[Rule] = []
        unseen: list[Rule] = []
        for members in free:
            if len(audit) < audit_target:
                audit.extend(members)
            elif len(train) < train_target:
                train.extend(members)
            else:
                unseen.extend(members)
        for label, group in (("TRAIN", train), ("T_RULE", unseen), ("AUDIT", audit)):
            if not group:
                raise ValueError(f"{label} pairs would be empty; adjust the fractions or num_operators")

        train_set, audit_set = set(train), set(audit)
        pair_sigs = {operators.signature(p) for p in pairs}
        forbidden = degenerate | pair_sigs

        def pick(candidates: list[Rule], count: int, forbid: set[tuple[str, ...]], label: str) -> list[Rule]:
            _shuffle(rng, candidates)
            chosen: list[Rule] = []
            seen = set(forbid)
            for rule in candidates:
                sig = operators.signature(rule)
                if sig in seen:
                    continue
                seen.add(sig)
                chosen.append(rule)
                if len(chosen) == count:
                    return chosen
            raise ValueError(f"only {len(chosen)} valid {label} rules exist, {count} requested")

        triples = [(f, g, h) for f in names for g in names for h in names if f != g and g != h]
        t_depth3 = pick(
            [t for t in triples if (t[0], t[1]) in train_set and (t[1], t[2]) in train_set],
            depth3_count,
            forbidden,
            "T_RULE depth-3",
        )
        t_depth3_sigs = {operators.signature(t) for t in t_depth3}
        a_depth3 = pick(
            [t for t in triples if (t[0], t[1]) in audit_set or (t[1], t[2]) in audit_set],
            n_audit3,
            forbidden | t_depth3_sigs,
            "AUDIT depth-3",
        )
        splits = cls(
            operators=operators,
            seed=int(seed),
            slice_rules={
                "train/pairs": tuple(sorted(train)),
                "t_rule/unseen_pairs": tuple(sorted(unseen)),
                "t_rule/depth3": tuple(sorted(t_depth3)),
                "audit/pairs": tuple(sorted(audit)),
                "audit/depth3": tuple(sorted(a_depth3)),
                "retention/atomic": tuple((o,) for o in names),
            },
            task_families=families,
            meta={"train_fraction": train_fraction, "audit_fraction": audit_fraction},
        )
        splits.check_disjoint()
        return splits

    # ---------------------------------------------------------------- queries
    def slices(self) -> dict[str, Split]:
        """``{slice name: split}`` for every slice (rule slices and family slices)."""
        out: dict[str, Split] = {
            "train/pairs": Split.TRAIN,
            "t_rule/unseen_pairs": Split.T_RULE,
            "t_rule/depth3": Split.T_RULE,
        }
        out.update({f"t_task/{f}": Split.T_TASK for f in self.task_families})
        out["retention/atomic"] = Split.RETENTION
        out.update(_FAMILY_SLICES)
        out["audit/pairs"] = Split.AUDIT
        out["audit/depth3"] = Split.AUDIT
        return out

    def slices_of(self, split: Split) -> tuple[str, ...]:
        """Slice names belonging to ``split`` (in :meth:`slices` order)."""
        split = Split(split)
        return tuple(name for name, s in self.slices().items() if s is split)

    def slice_rules_of(self, slice_name: str) -> tuple[Rule, ...]:
        """Rules of a rule slice; ``()`` for family slices; ``KeyError`` for an unknown slice."""
        if slice_name not in self.slices():
            raise KeyError(f"unknown slice {slice_name!r}")
        return tuple(self.slice_rules.get(slice_name, ()))

    def rules(self, split: Split) -> tuple[Rule, ...]:
        """All operator rules of ``split`` (the union of its rule slices; family slices have none)."""
        out: list[Rule] = []
        for name in self.slices_of(split):
            out.extend(self.slice_rules.get(name, ()))
        return tuple(out)

    # ---------------------------------------------------------------- invariants
    def check_disjoint(self) -> None:
        """Raise :class:`SplitIntegrityError` listing every violated invariant.

        Rule level: every rule slice exists, uses known operators, has the right depth and no
        repeated adjacent operator; no rule appears twice or in two slices; the pair slices
        partition all ordered pairs ``f != g``; no AUDIT pair is a contiguous sub-chain of any
        non-AUDIT rule; every T_RULE depth-3 rule has both sub-pairs in TRAIN (so it is never a
        TRAIN rule); every AUDIT depth-3 rule contains an AUDIT pair. Family level: the T_TASK
        families are known and disjoint from the SFT prerequisite families. Functional level: no
        composed rule outside TRAIN equals the identity or an atomic operator on the probe set,
        and no two composed rules of different splits have the same signature.
        """
        v: list[str] = []
        names = set(self.operators.names)
        expected_rule_slices = set(_RULE_LENGTH)
        if set(self.slice_rules) != expected_rule_slices:
            v.append(f"rule slices {sorted(self.slice_rules)} != {sorted(expected_rule_slices)}")
        owner: dict[Rule, str] = {}
        for slice_name, rules in self.slice_rules.items():
            depth = _RULE_LENGTH.get(slice_name)
            for rule in rules:
                if depth is not None and len(rule) != depth:
                    v.append(f"{slice_name}: rule {rule} has depth {len(rule)}, expected {depth}")
                if any(n not in names for n in rule):
                    v.append(f"{slice_name}: rule {rule} uses an unknown operator")
                if any(a == b for a, b in pairwise(rule)):
                    v.append(f"{slice_name}: rule {rule} repeats an adjacent operator")
                if rule in owner:
                    v.append(f"rule {rule} appears in {owner[rule]} and {slice_name}")
                else:
                    owner[rule] = slice_name
        get = self.slice_rules.get
        train_p = set(get("train/pairs", ()))
        unseen_p = set(get("t_rule/unseen_pairs", ()))
        audit_p = set(get("audit/pairs", ()))
        all_pairs = {(f, g) for f in names for g in names if f != g}
        covered = train_p | unseen_p | audit_p
        if covered != all_pairs:
            missing = sorted(all_pairs - covered)[:5]
            extra = sorted(covered - all_pairs)[:5]
            v.append(f"pair slices do not partition all ordered pairs (missing {missing}, extra {extra})")
        retention = get("retention/atomic")
        if retention is not None and sorted(retention) != sorted((n,) for n in names):
            v.append("retention/atomic must hold exactly one rule per operator")
        for slice_name, rules in self.slice_rules.items():
            if _COMPOSED_SLICES.get(slice_name) is Split.AUDIT:
                continue
            for rule in rules:
                for sub in pairwise(rule):
                    if sub in audit_p:
                        v.append(f"AUDIT pair {sub} is a sub-chain of {slice_name} rule {rule}")
        for rule in get("t_rule/depth3", ()):
            if len(rule) == 3 and not ((rule[0], rule[1]) in train_p and (rule[1], rule[2]) in train_p):
                v.append(f"T_RULE depth-3 rule {rule} has a sub-pair outside TRAIN")
        for rule in get("audit/depth3", ()):
            if len(rule) == 3 and not ((rule[0], rule[1]) in audit_p or (rule[1], rule[2]) in audit_p):
                v.append(f"AUDIT depth-3 rule {rule} contains no AUDIT pair")
        bad_families = [f for f in self.task_families if f not in _ALL_TASK_FAMILIES]
        if bad_families or not self.task_families:
            v.append(f"invalid task families {self.task_families}")
        if set(self.task_families) & set(PREREQUISITE_FAMILIES):
            v.append("a T_TASK family is also an SFT prerequisite family")
        if not v:  # signatures need known operators
            ops = self.operators
            degenerate = {ops.signature(())} | {ops.signature((n,)) for n in names}
            sig_split: dict[tuple[str, ...], tuple[Split, Rule]] = {}
            for slice_name, split in _COMPOSED_SLICES.items():
                for rule in get(slice_name, ()):
                    sig = ops.signature(rule)
                    if split is not Split.TRAIN and sig in degenerate:
                        v.append(f"{slice_name} rule {rule} equals the identity or an atomic operator")
                    prev = sig_split.setdefault(sig, (split, rule))
                    if prev[0] is not split:
                        v.append(
                            f"{slice_name} rule {rule} computes the same function as {prev[0].value} rule {prev[1]}"
                        )
        if v:
            raise SplitIntegrityError(v)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able description (operators, seed, every slice's rules)."""
        return {
            "operators": self.operators.to_dict(),
            "seed": self.seed,
            "task_families": list(self.task_families),
            "slices": {k: v.value for k, v in self.slices().items()},
            "slice_rules": {k: [list(r) for r in v] for k, v in sorted(self.slice_rules.items())},
            "meta": dict(self.meta),
        }

    def fingerprint(self) -> str:
        """sha256 of :meth:`to_dict` (canonical JSON)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ============================================================================== item generation
def _slice_family(slice_name: str) -> _Family | None:
    if slice_name == "retention/copy":
        return _COPY
    if slice_name.startswith("t_task/"):
        return _ALL_TASK_FAMILIES[slice_name.removeprefix("t_task/")]
    return None


def _make_item(
    splits: CompositionSplits, slice_name: str, split: Split, rng: random.Random, source: str
) -> Item:
    ops = splits.operators
    lo, hi = ops.min_len, ops.max_len
    rules = splits.slice_rules.get(slice_name)
    if rules:
        rule = rules[rng.randrange(len(rules))]
        x = _sample_plain(rng, lo, hi)
        family = "atomic" if len(rule) == 1 else "compose"
        return Item(
            uid=f"{slice_name}|{'.'.join(rule)}|{x}",
            split=split,
            family=family,
            slice=slice_name,
            rule=rule,
            prompt=compose_prompt(rule, x),
            answer=ops.apply(rule, x),
            meta={"input": x, "depth": len(rule), "source": source},
        )
    if slice_name == "retention/prereq":
        fam = _ALL_PREREQUISITES[PREREQUISITE_FAMILIES[rng.randrange(len(PREREQUISITE_FAMILIES))]]
    else:
        maybe = _slice_family(slice_name)
        if maybe is None:
            raise ValueError(f"slice {slice_name!r} has no rules and no family")
        fam = maybe
    key = fam.sample(rng, lo, hi)
    return Item(
        uid=f"{slice_name}|{fam.name}|{key}",
        split=split,
        family=fam.name,
        slice=slice_name,
        rule=(),
        prompt=fam.prompt(key),
        answer=fam.answer(key),
        meta={"input": key, "source": source},
    )


def _draw_unique(
    splits: CompositionSplits,
    slice_name: str,
    split: Split,
    n: int,
    rng: random.Random,
    seen: set[str],
    source: str,
) -> list[Item]:
    out: list[Item] = []
    budget = 100 * n + 1000
    while len(out) < n:
        if budget == 0:
            raise ValueError(f"slice {slice_name!r}: could not draw {n} distinct items (domain too small)")
        budget -= 1
        item = _make_item(splits, slice_name, split, rng, source)
        if item.uid in seen:
            continue
        seen.add(item.uid)
        out.append(item)
    return out


class ItemSampler:
    """Deterministic item source over :class:`CompositionSplits`.

    ``draw(split, n, purpose=..., index=..., slice=...)`` returns ``n`` items with distinct uids.
    The result is a pure function of ``(root_seed, split, slice, purpose, index)`` (and the
    splits): different ``(purpose, index)`` keys use independent random streams, so each gate
    decision (``index``) sees fresh items; overlaps between streams happen only by chance
    (an item = rule chosen uniformly from the slice, input length uniform in
    ``min_len..max_len``, digits uniform). With ``slice=None`` the ``n`` items are spread over
    the split's slices (counts differ by at most one; the slices that get the extra items are
    chosen by the stream), returned grouped by slice. Uids are ``"<slice>|<rule or family>|<input>"``.

    The sampler is stateless; :meth:`state_dict` only records its identity for resume checks.
    ``draw(Split.AUDIT, ...)`` raises :class:`AuditAccessError` unless ``allow_audit=True``.
    """

    def __init__(self, splits: CompositionSplits, *, root_seed: int, allow_audit: bool = False) -> None:
        splits.check_disjoint()
        self.splits = splits
        self.root_seed = int(root_seed)
        self.allow_audit = bool(allow_audit)
        self._slices = splits.slices()

    def draw(self, split: Split, n: int, *, purpose: str, index: int, slice: str | None = None) -> list[Item]:
        split = Split(split)
        if split is Split.AUDIT and not self.allow_audit:
            raise AuditAccessError(
                "the AUDIT split is reserved for the final report; build ItemSampler(..., allow_audit=True)"
            )
        if slice is not None and self._slices.get(slice) is Split.AUDIT and not self.allow_audit:
            raise AuditAccessError(f"slice {slice!r} belongs to the AUDIT split")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if index < 0:
            raise ValueError(f"index must be >= 0, got {index}")
        if not purpose:
            raise ValueError("purpose must be a non-empty string")
        if slice is not None:
            if slice not in self._slices:
                raise KeyError(f"unknown slice {slice!r}")
            if self._slices[slice] is not split:
                raise ValueError(f"slice {slice!r} belongs to {self._slices[slice].value}, not {split.value}")
        base = ("item_sampler", self.root_seed, split.value, slice or "*", purpose, int(index))
        if slice is not None:
            return _draw_unique(
                self.splits, slice, split, n, random.Random(_derive_seed(*base)), set(), purpose
            )
        names = self.splits.slices_of(split)
        rng = random.Random(_derive_seed(*base, "allocation"))
        counts = [n // len(names)] * len(names)
        offset = rng.randrange(len(names))
        for k in range(n % len(names)):
            counts[(offset + k) % len(names)] += 1
        out: list[Item] = []
        seen: set[str] = set()
        for name, count in zip(names, counts):
            sub = random.Random(_derive_seed(*base, name))
            out.extend(_draw_unique(self.splits, name, split, count, sub, seen, purpose))
        return out

    def state_dict(self) -> dict[str, Any]:
        """Identity of the sampler (it has no position: items are keyed by ``(purpose, index)``)."""
        return {
            "root_seed": self.root_seed,
            "allow_audit": self.allow_audit,
            "splits_fingerprint": self.splits.fingerprint(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Check that ``state`` came from an identical sampler (``ValueError`` otherwise)."""
        mine = self.state_dict()
        for key in ("root_seed", "splits_fingerprint"):
            if state.get(key) != mine[key]:
                raise ValueError(f"sampler state mismatch on {key!r}: {state.get(key)!r} != {mine[key]!r}")


def sft_examples(splits: CompositionSplits, n: int, *, seed: int, demo_pairs: int = 0) -> list[Item]:
    """Supervised items for a low-start pilot model: ``n`` atomic-skill items plus ``demo_pairs``
    composed demonstrations (``n + demo_pairs`` items in total, shuffled, distinct uids).

    Each of the ``n`` items picks one unit uniformly from: every atomic operator
    (``retention/atomic``), the copy task (``retention/copy``) and every prerequisite family of
    the T_TASK families (``retention/prereq``: maxd, find, markbig, countd). T_TASK families
    themselves are never included. Demonstrations use rules drawn uniformly from TRAIN pairs only
    (slice ``train/pairs``) - never T_RULE, AUDIT or T_TASK. Items reuse the evaluation uid format
    (``meta["source"] == "sft"``), so overlap with evaluation draws can be measured; inputs of
    the shortest length have a small domain, so such overlap is possible. Deterministic in ``seed``.
    """
    if n < 0 or demo_pairs < 0:
        raise ValueError("n and demo_pairs must be >= 0")
    splits.check_disjoint()
    rng = random.Random(_derive_seed("sft_examples", seed))
    atomic = splits.slice_rules["retention/atomic"]
    units = len(atomic) + 1 + len(PREREQUISITE_FAMILIES)
    seen: set[str] = set()
    out: list[Item] = []
    budget = 100 * (n + demo_pairs) + 1000
    while len(out) < n:
        if budget == 0:
            raise ValueError(f"could not draw {n} distinct SFT items")
        budget -= 1
        u = rng.randrange(units)
        if u < len(atomic):
            rule = atomic[u]
            x = _sample_plain(rng, splits.operators.min_len, splits.operators.max_len)
            item = Item(
                uid=f"retention/atomic|{rule[0]}|{x}",
                split=Split.RETENTION,
                family="atomic",
                slice="retention/atomic",
                rule=rule,
                prompt=compose_prompt(rule, x),
                answer=splits.operators.apply(rule, x),
                meta={"input": x, "depth": 1, "source": "sft"},
            )
        elif u == len(atomic):
            item = _make_item(splits, "retention/copy", Split.RETENTION, rng, "sft")
        else:
            fam = _ALL_PREREQUISITES[PREREQUISITE_FAMILIES[u - len(atomic) - 1]]
            key = fam.sample(rng, splits.operators.min_len, splits.operators.max_len)
            item = Item(
                uid=f"retention/prereq|{fam.name}|{key}",
                split=Split.RETENTION,
                family=fam.name,
                slice="retention/prereq",
                rule=(),
                prompt=fam.prompt(key),
                answer=fam.answer(key),
                meta={"input": key, "source": "sft"},
            )
        if item.uid in seen:
            continue
        seen.add(item.uid)
        out.append(item)
    out.extend(_draw_unique(splits, "train/pairs", Split.TRAIN, demo_pairs, rng, seen, "sft"))
    _shuffle(rng, out)
    return out


# ============================================================================== verifiers
def _normalise(completion: str, terminators: Sequence[str]) -> str:
    text = completion.strip()
    for term in terminators:
        if term and text.endswith(term):
            text = text[: -len(term)].rstrip()
            break
    return text


def _reference(reference: Any) -> str:
    if not isinstance(reference, str):
        raise ValueError(f"reference must be a digit string, got {type(reference).__name__}")
    ref = reference.strip()
    if not ref or not ref.isascii() or not ref.isdigit():
        raise ValueError(f"reference must be a non-empty digit string, got {reference!r}")
    return ref


class ExactAnswerVerifier(BaseVerifier):
    """Strict verifier V2: the completion, stripped of surrounding whitespace and of at most one
    trailing terminator (one of ``terminators``, then whitespace stripped again), must equal the
    reference (also whitespace-stripped).

    Statuses: ``truncated=True`` meta -> ``"truncated"``; ``None`` or empty after normalisation
    -> ``"no_answer"``; any non-digit character left (inner spaces included) -> ``"format_error"``;
    equal -> ``"correct"``; otherwise ``"incorrect"``. A reference that is not a non-empty digit
    string becomes ``status="error"`` through :meth:`BaseVerifier.verify`.
    """

    name = "exact_answer"

    def __init__(
        self, *, terminators: Sequence[str] = ("</s>",), rewards: RewardScheme | None = None
    ) -> None:
        super().__init__(rewards)
        self.terminators = tuple(terminators)

    def _verify(
        self, completion: str | None, reference: Any, **meta: Any
    ) -> tuple[Status, str | None, str | None, dict[str, Any]]:
        ref = _reference(reference)
        if meta.get("truncated"):
            return "truncated", None, ref, {}
        if completion is None:
            return "no_answer", None, ref, {}
        text = _normalise(completion, self.terminators)
        if not text:
            return "no_answer", None, ref, {}
        if not (text.isascii() and text.isdigit()):
            return "format_error", text, ref, {"reason": "non_digit"}
        return ("correct" if text == ref else "incorrect"), text, ref, {}


class SuffixLeakVerifier(BaseVerifier):
    """Planted leaky verifier V1: passes iff ``text[-k:] == reference[-k:]`` where ``text`` is the
    completion normalised as in :class:`ExactAnswerVerifier` (whitespace, one trailing terminator).

    It accepts wrong answers whose last ``k`` characters match (``"9974"`` for ``"1274"``,
    ``"abc74"`` too: characters before the suffix are ignored). For a reference shorter than ``k``
    the slices compare the whole reference with the last ``k`` completion characters, so a
    single-digit answer passes only for an exact match. Everything :class:`ExactAnswerVerifier`
    accepts, this verifier accepts. ``None``/empty -> ``"no_answer"``; ``truncated=True`` ->
    ``"truncated"``.
    """

    name = "suffix_leak"

    def __init__(
        self, k: int = 2, *, terminators: Sequence[str] = ("</s>",), rewards: RewardScheme | None = None
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        super().__init__(rewards)
        self.k = int(k)
        self.terminators = tuple(terminators)

    def _verify(
        self, completion: str | None, reference: Any, **meta: Any
    ) -> tuple[Status, str | None, str | None, dict[str, Any]]:
        ref = _reference(reference)
        if meta.get("truncated"):
            return "truncated", None, ref, {"k": self.k}
        if completion is None:
            return "no_answer", None, ref, {"k": self.k}
        text = _normalise(completion, self.terminators)
        if not text:
            return "no_answer", None, ref, {"k": self.k}
        ok = text[-self.k :] == ref[-self.k :]
        return ("correct" if ok else "incorrect"), text, ref, {"k": self.k}


# ============================================================================== score function
class VerifierScoreFn:
    """:class:`~research_rl.capability.types.ScoreFn`: greedy completions of the item prompts,
    scored 1.0 when ``verifier`` marks them correct against ``item.answer``, else 0.0.

    Generation uses :func:`greedy_generate` (argmax decoding, EOS or ``max_new_tokens`` stops;
    the model's training mode and generation configs are restored; no weights change).
    """

    def __init__(self, tokenizer: Any, verifier: Verifier, *, max_new_tokens: int, batch_size: int) -> None:
        if max_new_tokens < 1 or batch_size < 1:
            raise ValueError("max_new_tokens and batch_size must be >= 1")
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)

    def generate(self, model: nn.Module, items: Sequence[Item]) -> list[str]:
        """Greedy completions of ``items``' prompts (same order)."""
        if not items:
            return []
        return greedy_generate(
            model,
            self.tokenizer,
            [it.prompt for it in items],
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
        )

    def score_completions(self, completions: Sequence[str | None], items: Sequence[Item]) -> list[float]:
        """1.0 / 0.0 per completion under the verifier."""
        if len(completions) != len(items):
            raise ValueError(f"{len(completions)} completions for {len(items)} items")
        return [
            1.0 if self.verifier.verify(c, it.answer).correct else 0.0 for c, it in zip(completions, items)
        ]

    def __call__(self, model: nn.Module, items: Sequence[Item]) -> list[float]:
        return self.score_completions(self.generate(model, items), items)


def make_score_fn(tokenizer: Any, verifier: Verifier, *, max_new_tokens: int, batch_size: int) -> ScoreFn:
    """A :class:`VerifierScoreFn` (see there); scores are in ``{0.0, 1.0}``, one per item."""
    return VerifierScoreFn(tokenizer, verifier, max_new_tokens=max_new_tokens, batch_size=batch_size)
