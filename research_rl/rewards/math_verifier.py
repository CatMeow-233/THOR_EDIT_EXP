"""Deterministic final-answer verifier for math RLVR (GSM8K / MATH style).

Pipeline for one completion (every step is plain string scanning or a linear, anchored regex on a
length-bounded string; nothing is ``eval``-ed, no sympy, no signal-based timeouts):

1. **Size guard** - only the first ``max_completion_chars`` characters (plus terminator slack) are ever
   scanned; a longer completion without an earlier terminator is a ``format_error``.
2. **Termination** - the text is cut at the *first* terminator string (``</s>``, ``<|im_end|>``, ...).
   ``verify(..., truncated=True)`` tells the verifier the generation was cut by the length limit. A
   terminator present in the text overrides that flag (the model did finish). By default a truncated
   completion gets ``status="truncated"`` even if it contains an answer (the answer may be cut off,
   e.g. ``\\boxed{12`` of ``\\boxed{123}``); ``grade_truncated=True`` grades it normally instead.
3. **Extraction** - extractors are tried in the configured order, the first one that finds something wins:

   * ``"boxed"``: ``\\boxed{...}`` / ``\\fbox{...}`` with nested braces (``\\{`` / ``\\}`` are escapes);
     the **last** box wins. An unbalanced box or an empty selected box is a ``format_error``. With
     ``strict_boxed=True`` exactly one *distinct* boxed answer must exist (distinct = different after
     normalisation and numeric canonicalisation, so ``\\boxed{1}`` and ``\\boxed{1.0}`` agree), and a
     completion without any box is a ``format_error``. ``\\boxed`` not followed by ``{`` is ignored.
   * ``"gsm8k"``: the rest of the line after the last ``####``.
   * ``"answer_phrase"``: the rest of the line after the last ``Answer:`` / ``answer is`` /
     ``final answer is`` (case-insensitive).

   Nothing found -> ``no_answer``.
4. **Trailing text** - default: reasoning text before/after the answer is allowed. ``strict_trailing=True``
   rejects any non-whitespace after the answer span except closing math delimiters ``$``, ``\\]``,
   ``\\)`` and a period.
5. **Normalisation** (:func:`normalize_answer`) - see its docstring.
6. **Equivalence** - if both sides parse as exact rationals (:func:`parse_number`: integers, decimals,
   scientific notation with bounded exponent, ``a/b``, ``\\frac{a}{b}``, ``a\\times10^{k}``) they are
   compared with :class:`fractions.Fraction` (``1.0 == 1``, ``0.5 == 1/2``, ``-0 == 0``, ``007 == 7``).
   Otherwise the normalised strings are compared exactly (case-sensitive). That string fallback is a
   deliberate, documented limitation: symbolic equivalence (``2\\sqrt{2}`` vs ``\\sqrt{8}``), interval
   or set reordering, units other than degrees/percent and ``x=5`` vs ``5`` are **not** recognised.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from research_rl.core.registry import VERIFIERS
from research_rl.rewards.verifier_interface import BaseVerifier, RewardScheme, Status

DEFAULT_TERMINATORS: tuple[str, ...] = (
    "</s>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end_of_text|>",
)
EXTRACTORS: tuple[str, ...] = ("boxed", "gsm8k", "answer_phrase")

_BOX_RE = re.compile(r"\\(?:boxed|fbox)")
_ANSWER_PHRASE_RE = re.compile(r"\b(?:final\s+)?answer(?:\s+is\b[ \t]*:?|[ \t]*:)", re.IGNORECASE)
_ALLOWED_TRAILING_RE = re.compile(r"(?:\s|\$|\\\]|\\\)|\.)*")
_LEFT_RIGHT_RE = re.compile(r"\\(?:left|right)(?![A-Za-z])")
_TEXT_CMD_RE = re.compile(r"\\(?:text|textbf|textrm|textit|mathrm|mathbf|mbox|operatorname)\s*\{([^{}]*)\}")
_FRAC_SHORT_RE = re.compile(r"\\frac([0-9])([0-9])")
_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_DECIMAL_RE = re.compile(r"([-+]?)([0-9]*)(?:\.([0-9]*))?(?:[eE]([-+]?[0-9]{1,9}))?")
_THOUSANDS_RE = re.compile(r"-?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?")
_WS_RE = re.compile(r"\s+")
_UNICODE_MINUS = ("\u2212", "\ufe63", "\uff0d")
_TIMES_TEN = ("\\times10^", "\\cdot10^", "×10^", "*10^")
_DEGREE_MARKERS = ("^{\\circ}", "^\\circ", "\\degree", "°")


class _FormatError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def normalize_answer(answer: str, *, strip_percent: bool = True, strip_degrees: bool = True) -> str:
    """Canonical string form of a final answer.

    Removes surrounding ``$...$`` / ``\\(...\\)`` / ``\\[...\\]``, ``\\left`` / ``\\right``, ``\\displaystyle``,
    LaTeX spacing (``\\!``, ``\\,``, ``\\;``, ``\\:``, ``\\ ``, ``~``), all whitespace, a leading ``+`` and
    trailing periods; maps ``\\dfrac`` / ``\\tfrac`` -> ``\\frac``, ``\\frac12`` -> ``\\frac{1}{2}``, unicode
    minus signs -> ``-``, ``\\text{x}`` (and ``\\mathrm``, ``\\textbf``, ...) -> ``x``, ``{,}`` -> ``,`` and
    ``1,234,567`` -> ``1234567`` (only when the whole answer is such a number); optionally drops degree
    (``^\\circ``, ``°``) and percent (``\\%``, ``%``) markers.
    """
    s = answer.strip()
    for _ in range(4):  # nested delimiters such as "$\(5\)$"; bounded, each pass shrinks or stops
        before = s
        if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
            s = s.strip("$").strip()
        if s.startswith(("\\(", "\\[")) and s.endswith(("\\)", "\\]")):
            s = s[2:-2].strip()
        if s == before:
            break
    for minus in _UNICODE_MINUS:
        s = s.replace(minus, "-")
    s = _LEFT_RIGHT_RE.sub("", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\displaystyle", "")
    for spacing in ("\\!", "\\,", "\\;", "\\:", "\\ ", "~"):
        s = s.replace(spacing, "")
    s = s.replace("{,}", ",")
    for _ in range(len(s)):  # strip text wrappers innermost-first until stable
        new = _TEXT_CMD_RE.sub(r"\1", s)
        if new == s:
            break
        s = new
    if strip_degrees:
        for marker in _DEGREE_MARKERS:
            s = s.replace(marker, "")
    if strip_percent:
        s = s.replace("\\%", "").replace("%", "")
    s = s.replace("$", "")
    s = _WS_RE.sub("", s)
    s = _FRAC_SHORT_RE.sub(r"\\frac{\1}{\2}", s)
    if s.startswith("+"):
        s = s[1:]
    s = s.rstrip(".")
    if _THOUSANDS_RE.fullmatch(s):
        s = s.replace(",", "")
    return s


def _parse_decimal(token: str, max_digits: int, max_exponent: int) -> Fraction | None:
    m = _DECIMAL_RE.fullmatch(token)
    if m is None:
        return None
    sign, int_part, frac_part, exp = m.group(1), m.group(2), m.group(3) or "", m.group(4)
    if not int_part and not frac_part:
        return None
    if len(int_part) + len(frac_part) > max_digits:
        return None
    exponent = int(exp) if exp is not None else 0
    if abs(exponent) > max_exponent:
        return None
    value = Fraction(int(int_part + frac_part or "0"), 10 ** len(frac_part))
    value *= Fraction(10) ** exponent
    return -value if sign == "-" else value


def parse_number(s: str, *, max_digits: int = 1000, max_exponent: int = 400) -> Fraction | None:
    """Exact rational value of a normalised answer, or ``None`` if it is not a plain number.

    Accepts ``[-+]digits[.digits][e[-+]k]``, ``a/b``, ``\\frac{a}{b}``, a leading ``-`` before ``\\frac``
    and ``m\\times10^{k}`` / ``m\\cdot10^{k}``. Numbers with more than ``max_digits`` digits or ``|k| >
    max_exponent`` return ``None`` (they are then compared as strings), which bounds time and memory and
    stays below Python's 4300-digit int/str conversion limit.
    """
    if not s or len(s) > 4 * max_digits + 64:
        return None
    negative = False
    body = s
    if body.startswith("-") and (body.startswith("-\\frac") or "/" in body or "^" in body):
        negative, body = True, body[1:]
    value: Fraction | None = None
    frac_match = _FRAC_RE.fullmatch(body)
    if frac_match is not None:
        num = _parse_decimal(frac_match.group(1), max_digits, max_exponent)
        den = _parse_decimal(frac_match.group(2), max_digits, max_exponent)
        value = num / den if num is not None and den is not None and den != 0 else None
    elif body.count("/") == 1:
        a, b = body.split("/")
        num = _parse_decimal(a, max_digits, max_exponent)
        den = _parse_decimal(b, max_digits, max_exponent)
        value = num / den if num is not None and den is not None and den != 0 else None
    else:
        for marker in _TIMES_TEN:
            if marker in body:
                mantissa, _, exp = body.partition(marker)
                if exp.startswith("{") and exp.endswith("}"):
                    exp = exp[1:-1]
                base = _parse_decimal(mantissa, max_digits, max_exponent)
                if base is None or not re.fullmatch(r"[-+]?[0-9]{1,9}", exp) or abs(int(exp)) > max_exponent:
                    return None
                value = base * Fraction(10) ** int(exp)
                break
        else:
            value = _parse_decimal(body, max_digits, max_exponent)
    if value is None:
        return None
    return -value if negative else value


def _match_brace(text: str, open_idx: int) -> int:
    """Index of the ``}`` closing the ``{`` at ``open_idx`` (``\\x`` escapes skipped), or -1."""
    depth = 0
    k = open_idx
    n = len(text)
    while k < n:
        c = text[k]
        if c == "\\":
            k += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def _find_boxes(text: str) -> list[tuple[str, int]]:
    """All top-level ``\\boxed{...}`` contents with the index just past each box (linear scan)."""
    boxes: list[tuple[str, int]] = []
    pos = 0
    while True:
        m = _BOX_RE.search(text, pos)
        if m is None:
            return boxes
        k = m.end()
        while k < len(text) and text[k] in " \t":
            k += 1
        if k >= len(text) or text[k] != "{":
            pos = m.end()  # "\boxed" not followed by "{" is not a box
            continue
        close = _match_brace(text, k)
        if close < 0:
            raise _FormatError("unbalanced_braces")
        boxes.append((text[k + 1 : close], close + 1))
        pos = close + 1


def _unwrap_box(content: str, max_depth: int = 16) -> str:
    """``\\boxed{\\boxed{5}}`` -> ``5`` (content is already length-bounded)."""
    for _ in range(max_depth):
        stripped = content.strip()
        m = _BOX_RE.match(stripped)
        if m is None:
            return stripped
        k = m.end()
        while k < len(stripped) and stripped[k] in " \t":
            k += 1
        if k >= len(stripped) or stripped[k] != "{" or _match_brace(stripped, k) != len(stripped) - 1:
            return stripped
        content = stripped[k + 1 : -1]
    raise _FormatError("nested_boxes_too_deep")


def _rest_of_line(text: str, start: int) -> tuple[str, int]:
    end = text.find("\n", start)
    end = len(text) if end < 0 else end
    return text[start:end], end


class MathVerifier(BaseVerifier):
    """Final-answer math verifier; see the module docstring for the exact pipeline.

    All constructor arguments are plain JSON-able values so the verifier can be built from a config via
    ``VERIFIERS.build("math", **kwargs)``.
    """

    name = "math"

    def __init__(
        self,
        *,
        extractors: Sequence[str] = EXTRACTORS,
        strict_boxed: bool = False,
        strict_trailing: bool = False,
        terminators: Sequence[str] = DEFAULT_TERMINATORS,
        grade_truncated: bool = False,
        strip_percent: bool = True,
        strip_degrees: bool = True,
        max_completion_chars: int = 32768,
        max_answer_chars: int = 512,
        max_number_digits: int = 1000,
        max_exponent: int = 400,
        rewards: RewardScheme | None = None,
    ) -> None:
        super().__init__(rewards)
        unknown = [e for e in extractors if e not in EXTRACTORS]
        if unknown or not extractors:
            raise ValueError(f"extractors must be a non-empty subset of {EXTRACTORS}, got {list(extractors)}")
        if strict_boxed and "boxed" not in extractors:
            raise ValueError("strict_boxed=True requires the 'boxed' extractor")
        if any(not t for t in terminators):
            raise ValueError("terminators must be non-empty strings")
        if min(max_completion_chars, max_answer_chars, max_number_digits) <= 0 or max_exponent < 0:
            raise ValueError("size limits must be positive")
        if max_number_digits > 4000:
            raise ValueError("max_number_digits must stay below Python's 4300-digit int conversion limit")
        self.extractors = tuple(extractors)
        self.strict_boxed = strict_boxed
        self.strict_trailing = strict_trailing
        self.terminators = tuple(terminators)
        self.grade_truncated = grade_truncated
        self.strip_percent = strip_percent
        self.strip_degrees = strip_degrees
        self.max_completion_chars = max_completion_chars
        self.max_answer_chars = max_answer_chars
        self.max_number_digits = max_number_digits
        self.max_exponent = max_exponent

    def __repr__(self) -> str:
        return (
            f"MathVerifier(extractors={self.extractors}, strict_boxed={self.strict_boxed}, "
            f"strict_trailing={self.strict_trailing}, grade_truncated={self.grade_truncated})"
        )

    # ------------------------------------------------------------------ helpers
    def normalize(self, answer: str) -> str:
        return normalize_answer(answer, strip_percent=self.strip_percent, strip_degrees=self.strip_degrees)

    def canonical(self, normalized: str) -> Fraction | str:
        value = parse_number(normalized, max_digits=self.max_number_digits, max_exponent=self.max_exponent)
        return normalized if value is None else value

    def equivalent(self, answer: str, reference: str) -> bool:
        """Equivalence of two already-normalised answers (exact rational, else exact string)."""
        a, b = self.canonical(answer), self.canonical(reference)
        if isinstance(a, Fraction) and isinstance(b, Fraction):
            return a == b
        return answer == reference

    def strip_terminators(self, text: str) -> tuple[str, bool]:
        """Cut ``text`` at the first terminator; returns ``(text, terminated)``."""
        slack = max((len(t) for t in self.terminators), default=0)
        head = text[: self.max_completion_chars + slack]
        cut = -1
        for term in self.terminators:
            idx = head.find(term)
            if idx >= 0 and (cut < 0 or idx < cut):
                cut = idx
        return (text[:cut], True) if cut >= 0 else (text, False)

    def reference_answer(self, reference: Any) -> str:
        """Normalised reference. A raw GSM8K solution (``... #### 72``) or a ``\\boxed{}`` solution is reduced
        to its final answer; ints / floats / Fractions are stringified."""
        if reference is None or isinstance(reference, bool):
            raise ValueError(f"invalid reference {reference!r}")
        ref = reference if isinstance(reference, str) else str(reference)
        if len(ref) > self.max_completion_chars:
            raise ValueError("reference longer than max_completion_chars")
        boxes = _find_boxes(ref)
        if boxes:
            ref = _unwrap_box(boxes[-1][0])
        elif "####" in ref:
            ref = _rest_of_line(ref, ref.rfind("####") + 4)[0]
        normalized = self.normalize(ref)
        if not normalized:
            raise ValueError("empty reference answer")
        return normalized

    def _extract(self, text: str) -> tuple[str, int, str] | None:
        """``(raw_answer, end_of_answer_span, extractor)`` or ``None``; raises ``_FormatError``."""
        for extractor in self.extractors:
            if extractor == "boxed":
                boxes = _find_boxes(text)
                if not boxes:
                    if self.strict_boxed:
                        raise _FormatError("no_box")
                    continue
                if self.strict_boxed:
                    keys = set()
                    for content, _ in boxes:
                        answer = self._checked_answer(_unwrap_box(self._bounded(content)))
                        keys.add(self.canonical(answer))
                    if len(keys) > 1:
                        raise _FormatError("conflicting_boxes")
                content, end = boxes[-1]
                return _unwrap_box(self._bounded(content)), end, "boxed"
            if extractor == "gsm8k":
                idx = text.rfind("####")
                if idx >= 0:
                    answer, end = _rest_of_line(text, idx + 4)
                    return answer, end, "gsm8k"
            if extractor == "answer_phrase":
                last = None
                for last in _ANSWER_PHRASE_RE.finditer(text):  # noqa: B007 - we want the last match
                    pass
                if last is not None:
                    answer, end = _rest_of_line(text, last.end())
                    return answer, end, "answer_phrase"
        return None

    def _bounded(self, raw: str) -> str:
        if len(raw) > self.max_answer_chars:
            raise _FormatError("answer_too_long")
        return raw

    def _checked_answer(self, raw: str) -> str:
        answer = self.normalize(self._bounded(raw))
        if not answer:
            raise _FormatError("empty_answer")
        return answer

    # ------------------------------------------------------------------ verify
    def _verify(
        self, completion: str | None, reference: Any, *, truncated: bool = False, **meta: Any
    ) -> tuple[Status, str | None, str | None, dict[str, Any]]:
        try:
            ref = self.reference_answer(reference)
        except (ValueError, _FormatError) as exc:
            return "error", None, None, {"reason": "bad_reference", "message": str(exc)}
        text, terminated = self.strip_terminators(completion or "")
        cut = bool(truncated) and not terminated
        details: dict[str, Any] = {"terminated": terminated, "truncated": cut}
        if cut and not self.grade_truncated:
            return "truncated", None, ref, {**details, "reason": "cut_by_length"}
        if len(text) > self.max_completion_chars:
            return "format_error", None, ref, {**details, "reason": "completion_too_long"}
        if not text.strip():
            return "no_answer", None, ref, {**details, "reason": "empty_completion"}
        try:
            found = self._extract(text)
            if found is None:
                return "no_answer", None, ref, {**details, "reason": "no_answer_found"}
            raw, end, extractor = found
            details["extractor"] = extractor
            if self.strict_trailing and not _ALLOWED_TRAILING_RE.fullmatch(text, end):
                raise _FormatError("trailing_text")
            answer = self._checked_answer(raw)
        except _FormatError as exc:
            return "format_error", None, ref, {**details, "reason": exc.reason}
        status: Status = "correct" if self.equivalent(answer, ref) else "incorrect"
        return status, answer, ref, details


VERIFIERS.register("math")(MathVerifier)
