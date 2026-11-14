"""MathVerifier: table-driven boundary tests with explicit expectations + independent Fraction reference."""

from __future__ import annotations

import random
import time
from fractions import Fraction

import pytest

from research_rl.rewards import MathVerifier, RewardScheme, normalize_answer, parse_number

# (id, completion, reference, verifier kwargs, verify meta, expected status, expected parsed answer or None)
CASES: list[tuple[str, str | None, object, dict, dict, str, str | None]] = [
    # ---- correct answers in every supported answer format
    ("boxed_int", "so \\boxed{42}", "42", {}, {}, "correct", "42"),
    ("boxed_in_dollars_period", "Thus $\\boxed{42}$.", "42", {}, {}, "correct", "42"),
    ("fbox", "\\fbox{42}", "42", {}, {}, "correct", "42"),
    ("nested_box", "\\boxed{\\boxed{42}}", "42", {}, {}, "correct", "42"),
    (
        "nested_braces",
        "\\boxed{\\frac{\\sqrt{3}}{2}}",
        "\\frac{\\sqrt{3}}{2}",
        {},
        {},
        "correct",
        "\\frac{\\sqrt{3}}{2}",
    ),
    ("escaped_open_brace", "\\boxed{\\{}", "\\{", {}, {}, "correct", "\\{"),
    ("escaped_close_brace", "\\boxed{a\\}} b}", "a\\}", {}, {}, "correct", "a\\}"),
    ("escaped_brace_in_box", "\\boxed{\\{1,2\\}}", "\\{1,2\\}", {}, {}, "correct", "\\{1,2\\}"),
    ("gsm8k", "6*7=42\n#### 42", "42", {}, {}, "correct", "42"),
    ("gsm8k_ref_solution", "#### 72", "48+24=72 clips.\n#### 72", {}, {}, "correct", "72"),
    ("boxed_ref_solution", "#### 9", "so $\\boxed{9}$", {}, {}, "correct", "9"),
    ("answer_colon", "Answer: 7", "7", {}, {}, "correct", "7"),
    ("answer_is_period", "The answer is 7.", "7", {}, {}, "correct", "7"),
    ("final_answer_is_math", "So the final answer is $12$", "12", {}, {}, "correct", "12"),
    ("int_reference", "\\boxed{3}", 3, {}, {}, "correct", "3"),
    ("float_reference", "\\boxed{0.1}", 0.1, {}, {}, "correct", "0.1"),
    ("fraction_reference", "\\boxed{0.25}", Fraction(1, 4), {}, {}, "correct", "0.25"),
    # ---- wrong answers
    ("boxed_wrong", "\\boxed{41}", "42", {}, {}, "incorrect", "41"),
    ("first_right_last_wrong", "\\boxed{42} wait, \\boxed{41}", "42", {}, {}, "incorrect", "41"),
    ("first_wrong_last_right", "\\boxed{41} wait, \\boxed{42}", "42", {}, {}, "correct", "42"),
    ("gsm8k_wrong", "#### 43", 42, {}, {}, "incorrect", "43"),
    ("no_float_rounding", "\\boxed{10000000000000001}", "10000000000000000", {}, {}, "incorrect", None),
    ("third_not_decimal", "\\boxed{0.3333333333}", "1/3", {}, {}, "incorrect", None),
    ("symbol_vs_number", "\\boxed{x}", 5, {}, {}, "incorrect", "x"),
    ("case_sensitive", "\\boxed{B}", "b", {}, {}, "incorrect", "B"),
    ("tuple_not_thousands", "\\boxed{1,2}", "12", {}, {}, "incorrect", "1,2"),
    ("interval_brackets", "\\boxed{(1,2]}", "[1,2]", {}, {}, "incorrect", "(1,2]"),
    ("symbolic_not_supported", "\\boxed{\\sqrt{8}}", "2\\sqrt{2}", {}, {}, "incorrect", "\\sqrt{8}"),
    ("box_order_beats_gsm8k", "#### 5\n\\boxed{6}", "5", {}, {}, "incorrect", "6"),
    # ---- format errors
    ("no_box_strict", "the answer is 5", "5", {"strict_boxed": True}, {}, "format_error", None),
    ("empty_box", "\\boxed{}", "5", {}, {}, "format_error", None),
    ("whitespace_box", "\\boxed{  }", "5", {}, {}, "format_error", None),
    ("empty_last_box", "\\boxed{5} \\boxed{}", "5", {}, {}, "format_error", None),
    ("unbalanced", "\\boxed{\\frac{1}{2}", "0.5", {}, {}, "format_error", None),
    ("unbalanced_after_good", "\\boxed{5} and \\boxed{6", "5", {}, {}, "format_error", None),
    ("conflicting_strict", "\\boxed{1} \\boxed{2}", "2", {"strict_boxed": True}, {}, "format_error", None),
    ("same_value_strict", "\\boxed{1} \\boxed{1.0}", "1", {"strict_boxed": True}, {}, "correct", "1.0"),
    ("empty_box_strict", "\\boxed{} \\boxed{1}", "1", {"strict_boxed": True}, {}, "format_error", None),
    ("conflicting_nonstrict", "\\boxed{1} \\boxed{2}", "2", {}, {}, "correct", "2"),
    ("gsm8k_empty", "#### \nmore", "5", {}, {}, "format_error", None),
    ("answer_phrase_empty", "The answer is:\n5", "5", {}, {}, "format_error", None),
    ("answer_too_long", "\\boxed{" + "1" * 600 + "}", "1", {}, {}, "format_error", None),
    ("dollars_only", "\\boxed{$$}", "5", {}, {}, "format_error", None),
    # ---- extra text before / after
    ("reasoning_before", "Let x=3. Then 2x-1=5. \\boxed{5}", "5", {}, {}, "correct", "5"),
    ("text_after_default", "\\boxed{5} because 2+3=5", "5", {}, {}, "correct", "5"),
    (
        "text_after_strict",
        "\\boxed{5} because 2+3=5",
        "5",
        {"strict_trailing": True},
        {},
        "format_error",
        None,
    ),
    ("closers_after_strict", "$\\boxed{5}$.\n  ", "5", {"strict_trailing": True}, {}, "correct", "5"),
    ("display_after_strict", "\\[\\boxed{5}\\]", "5", {"strict_trailing": True}, {}, "correct", "5"),
    ("gsm8k_next_line_strict", "#### 5\nok", "5", {"strict_trailing": True}, {}, "format_error", None),
    ("gsm8k_trailing_ws_strict", "#### 5\n\n", "5", {"strict_trailing": True}, {}, "correct", "5"),
    # ---- no answer
    ("empty_string", "", "5", {}, {}, "no_answer", None),
    ("none_completion", None, "5", {}, {}, "no_answer", None),
    ("whitespace_only", "  \n\t", "5", {}, {}, "no_answer", None),
    ("no_marker", "I think it is 5", "5", {}, {}, "no_answer", None),
    ("answer_word_without_is", "the answer follows 5", "5", {}, {}, "no_answer", None),
    ("boxed_no_brace_ignored", "\\boxed 5", "5", {}, {}, "no_answer", None),
    ("extractor_disabled", "#### 5", "5", {"extractors": ("boxed",)}, {}, "no_answer", None),
    # ---- terminators
    ("eos_after_answer", "\\boxed{5}</s>\\boxed{6}", "5", {}, {}, "correct", "5"),
    ("answer_only_after_eos", "thinking</s>\\boxed{5}", "5", {}, {}, "no_answer", None),
    ("wrong_after_eos_ignored", "\\boxed{5}<|im_end|> \\boxed{6}", "6", {}, {}, "incorrect", "5"),
    (
        "first_of_multiple_terminators",
        "\\boxed{5}<|eot_id|>x<|im_end|>\\boxed{7}",
        "5",
        {},
        {},
        "correct",
        "5",
    ),
    ("later_terminator_first", "\\boxed{5}<|endoftext|>\\boxed{6}</s>", "6", {}, {}, "incorrect", "5"),
    ("eos_inside_box", "\\boxed{5</s>}", "5", {}, {}, "format_error", None),
    ("strict_trailing_after_eos", "\\boxed{5}</s> junk", "5", {"strict_trailing": True}, {}, "correct", "5"),
    (
        "custom_terminator",
        "\\boxed{5}<END>\\boxed{6}",
        "6",
        {"terminators": ("<END>",)},
        {},
        "incorrect",
        "5",
    ),
    ("default_terms_off", "\\boxed{5}</s>\\boxed{6}", "6", {"terminators": ()}, {}, "correct", "6"),
    ("truncated_flag", "\\boxed{5}", "5", {}, {"truncated": True}, "truncated", None),
    ("truncated_no_answer", "abc", "5", {}, {"truncated": True}, "truncated", None),
    # a textual terminator does not clear an explicit truncated=True (TRL decodes with skip_special_tokens)
    ("truncated_textual_eos", "\\boxed{5}</s> more", "5", {}, {"truncated": True}, "truncated", None),
    (
        "truncated_eos_override_opt_in",
        "\\boxed{5}</s> more",
        "5",
        {"terminator_overrides_truncated": True},
        {"truncated": True},
        "correct",
        "5",
    ),
    ("not_truncated_flag", "\\boxed{5}", "5", {}, {"truncated": False}, "correct", "5"),
    ("grade_truncated", "\\boxed{5}", "5", {"grade_truncated": True}, {"truncated": True}, "correct", "5"),
    # ---- numeric / normalisation edge cases (boxed answer vs reference)
    ("negative_zero", "\\boxed{-0}", "0", {}, {}, "correct", "-0"),
    ("negative_zero_float", "\\boxed{-0.0}", "0", {}, {}, "correct", "-0.0"),
    ("one_point_zero", "\\boxed{1.0}", "1", {}, {}, "correct", "1.0"),
    ("half_vs_slash", "\\boxed{0.5}", "1/2", {}, {}, "correct", "0.5"),
    ("frac_vs_decimal", "\\boxed{\\frac{1}{2}}", "0.5", {}, {}, "correct", "\\frac{1}{2}"),
    ("dfrac_vs_short_frac", "\\boxed{\\dfrac{1}{2}}", "\\frac12", {}, {}, "correct", "\\frac{1}{2}"),
    ("tfrac", "\\boxed{\\tfrac{3}{6}}", "0.5", {}, {}, "correct", "\\frac{3}{6}"),
    ("neg_frac", "\\boxed{-\\frac{3}{4}}", "-0.75", {}, {}, "correct", "-\\frac{3}{4}"),
    ("frac_neg_numerator", "\\boxed{\\frac{-3}{4}}", "-0.75", {}, {}, "correct", "\\frac{-3}{4}"),
    ("neg_slash", "\\boxed{-3/4}", "-0.75", {}, {}, "correct", "-3/4"),
    ("leading_zeros", "\\boxed{007}", "7", {}, {}, "correct", "007"),
    ("leading_dot", "\\boxed{0.50}", ".5", {}, {}, "correct", "0.50"),
    ("unicode_minus", "\\boxed{\u22125}", "-5", {}, {}, "correct", "-5"),
    ("thousands", "\\boxed{1,234,567}", "1234567", {}, {}, "correct", "1234567"),
    ("thousands_decimal", "#### -1,234.50", "-1234.5", {}, {}, "correct", "-1234.50"),
    ("latex_thousands", "\\boxed{1{,}234}", "1234", {}, {}, "correct", "1234"),
    ("leading_plus", "\\boxed{+5}", "5", {}, {}, "correct", "5"),
    ("trailing_period_in_box", "\\boxed{5.}", 5, {}, {}, "correct", "5"),
    ("spaces_inside", "\\boxed{ 1 2 }", "12", {}, {}, "correct", "12"),
    (
        "big_int_exact",
        "\\boxed{12345678901234567890123456789}",
        "12345678901234567890123456789",
        {},
        {},
        "correct",
        None,
    ),
    (
        "big_int_last_digit",
        "\\boxed{12345678901234567890123456789}",
        "12345678901234567890123456788",
        {},
        {},
        "incorrect",
        None,
    ),
    ("sci_vs_int", "\\boxed{1e3}", "1000", {}, {}, "correct", "1e3"),
    ("sci_negative_exp", "\\boxed{1.5E-3}", "0.0015", {}, {}, "correct", "1.5E-3"),
    ("times_ten", "\\boxed{2.5 \\times 10^{2}}", "250", {}, {}, "correct", "2.5\\times10^{2}"),
    ("huge_exponent_bounded", "\\boxed{1e999999999}", "1", {}, {}, "incorrect", "1e999999999"),
    ("huge_exponent_string_eq", "\\boxed{1e401}", "1e401", {}, {}, "correct", "1e401"),
    ("div_by_zero_not_number", "\\boxed{1/0}", "0", {}, {}, "incorrect", "1/0"),
    ("percent_latex", "\\boxed{50\\%}", "50", {}, {}, "correct", "50"),
    ("percent_plain", "#### 50%", "50", {}, {}, "correct", "50"),
    ("percent_kept", "\\boxed{50\\%}", "50", {"strip_percent": False}, {}, "incorrect", "50\\%"),
    ("degrees_circ", "\\boxed{90^\\circ}", "90", {}, {}, "correct", "90"),
    ("degrees_braced", "\\boxed{90^{\\circ}}", "90", {}, {}, "correct", "90"),
    ("degrees_symbol", "\\boxed{90°}", "90", {}, {}, "correct", "90"),
    ("degrees_kept", "\\boxed{90^\\circ}", "90", {"strip_degrees": False}, {}, "incorrect", "90^\\circ"),
    ("text_choice", "\\boxed{\\text{(B)}}", "(B)", {}, {}, "correct", "(B)"),
    ("left_right_interval", "\\boxed{\\left(1, 2\\right]}", "(1,2]", {}, {}, "correct", "(1,2]"),
    ("rightarrow_kept", "\\boxed{a\\rightarrow b}", "a\\rightarrowb", {}, {}, "correct", "a\\rightarrowb"),
    ("expression_spacing", "\\boxed{x^2 + 1}", "x^2+1", {}, {}, "correct", "x^2+1"),
    ("unicode_superscript_not_digit", "\\boxed{2²}", "4", {}, {}, "incorrect", "2²"),
    ("boxed_space_before_brace", "\\boxed {5}", "5", {}, {}, "correct", "5"),
    ("displaystyle_short_frac", "\\boxed{\\displaystyle\\frac12}", "0.5", {}, {}, "correct", "\\frac{1}{2}"),
    ("tilde_spacing", "\\boxed{1~000}", "1000", {}, {}, "correct", "1000"),
    ("exp_at_bound", "\\boxed{1e400}", "1" + "0" * 400, {}, {}, "correct", "1e400"),
    ("exp_at_bound_wrong", "\\boxed{1e400}", "1" + "0" * 399, {}, {}, "incorrect", "1e400"),
    # ---- null delimiters \left. / \right. leave nothing behind (no "5" -> ".5" false positive)
    ("null_delims_not_half", "\\boxed{\\left. 5 \\right.}", "0.5", {}, {}, "incorrect", "5"),
    ("null_delims_value", "\\boxed{\\left. 5 \\right.}", "5", {}, {}, "correct", "5"),
    ("null_delim_with_bar", "\\boxed{\\left.x\\right|_0}", "x|_0", {}, {}, "correct", "x|_0"),
    # ---- escaped dollar (currency) on the answer and reference side
    ("escaped_dollar_boxed", "\\boxed{\\$18}", "18", {}, {}, "correct", "18"),
    ("escaped_dollar_gsm8k", "#### \\$18", "18", {}, {}, "correct", "18"),
    ("escaped_dollar_reference", "\\boxed{5}", "\\$5", {}, {}, "correct", "5"),
    ("escaped_dollar_thousands", "\\boxed{\\$1,234}", "1234", {}, {}, "correct", "1234"),
    ("escaped_dollar_space_in_math", "The answer is $\\$ 18$ total", "18", {}, {}, "correct", "18"),
    ("escaped_dollar_in_math", "The answer is $\\$18$ total", "18", {}, {}, "correct", "18"),
    # ---- line-based extractors: bounded answer span, same-line reasoning text is trailing text
    ("phrase_then_sentence", "The answer is 42. Let me double check.", "42", {}, {}, "correct", "42"),
    ("phrase_math_then_unit", "The answer is $7$ dollars.", "7", {}, {}, "correct", "7"),
    ("gsm8k_then_unit", "#### 5 apples", "5", {}, {}, "correct", "5"),
    ("phrase_comma_reasoning", "The answer is 5, as shown above", "5", {}, {}, "correct", "5"),
    ("markdown_answer_colon", "**Answer:** 5", "5", {}, {}, "correct", "5"),
    ("markdown_final_answer", "**Final Answer**: 5", "5", {}, {}, "correct", "5"),
    ("markdown_bold_value", "The answer is **5**", "5", {}, {}, "correct", "5"),
    ("markdown_bold_value_period", "The answer is **5**.", "5", {}, {}, "correct", "5"),
    ("phrase_period_then_math", "The answer is 5. 2+3=5", "5", {}, {}, "correct", "5"),
    ("phrase_comma_then_number", "The answer is 5, 6 is wrong", "5", {}, {}, "correct", "5"),
    ("phrase_two_currency_signs", "The answer is $18, since $20 - $2 = $18.", "18", {}, {}, "correct", "18"),
    ("gsm8k_two_currency_signs", "#### $18 (from $20)", "18", {}, {}, "correct", "18"),
    ("phrase_unit_dropped_limitation", "The answer is 5 cm", "5 cm", {}, {}, "incorrect", "5"),
    ("phrase_paren_reasoning", "The answer is 5 (since 2+3=5)", "5", {}, {}, "correct", "5"),
    ("phrase_currency_unclosed", "The answer is $18. Done", "18", {}, {}, "correct", "18"),
    ("phrase_decimal_period", "The answer is 3.14.", "3.14", {}, {}, "correct", "3.14"),
    ("phrase_expression", "Answer: x^2 + 1", "x^2+1", {}, {}, "correct", "x^2+1"),
    ("phrase_choice", "The answer is (B)", "(B)", {}, {}, "correct", "(B)"),
    ("phrase_interval", "The answer is (1, 2] here", "(1,2]", {}, {}, "correct", "(1,2]"),
    ("phrase_frac", "Answer: \\frac{1}{2}", "0.5", {}, {}, "correct", "\\frac{1}{2}"),
    ("phrase_paren_math", "Answer: \\(\\frac{1}{2}\\) cups", "0.5", {}, {}, "correct", "\\frac{1}{2}"),
    ("phrase_word_answer", "The answer is yes because 2>1", "yes", {}, {}, "correct", "yes"),
    ("phrase_wrong_then_text", "The answer is 41. Close to 42", "42", {}, {}, "incorrect", "41"),
    ("gsm8k_last_marker_wins", "#### 5\n#### 6", "6", {}, {}, "correct", "6"),
    ("gsm8k_last_marker_wrong", "#### 6\n#### 5", "6", {}, {}, "incorrect", "5"),
    ("phrase_last_match_wins", "Answer: 5 ... The answer is 6", "6", {}, {}, "correct", "6"),
    ("phrase_last_match_wrong", "Answer: 6 ... The answer is 5", "6", {}, {}, "incorrect", "5"),
    # ---- strict_trailing applies to line-based extractors too
    ("gsm8k_unit_strict", "#### 5 apples", "5", {"strict_trailing": True}, {}, "format_error", None),
    (
        "phrase_because_strict",
        "The answer is 5 because x",
        "5",
        {"strict_trailing": True},
        {},
        "format_error",
        None,
    ),
    ("phrase_period_strict", "The answer is **5**.", "5", {"strict_trailing": True}, {}, "correct", "5"),
    ("phrase_math_strict", "The answer is $$7$$.\n", "7", {"strict_trailing": True}, {}, "correct", "7"),
    # ---- reference extraction: last box / last #### / bounded #### span
    ("ref_last_box", "\\boxed{6}", "\\boxed{5} or \\boxed{6}", {}, {}, "correct", "6"),
    ("ref_last_box_wrong", "\\boxed{5}", "\\boxed{5} or \\boxed{6}", {}, {}, "incorrect", "5"),
    ("ref_last_hashes", "\\boxed{6}", "#### 5\n#### 6", {}, {}, "correct", "6"),
    ("ref_last_hashes_wrong", "\\boxed{5}", "#### 5\n#### 6", {}, {}, "incorrect", "5"),
    ("ref_hashes_with_unit", "\\boxed{5}", "so 5 apples\n#### 5 apples", {}, {}, "correct", "5"),
    ("ref_hashes_expression", "\\boxed{x^2+1}", "#### x^2 + 1", {}, {}, "correct", "x^2+1"),
]


@pytest.mark.parametrize(
    ("completion", "reference", "cfg", "meta", "status", "parsed"),
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_math_verifier_table(completion, reference, cfg, meta, status, parsed):
    result = MathVerifier(**cfg).verify(completion, reference, **meta)
    assert result.status == status, result
    assert result.correct == (status == "correct")
    assert result.reward == (1.0 if status == "correct" else 0.0)
    if parsed is not None:
        assert result.parsed_answer == parsed
    if status in ("format_error", "no_answer", "truncated"):
        assert result.parsed_answer is None


def test_case_ids_unique():
    ids = [c[0] for c in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("completion", "cfg", "reason"),
    [
        ("the answer is 5", {"strict_boxed": True}, "no_box"),
        ("\\boxed{}", {}, "empty_answer"),
        ("\\boxed{\\frac{1}{2}", {}, "unbalanced_braces"),
        ("\\boxed{1} \\boxed{2}", {"strict_boxed": True}, "conflicting_boxes"),
        ("\\boxed{5} because", {"strict_trailing": True}, "trailing_text"),
        ("\\boxed{" + "1" * 600 + "}", {}, "answer_too_long"),
        ("a" * 40000 + "\\boxed{5}", {}, "completion_too_long"),
        ("\\boxed{" * 20 + "5" + "}" * 20, {}, "nested_boxes_too_deep"),
        ("#### 5 apples", {"strict_trailing": True}, "trailing_text"),
        ("Answer: " + "1 " * 400, {}, "answer_too_long"),
    ],
    ids=lambda v: v if isinstance(v, str) and len(v) < 40 else "",
)
def test_format_error_reasons(completion, cfg, reason):
    result = MathVerifier(**cfg).verify(completion, "5")
    assert (result.status, result.details["reason"]) == ("format_error", reason)


def test_huge_integers_beyond_digit_limit_compare_as_strings():
    nines = "9" * 2000  # > max_number_digits (1000): exact string comparison, no int conversion
    v = MathVerifier(max_answer_chars=4096)
    assert MathVerifier().verify(f"\\boxed{{{nines}}}", "9").details["reason"] == "answer_too_long"
    assert MathVerifier().verify("\\boxed{9}", nines).details["reason"] == "bad_reference"
    assert v.verify(f"\\boxed{{{nines}}}", nines).status == "correct"
    assert v.verify(f"\\boxed{{{nines[:-1]}8}}", nines).status == "incorrect"
    within = "9" * 900
    assert v.verify(f"\\boxed{{{within}.0}}", within).status == "correct"  # exact rational path
    assert parse_number(nines) is None and parse_number(within) == Fraction(int(within))


@pytest.mark.parametrize(
    ("reference", "reason"),
    [
        (None, "bad_reference"),
        ("", "bad_reference"),
        ("   $ $ ", "bad_reference"),
        (True, "bad_reference"),
        ("1" * 600, "bad_reference"),
    ],
)
def test_bad_reference_is_error(reference, reason):
    result = MathVerifier().verify("\\boxed{5}", reference)
    assert (result.status, result.details["reason"], result.reward) == ("error", reason, 0.0)


def test_reward_scheme_is_applied():
    scheme = RewardScheme(correct=2.0, incorrect=-1.0, format_error=-0.5, no_answer=-0.25, truncated=-2.0)
    v = MathVerifier(rewards=scheme)
    assert v.verify("\\boxed{5}", "5").reward == 2.0
    assert v.verify("\\boxed{6}", "5").reward == -1.0
    assert v.verify("\\boxed{}", "5").reward == -0.5
    assert v.verify("nothing", "5").reward == -0.25
    assert v.verify("\\boxed{5}", "5", truncated=True).reward == -2.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$\\boxed{5}$", "\\boxed{5}"),
        ("$$ 5 $$", "5"),
        ("\\( \\dfrac{1}{2} \\)", "\\frac{1}{2}"),
        ("\\[x\\]", "x"),
        ("\\left[ 1, 2 \\right)", "[1,2)"),
        ("\\text{ cm }", "cm"),
        ("\\mathrm{\\text{a}}b", "ab"),
        ("1\\,000", "1000"),
        ("12{,}345{,}678", "12345678"),
        ("\u2212\u20093", "-3"),
        ("+.5.", ".5"),
        ("45^\\circ", "45"),
        ("10\\%", "10"),
        ("\\frac34", "\\frac{3}{4}"),
        ("\\displaystyle\\frac12", "\\frac{1}{2}"),
        ("1~000", "1000"),
        ("\\$1,234", "1234"),
        ("$\\$5$", "5"),
        ("\\left. 5 \\right.", "5"),
        ("\\left.\\frac{1}{2}\\right|", "\\frac{1}{2}|"),
        ("\\left\\{1\\right\\}", "\\{1\\}"),
        ("1,23", "1,23"),
    ],
)
def test_normalize_answer_table(raw, expected):
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", Fraction(0)),
        ("-0", Fraction(0)),
        ("007", Fraction(7)),
        ("1.250", Fraction(5, 4)),
        (".5", Fraction(1, 2)),
        ("5.", Fraction(5)),
        ("-2.5e-2", Fraction(-1, 40)),
        ("3E+2", Fraction(300)),
        ("6/8", Fraction(3, 4)),
        ("-6/8", Fraction(-3, 4)),
        ("1/-2", Fraction(-1, 2)),
        ("\\frac{2}{-4}", Fraction(-1, 2)),
        ("-\\frac{0.5}{2}", Fraction(-1, 4)),
        ("1.2\\times10^{-1}", Fraction(3, 25)),
        ("4\\cdot10^3", Fraction(4000)),
        ("1e400", Fraction(10**400)),
        ("1e-400", Fraction(1, 10**400)),
        ("1\\times10^{400}", Fraction(10**400)),
        ("1\\times10^{401}", None),
        ("1\\times10^{999999999}", None),
        ("1/0", None),
        ("\\frac{1}{0}", None),
        (".", None),
        ("", None),
        ("--5", None),
        ("1e", None),
        ("1e401", None),
        ("0x10", None),
        ("１２", None),  # full-width digits are not ASCII digits
        ("1/2/3", None),
        ("\\frac{a}{2}", None),
    ],
)
def test_parse_number_table(text, expected):
    assert parse_number(text) == expected


def _decimal_string(value: Fraction) -> str | None:
    """Exact terminating decimal of ``value`` (independent of the implementation) or None."""
    den = value.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den != 1:
        return None
    places = max(twos, fives)
    scaled = abs(value.numerator) * (10**places) // value.denominator
    digits = str(scaled).rjust(places + 1, "0")
    body = digits if places == 0 else f"{digits[:-places]}.{digits[-places:]}"
    return ("-" if value < 0 else "") + body


def _renderings(value: Fraction, rng: random.Random) -> list[str]:
    k = rng.randint(1, 4)
    a, b = value.numerator * k, value.denominator * k  # non-reduced on purpose
    out = [f"{a}/{b}", f"\\frac{{{a}}}{{{b}}}", f"\\dfrac{{{a}}}{{{b}}}"]
    dec = _decimal_string(value)
    if dec is not None:
        out.append(dec)
        e = rng.randint(1, 3)
        mant = _decimal_string(value * 10**e)
        assert mant is not None
        out.append(f"{mant}e-{e}")
        out.append(f"{mant}\\times10^{{-{e}}}")
    return out


def test_seeded_equivalence_against_fraction_reference():
    rng = random.Random(1234)
    checked = {True: 0, False: 0}
    v = MathVerifier()
    for _ in range(300):
        den = rng.choice([1, 2, 3, 4, 5, 8, 10, 16, 20, 25, 7, 12])
        x = Fraction(rng.randint(-50, 50), den)
        y = x if rng.random() < 0.5 else x + Fraction(rng.choice([1, -1]), rng.choice([1, 2, 1000, 3]))
        ref = rng.choice(_renderings(y, rng))
        ans = rng.choice(_renderings(x, rng))
        expected = x == y
        result = v.verify(f"so \\boxed{{{ans}}}", ref)
        assert result.correct is expected, (ans, ref, result)
        checked[expected] += 1
    assert checked[True] > 50 and checked[False] > 50


@pytest.mark.parametrize(
    ("completion", "cfg"),
    [
        ("\\boxed{" * 4000, {}),
        ("{" * 32000, {}),
        ("\\boxed{" + "{" * 32000, {}),
        ("answer is " * 3200, {}),
        ("####" * 8000, {}),
        ("Answer:" + " " * 32000, {}),
        ("\\boxed{1}" * 3500, {"strict_boxed": True}),
        ("\\boxed{" + "\\text{" * 80 + "x" + "}" * 81, {}),
        ("\\boxed{5}" + "\\boxed{5}" * 3000 + "z" * 4000, {"strict_trailing": True}),
        ("\\boxed{5}</s>" + "a" * 2_000_000, {}),
        ("a" * 2_000_000, {}),
        ("1" * 30000 + "e" + "9" * 20, {"extractors": ("answer_phrase",)}),
        ("\\boxed{1\\times10^{999999999}}", {}),
        ("The answer is " + "(" * 30000, {}),
        ("#### " + "1 " * 15000, {"strict_trailing": True}),
        ("Answer: $" + "\\$" * 15000, {}),
        ("answer" + " " * 32000 + "x", {}),
        ("final answer" + " " * 32000 + "**" + " " * 100, {}),
    ],
    ids=[
        "open_boxes",
        "open_braces",
        "box_then_braces",
        "answer_is_repeat",
        "hash_repeat",
        "answer_spaces",
        "strict_many_boxes",
        "nested_text",
        "strict_trailing_many",
        "tail_after_eos",
        "long_no_eos",
        "long_number",
        "times_ten_huge_exponent",
        "phrase_open_parens",
        "hash_many_tokens",
        "phrase_unclosed_dollar_escapes",
        "answer_long_space_run",
        "answer_spaces_then_bold",
    ],
)
def test_pathological_inputs_are_fast(completion, cfg):
    v = MathVerifier(**cfg)
    start = time.perf_counter()
    result = v.verify(completion, "5")
    assert time.perf_counter() - start < 2.0, result  # ~30 ms measured; loose for shared CI
    assert result.status in ("correct", "incorrect", "format_error", "no_answer")


def test_long_tail_after_terminator_is_ignored():
    result = MathVerifier().verify("\\boxed{5}</s>" + "a" * 2_000_000, "5")
    assert result.status == "correct"


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        MathVerifier(extractors=("boxed", "eval"))
    with pytest.raises(ValueError):
        MathVerifier(extractors=("gsm8k",), strict_boxed=True)
    with pytest.raises(ValueError):
        MathVerifier(max_number_digits=5000)
    with pytest.raises(ValueError):
        MathVerifier(terminators=("",))


def test_answer_phrase_regex_is_linear_on_long_space_runs():
    # an ambiguous "[ \t]*(**)?[ \t]*:" regex was quadratic here (~1.4 s for 32k spaces); now ~2 ms
    v = MathVerifier()
    start = time.perf_counter()
    result = v.verify("answer" + " " * 32000 + "x", "5")
    assert time.perf_counter() - start < 0.3 and result.status == "no_answer"
