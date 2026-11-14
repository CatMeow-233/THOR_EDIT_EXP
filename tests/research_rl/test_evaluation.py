"""Evaluation metrics and capability retention, checked against independent references."""

from __future__ import annotations

import copy
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from research_rl.core.config import DataSpec, ExperimentConfig, ModelSpec, TrainerSpec
from research_rl.core.experiment import ExperimentRecorder
from research_rl.evaluation import (
    CapabilityRetention,
    ProbeTask,
    RetentionReport,
    RetentionSuite,
    exact_match,
    exact_match_accuracy,
    greedy_generate,
    logit_kl,
    mean_pass_at_k,
    mean_with_bootstrap_ci,
    pass_at_k,
    prefix_match_scorer,
    sequence_nll,
)
from research_rl.models.tiny import build_tiny_llama


# ---------------------------------------------------------------------------------- helpers
def sharp_model(tokenizer, seed: int = 0, scale: float = 30.0):
    """Tiny Llama whose logits are scaled up: a fresh init is nearly uniform (margins ~0.01),
    which would make greedy/KL/NLL tests insensitive. Scaling lm_head makes top-1 margins O(1)."""
    model = build_tiny_llama(len(tokenizer), seed=seed)
    with torch.no_grad():
        model.lm_head.weight.mul_(scale)
    return model.eval()


def perturbed(model, seed: int = 123, scale: float = 0.5):
    out = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in out.parameters():
            p.add_(scale * p.abs().mean() * torch.randn(p.shape, generator=g))
    return out


def ids_of(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def manual_greedy(model, tokenizer, prompt: str, max_new_tokens: int, eos_id: int | None) -> str:
    """Reference greedy decoding: full forward pass per step (no KV cache), plain argmax."""
    ids = ids_of(tokenizer, prompt)
    new: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=torch.tensor([ids + new])).logits[0, -1]
        tok = int(torch.argmax(logits))
        if tok == eos_id:
            break
        new.append(tok)
    return tokenizer.decode(new, skip_special_tokens=True)


@torch.no_grad()
def manual_nll(model, tokenizer, prompt: str, completion: str, completion_only: bool, bos: bool = False):
    """Reference NLL with explicit per-position loops in float64."""
    p = ids_of(tokenizer, prompt)
    c = ids_of(tokenizer, completion)
    if bos:
        p = [tokenizer.bos_token_id, *p]
    ids = p + c
    logits = model(input_ids=torch.tensor([ids])).logits[0].double()
    total, n = 0.0, 0
    first = len(p) if completion_only else 1
    for t in range(max(first, 1), len(ids)):
        row = logits[t - 1]
        logz = math.log(sum(math.exp(float(v)) for v in row))
        total += logz - float(row[ids[t]])
        n += 1
    return total, n


@torch.no_grad()
def manual_kl_sum(base, cur, tokenizer, prompt: str) -> tuple[float, int]:
    ids = torch.tensor([ids_of(tokenizer, prompt)])
    pb = torch.softmax(base(input_ids=ids).logits[0].double(), -1)
    pc = torch.softmax(cur(input_ids=ids).logits[0].double(), -1)
    total = 0.0
    for t in range(ids.shape[1]):
        for v in range(pb.shape[1]):
            total += float(pb[t, v]) * (math.log(float(pb[t, v])) - math.log(float(pc[t, v])))
    return total, ids.shape[1]


def recorder(tmp_path) -> ExperimentRecorder:
    cfg = ExperimentConfig(
        study_id="retention",
        model=ModelSpec("tiny"),
        data=DataSpec("probes"),
        trainer=TrainerSpec("grpo"),
        output_dir=str(tmp_path),
    )
    return ExperimentRecorder(cfg)


def test_sharp_model_has_clear_margins(tokenizer):
    model = sharp_model(tokenizer)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids_of(tokenizer, "12+3=45 hello")])).logits[0]
    top2 = logits.topk(2).values
    assert float((top2[:, 0] - top2[:, 1]).min()) > 0.05


# ---------------------------------------------------------------------------------- exact match
def test_exact_match_and_accuracy():
    assert exact_match(" 42\n", "42")
    assert exact_match("a   b", "a b")
    assert not exact_match(" 42", "42", normalize=False)
    assert not exact_match("Yes", "yes")
    assert exact_match("Yes", "yes", case_sensitive=False)
    acc = exact_match_accuracy(["1", " 2 ", "3", "x"], ["1", "2", "4", "X"])
    assert acc == 0.5
    assert exact_match_accuracy(["1", " 2 ", "3", "x"], ["1", "2", "4", "X"], case_sensitive=False) == 0.75
    with pytest.raises(ValueError):
        exact_match_accuracy(["1"], ["1", "2"])
    with pytest.raises(ValueError):
        exact_match_accuracy([], [])


# ---------------------------------------------------------------------------------- pass@k
def brute_force_pass_at_k(n: int, c: int, k: int) -> Fraction:
    correct = set(range(c))
    subsets = list(itertools.combinations(range(n), k))
    hits = sum(1 for s in subsets if correct.intersection(s))
    return Fraction(hits, len(subsets))


def test_pass_at_k_matches_brute_force_enumeration():
    checked = 0
    for n in range(1, 9):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert pass_at_k(n, c, k) == pytest.approx(float(brute_force_pass_at_k(n, c, k)), abs=1e-12)
                checked += 1
    assert checked == sum((n + 1) * n for n in range(1, 9))


@pytest.mark.parametrize(("n", "c", "k"), [(200, 7, 50), (500, 1, 1), (500, 13, 100), (1000, 0, 10)])
def test_pass_at_k_large_n_matches_exact_binomial(n, c, k):
    exact = 1 - Fraction(math.comb(n - c, k), math.comb(n, k))
    assert pass_at_k(n, c, k) == pytest.approx(float(exact), rel=1e-12, abs=1e-15)


def test_pass_at_k_special_values_and_errors():
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)  # pass@1 = c/n
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 6, 5) == 1.0  # n - c < k
    for bad in [(0, 0, 1), (5, 6, 1), (5, -1, 1), (5, 2, 0), (5, 2, 6)]:
        with pytest.raises(ValueError):
            pass_at_k(*bad)
    assert mean_pass_at_k([4, 4], [1, 0], 2) == pytest.approx((1 - 3 / 6 + 0.0) / 2)


# ---------------------------------------------------------------------------------- bootstrap
def test_bootstrap_ci_follows_documented_procedure():
    rng = np.random.default_rng(7)
    x = rng.normal(size=37)
    res = mean_with_bootstrap_ci(x.tolist(), confidence=0.9, n_resamples=500, seed=3)
    ref_rng = np.random.default_rng(3)
    idx = ref_rng.integers(0, 37, size=(500, 37))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.05, 0.95])
    assert res.mean == pytest.approx(float(x.mean()), abs=1e-15)
    assert (res.low, res.high) == (pytest.approx(lo, abs=1e-15), pytest.approx(hi, abs=1e-15))
    assert res.n == 37 and res.to_dict()["ci_low"] == res.low


def test_bootstrap_ci_matches_normal_theory_on_large_sample():
    """Independent check: for n=2000 normal samples the 95% percentile-bootstrap half-width
    is close to 1.96 * s / sqrt(n)."""
    x = np.random.default_rng(11).normal(loc=2.0, scale=3.0, size=2000)
    res = mean_with_bootstrap_ci(x, confidence=0.95, n_resamples=4000, seed=0)
    expected = 1.959964 * x.std(ddof=1) / math.sqrt(x.size)
    assert (res.high - res.low) / 2 == pytest.approx(expected, rel=0.08)
    assert res.low < x.mean() < res.high


def test_bootstrap_ci_determinism_and_edge_cases():
    vals = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    a = mean_with_bootstrap_ci(vals, seed=5)
    assert a == mean_with_bootstrap_ci(vals.tolist(), seed=5)
    assert (a.low, a.high) != (
        mean_with_bootstrap_ci(vals, seed=6).low,
        mean_with_bootstrap_ci(vals, seed=6).high,
    )
    assert a.mean == 0.625
    wide = mean_with_bootstrap_ci(vals, confidence=0.99, seed=5)
    narrow = mean_with_bootstrap_ci(vals, confidence=0.5, seed=5)
    assert (
        wide.low <= narrow.low
        and wide.high >= narrow.high
        and narrow.high - narrow.low < wide.high - wide.low
    )
    const = mean_with_bootstrap_ci([0.25] * 10)
    assert (const.mean, const.low, const.high) == (0.25, 0.25, 0.25)
    single = mean_with_bootstrap_ci([3.0])
    assert (single.low, single.high) == (3.0, 3.0)
    for bad in ([], [float("nan")]):
        with pytest.raises(ValueError):
            mean_with_bootstrap_ci(bad)
    with pytest.raises(ValueError):
        mean_with_bootstrap_ci([1.0], confidence=1.0)


# ---------------------------------------------------------------------------------- token NLL
PAIRS = [("12+3=", "15"), ("abc", "defgh"), ("x", "y"), ("hello wor", "ld!")]


@pytest.mark.parametrize("completion_only", [True, False])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_sequence_nll_matches_manual_cross_entropy(tokenizer, completion_only, batch_size):
    model = sharp_model(tokenizer)
    model.train()
    res = sequence_nll(model, tokenizer, PAIRS, completion_only=completion_only, batch_size=batch_size)
    assert model.training  # mode restored
    refs = [manual_nll(model.eval(), tokenizer, p, c, completion_only) for p, c in PAIRS]
    for got_nll, got_n, (ref_nll, ref_n) in zip(res.per_example_nll, res.per_example_tokens, refs):
        assert got_n == ref_n
        assert got_nll == pytest.approx(ref_nll, rel=1e-5, abs=1e-6)
    total = sum(r[0] for r in refs)
    count = sum(r[1] for r in refs)
    expected_counts = [2, 5, 1, 3] if completion_only else [6, 7, 1, 11]
    assert res.per_example_tokens == expected_counts
    assert res.num_tokens == count
    assert res.nll_sum == pytest.approx(total, rel=1e-6)
    assert res.mean_nll == pytest.approx(total / count, rel=1e-6)  # micro average over tokens
    assert res.perplexity == pytest.approx(math.exp(total / count), rel=1e-6)
    macro = sum(r[0] / r[1] for r in refs) / len(refs)
    assert abs(res.mean_nll - macro) > 1e-3  # the test distinguishes micro from macro averaging


def test_sequence_nll_add_bos_scores_first_completion_token_of_empty_prompt(tokenizer):
    model = sharp_model(tokenizer)
    res = sequence_nll(model, tokenizer, [("", "42"), ("7", "x")], add_bos=True)
    refs = [
        manual_nll(model, tokenizer, "", "42", True, bos=True),
        manual_nll(model, tokenizer, "7", "x", True, bos=True),
    ]
    assert res.per_example_tokens == [2, 1]
    assert res.per_example_nll == pytest.approx([r[0] for r in refs], rel=1e-5)
    no_bos = sequence_nll(model, tokenizer, [("7", "x")])
    assert no_bos.nll_sum != pytest.approx(res.per_example_nll[1], rel=1e-6)  # BOS changes the context
    with pytest.raises(ValueError):
        sequence_nll(model, tokenizer, [("", "4")])  # single token: nothing predicts it
    with pytest.raises(ValueError):
        sequence_nll(model, tokenizer, [])


# ---------------------------------------------------------------------------------- greedy generation
PROMPTS = ["12+3=", "abc", "hello", "x*y=(", "7"]


def test_greedy_generate_matches_manual_argmax_loop(tokenizer):
    model = sharp_model(tokenizer)
    model.train()
    got = greedy_generate(model, tokenizer, PROMPTS, max_new_tokens=6)
    assert model.training
    ref = [manual_greedy(model.eval(), tokenizer, p, 6, tokenizer.eos_token_id) for p in PROMPTS]
    assert got == ref
    assert all(len(r) == 6 for r in ref)  # random model never emits EOS: bounded by max_new_tokens
    assert len(set(got)) > 1


def test_greedy_generate_batched_equals_unbatched(tokenizer):
    model = sharp_model(tokenizer)
    assert greedy_generate(model, tokenizer, PROMPTS, max_new_tokens=6, batch_size=3) == greedy_generate(
        model, tokenizer, PROMPTS, max_new_tokens=6, batch_size=1
    )


def test_greedy_generate_ignores_sampling_settings_in_generation_config(tokenizer):
    model = sharp_model(tokenizer)
    ref = [manual_greedy(model, tokenizer, p, 8, tokenizer.eos_token_id) for p in PROMPTS]
    model.generation_config.do_sample = True
    model.generation_config.temperature = 5.0
    model.generation_config.repetition_penalty = 3.0
    torch.manual_seed(0)
    assert greedy_generate(model, tokenizer, PROMPTS, max_new_tokens=8) == ref


def _first_token_id(tokenizer, text: str) -> int:
    return ids_of(tokenizer, text[0])[0]


# Each entry sets one logits-processor / stopping knob on the model's generation_config. Values are
# chosen so that HF generate *would* change the output if the knob leaked into greedy_generate.
GEN_CONFIG_KNOBS = {
    "no_repeat_ngram_size": lambda tok, ref: 1,  # every reference below repeats a character
    "suppress_tokens": lambda tok, ref: [_first_token_id(tok, ref[0])],
    "begin_suppress_tokens": lambda tok, ref: [_first_token_id(tok, ref[0])],
    "bad_words_ids": lambda tok, ref: [[_first_token_id(tok, ref[0])]],
    "sequence_bias": lambda tok, ref: {(_first_token_id(tok, ref[0]),): -1e4},
    "repetition_penalty": lambda tok, ref: 50.0,
    "stop_strings": lambda tok, ref: [ref[0][1]],
}


@pytest.mark.parametrize("knob", sorted(GEN_CONFIG_KNOBS))
def test_greedy_generate_ignores_logits_processors_in_generation_config(tokenizer, knob):
    model = sharp_model(tokenizer)
    ref = [manual_greedy(model, tokenizer, p, 8, tokenizer.eos_token_id) for p in PROMPTS]
    assert all(len(set(r)) < len(r) for r in ref[:3])  # repeats: no_repeat_ngram_size would bite
    value = GEN_CONFIG_KNOBS[knob](tokenizer, ref)
    original = model.generation_config
    setattr(original, knob, value)
    before = original.to_dict()
    if knob == "stop_strings":
        leaked = model.generate(
            input_ids=torch.tensor([ids_of(tokenizer, PROMPTS[0])]),
            max_new_tokens=8,
            tokenizer=tokenizer,
            pad_token_id=0,
        )
    else:
        leaked = model.generate(input_ids=torch.tensor([ids_of(tokenizer, PROMPTS[0])]), max_new_tokens=8)
    leaked_text = tokenizer.decode(leaked[0, len(PROMPTS[0]) :].tolist())
    assert leaked_text != ref[0]  # sanity: plain HF generate does honour the knob
    assert greedy_generate(model, tokenizer, PROMPTS, max_new_tokens=8) == ref
    assert greedy_generate(model, tokenizer, PROMPTS, max_new_tokens=8, batch_size=3) == ref
    assert model.generation_config is original and original.to_dict() == before  # restored, unchanged


@pytest.mark.parametrize(("knob", "value"), [("min_new_tokens", 8), ("min_length", 20)])
def test_greedy_generate_ignores_min_length_in_generation_config(tokenizer, knob, value):
    model = sharp_model(tokenizer)
    full = manual_greedy(model, tokenizer, "12+3=", 8, None)
    pos = next(i for i in range(2, len(full)) if full[i] not in full[:i])
    tok = copy.deepcopy(tokenizer)
    tok.eos_token = full[pos]  # the greedy path emits this EOS at step `pos`
    setattr(model.generation_config, knob, value)  # would forbid EOS until 8 new tokens
    assert greedy_generate(model, tok, ["12+3="], max_new_tokens=8) == [full[:pos]]


def test_greedy_generate_through_peft_wrapper_ignores_inner_generation_config(tokenizer):
    peft = pytest.importorskip("peft")
    model = sharp_model(tokenizer)
    ref = [manual_greedy(model, tokenizer, p, 8, tokenizer.eos_token_id) for p in PROMPTS]
    model.generation_config.suppress_tokens = [_first_token_id(tokenizer, ref[0])]
    model.generation_config.no_repeat_ngram_size = 1
    inner_config = model.generation_config
    # LoRA B is zero-initialised, so the wrapped model computes the same function as ``model``
    wrapped = peft.get_peft_model(model, peft.LoraConfig(r=2, target_modules=["q_proj"], lora_alpha=4))
    assert greedy_generate(wrapped, tokenizer, PROMPTS, max_new_tokens=8) == ref
    assert model.generation_config is inner_config and inner_config.no_repeat_ngram_size == 1


def test_greedy_generate_stops_at_tokenizer_eos(tokenizer):
    model = sharp_model(tokenizer)
    full = manual_greedy(model, tokenizer, "12+3=", 8, None)
    # pick a character first emitted at position >= 2 and declare it EOS on a tokenizer copy
    pos = next(i for i in range(2, len(full)) if full[i] not in full[:i])
    tok = copy.deepcopy(tokenizer)
    tok.eos_token = full[pos]
    assert tok.eos_token_id != tokenizer.eos_token_id
    got = greedy_generate(model, tok, ["12+3="], max_new_tokens=8)
    assert got == [full[:pos]]
    assert tokenizer.eos_token_id == 2  # caller's tokenizer untouched


# ---------------------------------------------------------------------------------- suite scores
def test_suite_scores_match_reference_predictions(tokenizer):
    model = sharp_model(tokenizer)
    preds = {p: manual_greedy(model, tokenizer, p, 5, tokenizer.eos_token_id) for p in PROMPTS}
    task_a = ProbeTask(
        "a", ((PROMPTS[0], preds[PROMPTS[0]]), (PROMPTS[1], "nope"), (PROMPTS[2], preds[PROMPTS[2]]))
    )
    task_b = ProbeTask(
        "b",
        ((PROMPTS[3], preds[PROMPTS[3]][:2]), (PROMPTS[4], "zz")),
        scorer=prefix_match_scorer,
    )
    task_c = ProbeTask("c", ((PROMPTS[3], preds[PROMPTS[3]][:2]),), max_new_tokens=2)
    res = RetentionSuite([task_a, task_b, task_c], max_new_tokens=5).evaluate(model, tokenizer)
    assert res.scores == {"a": pytest.approx(2 / 3), "b": 0.5, "c": 1.0}
    assert res.per_example_scores["a"] == [1.0, 0.0, 1.0]
    assert res.predictions["a"] == [preds[p] for p in PROMPTS[:3]]
    assert res.predictions["c"] == [preds[PROMPTS[3]][:2]]


def test_probe_task_and_suite_validation():
    with pytest.raises(ValueError):
        ProbeTask("t", ())
    with pytest.raises(ValueError):
        ProbeTask("", (("p", "a"),))
    with pytest.raises(ValueError):
        ProbeTask("t", (("", "a"),))
    t = ProbeTask("t", [("p", "a")])  # type: ignore[arg-type]
    assert t.examples == (("p", "a"),)
    with pytest.raises(ValueError):
        RetentionSuite([t, t])
    with pytest.raises(ValueError):
        RetentionSuite([])
    with pytest.raises(ValueError, match="reserved"):  # would collide with the logit_kl aggregate
        RetentionSuite([ProbeTask("all", (("p", "a"),)), t])


def test_prefix_match_scorer_normalises_whitespace():
    assert prefix_match_scorer(" 12  3x", "12 3") == 1.0  # raw " 12  3x".startswith("12 3") is False
    assert prefix_match_scorer("12\n3 rest", " 12 3 ") == 1.0
    assert prefix_match_scorer("12 4", "12 3") == 0.0
    assert prefix_match_scorer("1", "12") == 0.0


# ---------------------------------------------------------------------------------- logit KL
KL_PROMPTS = ["12+3=", "hello world", "ab"]


def test_logit_kl_identical_models_is_exactly_zero(tokenizer):
    model = sharp_model(tokenizer)
    assert logit_kl(model, model, tokenizer, KL_PROMPTS).mean_kl == 0.0
    clone = copy.deepcopy(model)
    res = logit_kl(model, clone, tokenizer, KL_PROMPTS)
    assert res.mean_kl == 0.0 and res.per_prompt_kl == (0.0, 0.0, 0.0)
    assert res.num_tokens == sum(len(p) for p in KL_PROMPTS)


def test_logit_kl_perturbed_matches_manual_reference(tokenizer):
    base = sharp_model(tokenizer)
    cur = perturbed(base)
    cur.train()
    base.train()
    res = logit_kl(base, cur, tokenizer, KL_PROMPTS)
    assert cur.training and base.training  # both modes restored
    base.eval()
    refs = [manual_kl_sum(base, cur.eval(), tokenizer, p) for p in KL_PROMPTS]
    assert res.per_prompt_kl == pytest.approx([r[0] for r in refs], rel=1e-9)
    total, count = sum(r[0] for r in refs), sum(r[1] for r in refs)
    assert res.mean_kl == pytest.approx(total / count, rel=1e-9)
    assert res.mean_kl > 0.05
    per_prompt_mean = sum(r[0] / r[1] for r in refs) / len(refs)
    assert abs(res.mean_kl - per_prompt_mean) > 1e-3 * res.mean_kl  # token-weighted, not prompt-weighted
    reverse = logit_kl(cur, base, tokenizer, KL_PROMPTS).mean_kl
    assert abs(reverse - res.mean_kl) > 1e-3 * res.mean_kl  # direction KL(base || current) matters


class MaskedVocab(torch.nn.Module):
    """Wraps a causal LM and sets the logits of ``masked_ids`` to -inf (a constrained vocab head)."""

    def __init__(self, inner, masked_ids) -> None:
        super().__init__()
        self.inner = inner
        self.masked_ids = list(masked_ids)

    def forward(self, input_ids, **kwargs):
        out = self.inner(input_ids=input_ids, **kwargs)
        out.logits[..., self.masked_ids] = float("-inf")
        return out


@torch.no_grad()
def manual_kl_sum_with_zeros(base, cur, tokenizer, prompt: str) -> float:
    """Reference KL in float64 that skips p_base == 0 terms (0 log 0 = 0) explicitly."""
    ids = torch.tensor([ids_of(tokenizer, prompt)])
    pb = torch.softmax(base(input_ids=ids).logits[0].double(), -1)
    pc = torch.softmax(cur(input_ids=ids).logits[0].double(), -1)
    total = 0.0
    for t in range(ids.shape[1]):
        for v in range(pb.shape[1]):
            p, q = float(pb[t, v]), float(pc[t, v])
            if p == 0.0:
                continue
            total += math.inf if q == 0.0 else p * (math.log(p) - math.log(q))
    return total


def test_logit_kl_handles_minus_inf_logits(tokenizer):
    base = sharp_model(tokenizer)
    cur = perturbed(base)
    masked_base = MaskedVocab(base, [5, 9])
    both_masked = logit_kl(masked_base, MaskedVocab(cur, [5, 9]), tokenizer, KL_PROMPTS)
    assert math.isfinite(both_masked.mean_kl)  # not NaN
    for prompt, got in zip(KL_PROMPTS, both_masked.per_prompt_kl):
        ref = manual_kl_sum_with_zeros(masked_base, MaskedVocab(cur, [5, 9]), tokenizer, prompt)
        assert got == pytest.approx(ref, rel=1e-9) and ref > 0
    same = logit_kl(masked_base, masked_base, tokenizer, KL_PROMPTS)
    assert same.mean_kl == 0.0 and same.per_prompt_kl == (0.0, 0.0, 0.0)  # not NaN
    base_only = logit_kl(masked_base, cur, tokenizer, KL_PROMPTS)  # p_base = 0, p_cur > 0: finite
    refs = [manual_kl_sum_with_zeros(masked_base, cur, tokenizer, p) for p in KL_PROMPTS]
    assert base_only.per_prompt_kl == pytest.approx(refs, rel=1e-9)
    assert math.isfinite(base_only.mean_kl)
    cur_only = logit_kl(base, MaskedVocab(cur, [5]), tokenizer, KL_PROMPTS)  # p_base > 0, p_cur = 0
    assert cur_only.mean_kl == math.inf and all(v == math.inf for v in cur_only.per_prompt_kl)


def test_suite_logit_kl_aggregates_per_task_and_overall(tokenizer):
    base = sharp_model(tokenizer)
    cur = perturbed(base)
    suite = RetentionSuite(
        [ProbeTask("t1", (("12+3=", "1"),)), ProbeTask("t2", (("hello world", "x"), ("ab", "y")))]
    )
    kl = suite.logit_kl(base, cur, tokenizer)
    r1 = manual_kl_sum(base, cur, tokenizer, "12+3=")
    r2 = [manual_kl_sum(base, cur, tokenizer, p) for p in ("hello world", "ab")]
    assert kl["t1"] == pytest.approx(r1[0] / r1[1], rel=1e-9)
    assert kl["t2"] == pytest.approx(sum(r[0] for r in r2) / sum(r[1] for r in r2), rel=1e-9)
    all_ref = (r1[0] + sum(r[0] for r in r2)) / (r1[1] + sum(r[1] for r in r2))
    assert kl["all"] == pytest.approx(all_ref, rel=1e-9)


# ---------------------------------------------------------------------------------- report arithmetic
def test_capability_retention_arithmetic_by_hand():
    c = CapabilityRetention.compute("math", 0.8, 0.6, 0.7)
    assert c.absolute_drop == pytest.approx(0.2)
    assert c.relative_drop == pytest.approx(0.25)
    assert c.retention_ratio == pytest.approx(0.75)
    assert c.recovered_retention_ratio == pytest.approx(0.875)
    assert c.recovery_ratio == pytest.approx(0.5)
    assert c.degraded

    worse = CapabilityRetention.compute("code", 0.5, 0.25, 0.125)
    assert worse.recovery_ratio == pytest.approx(-0.5)
    over = CapabilityRetention.compute("code", 0.5, 0.25, 0.75)
    assert over.recovery_ratio == pytest.approx(2.0)

    no_drop = CapabilityRetention.compute("chat", 0.5, 0.5, 0.6)
    assert no_drop.absolute_drop == 0.0 and no_drop.recovery_ratio is None and not no_drop.degraded
    assert no_drop.retention_ratio == 1.0 and no_drop.relative_drop == 0.0

    improved = CapabilityRetention.compute("x", 0.4, 0.6, 0.5)
    assert improved.absolute_drop == pytest.approx(-0.2)
    assert improved.relative_drop == pytest.approx(-0.5)
    assert improved.retention_ratio == pytest.approx(1.5)
    assert improved.recovery_ratio is None and not improved.degraded

    zero = CapabilityRetention.compute("z", 0.0, 0.0, 0.0)
    assert (
        zero.relative_drop,
        zero.retention_ratio,
        zero.recovered_retention_ratio,
        zero.recovery_ratio,
    ) == (
        None,
        None,
        None,
        None,
    )

    small = CapabilityRetention.compute("s", 0.5, 0.49, 0.5, drop_tolerance=0.02)
    assert not small.degraded and small.recovery_ratio is None
    edge = CapabilityRetention.compute("e", 0.5, 0.25, 0.5, drop_tolerance=0.25)  # drop == tolerance
    assert edge.absolute_drop == 0.25 and not edge.degraded and edge.recovery_ratio is None
    above = CapabilityRetention.compute("e", 0.5, 0.25, 0.5, drop_tolerance=0.125)
    assert above.degraded and above.recovery_ratio == 1.0
    assert CapabilityRetention.compute("s", 0.5, 0.49, 0.5).recovery_ratio == pytest.approx(1.0)

    no_rec = CapabilityRetention.compute("n", 1.0, 0.5)
    assert no_rec.recovery_ratio is None and no_rec.recovered_retention_ratio is None
    with pytest.raises(ValueError):
        CapabilityRetention.compute("bad", float("nan"), 0.5)


def test_retention_report_summary_by_hand():
    rep = RetentionReport.from_scores(
        {"math": 0.8, "code": 0.5, "chat": 0.5, "zero": 0.0},
        {"math": 0.6, "code": 0.25, "chat": 0.5, "zero": 0.25},
        {"math": 0.7, "code": 0.5, "chat": 0.6, "zero": 0.0},
    )
    s = rep.summary()
    assert s["num_capabilities"] == 4 and s["num_degraded"] == 2
    assert s["mean_base"] == pytest.approx(1.8 / 4)
    assert s["mean_after_training"] == pytest.approx(1.6 / 4)
    assert s["mean_after_recovery"] == pytest.approx(1.8 / 4)
    assert s["mean_absolute_drop"] == pytest.approx(0.2 / 4)
    # "zero" (base == 0) has an undefined retention ratio and is excluded, not counted as 0 or 1
    assert s["mean_retention_ratio"] == pytest.approx((0.75 + 0.5 + 1.0) / 3)
    assert s["mean_recovery_ratio"] == pytest.approx((0.5 + 1.0) / 2)  # chat/zero (no drop) excluded
    no_rec = RetentionReport.from_scores({"a": 1.0}, {"a": 1.0}).summary()
    assert no_rec["mean_after_recovery"] is None and no_rec["mean_recovery_ratio"] is None
    with pytest.raises(ValueError):
        RetentionReport.from_scores({"a": 1.0}, {"b": 1.0})
    with pytest.raises(ValueError):
        RetentionReport.from_scores({"a": 1.0}, {"a": 1.0}, {"a": 1.0, "b": 0.0})


# ---------------------------------------------------------------------------------- end to end
def test_suite_report_logged_to_experiment_recorder(tokenizer, tmp_path):
    base = sharp_model(tokenizer)
    trained = perturbed(base, scale=1.0)
    recovered = perturbed(base, seed=7, scale=0.02)  # small drift: partially restores the probes
    base_preds = {p: manual_greedy(base, tokenizer, p, 4, tokenizer.eos_token_id) for p in PROMPTS}
    trained_preds = {p: manual_greedy(trained, tokenizer, p, 4, tokenizer.eos_token_id) for p in PROMPTS}
    recovered_preds = {p: manual_greedy(recovered, tokenizer, p, 4, tokenizer.eos_token_id) for p in PROMPTS}
    changed = [p for p in PROMPTS if base_preds[p] != trained_preds[p]]
    assert len(changed) >= 2  # the perturbation really degrades the probes
    expected_recovered = sum(base_preds[p] == recovered_preds[p] for p in PROMPTS) / len(PROMPTS)
    probes = ProbeTask("probe", tuple((p, base_preds[p]) for p in PROMPTS))
    stable = ProbeTask("stable", ((PROMPTS[0], "##"),))  # scores 0 for every model -> base == 0
    suite = RetentionSuite([probes, stable], max_new_tokens=4)
    report = suite.report(base, trained, tokenizer, recovered)

    expected_trained = 1 - len(changed) / len(PROMPTS)
    expected_ratio = (expected_recovered - expected_trained) / (1.0 - expected_trained)
    assert expected_trained < expected_recovered < 1.0  # a partial, non-trivial recovery
    cap = report.capabilities["probe"]
    assert cap.base == 1.0 and cap.after_recovery == pytest.approx(expected_recovered)
    assert cap.after_training == pytest.approx(expected_trained)
    assert cap.recovery_ratio == pytest.approx(expected_ratio)
    assert report.capabilities["stable"].retention_ratio is None
    kl_rec = [manual_kl_sum(base, recovered, tokenizer, p) for p in PROMPTS]
    rec_ref = sum(r[0] for r in kl_rec) / sum(r[1] for r in kl_rec)
    assert report.kl_after_recovery is not None and report.kl_after_recovery["probe"] > 0
    assert report.kl_after_recovery["probe"] == pytest.approx(rec_ref, rel=1e-9)  # KL(base || recovered)
    reverse = [manual_kl_sum(recovered, base, tokenizer, p) for p in PROMPTS]
    assert abs(sum(r[0] for r in reverse) / sum(r[1] for r in reverse) - rec_ref) > 1e-3 * rec_ref
    kl_tr = [manual_kl_sum(base, trained, tokenizer, p) for p in PROMPTS]
    assert report.kl_after_training is not None
    assert report.kl_after_training["probe"] == pytest.approx(
        sum(r[0] for r in kl_tr) / sum(r[1] for r in kl_tr), rel=1e-9
    )

    rec = recorder(tmp_path)
    report.log_to(rec, step=7)
    stored = ExperimentRecorder.load(rec.run_dir)["evaluations"]
    assert len(stored) == 1 and stored[0]["name"] == "capability_retention" and stored[0]["step"] == 7
    res = stored[0]["results"]
    assert res["capabilities"]["probe"]["after_training"] == pytest.approx(expected_trained)
    assert res["capabilities"]["stable"]["retention_ratio"] is None  # stays JSON null, not "nan"
    assert res["summary"]["mean_recovery_ratio"] == pytest.approx(expected_ratio)
    assert res["kl_after_training"]["all"] == pytest.approx(report.kl_after_training["all"])


def test_identical_model_has_full_retention_and_zero_kl(tokenizer):
    base = sharp_model(tokenizer)
    preds = {p: manual_greedy(base, tokenizer, p, 4, tokenizer.eos_token_id) for p in PROMPTS}
    suite = RetentionSuite([ProbeTask("probe", tuple((p, preds[p]) for p in PROMPTS))], max_new_tokens=4)
    report = suite.report(base, copy.deepcopy(base), tokenizer)
    cap = report.capabilities["probe"]
    assert cap.base == 1.0 and cap.after_training == 1.0
    assert cap.retention_ratio == 1.0 and cap.absolute_drop == 0.0
    assert cap.degraded is False and cap.recovery_ratio is None and cap.after_recovery is None
    assert report.kl_after_training == {"probe": 0.0, "all": 0.0}
    assert report.kl_after_recovery is None
    no_kl = suite.report(base, base, tokenizer, base, with_kl=False)
    assert no_kl.kl_after_training is None and no_kl.kl_after_recovery is None
    assert no_kl.capabilities["probe"].after_recovery == 1.0
