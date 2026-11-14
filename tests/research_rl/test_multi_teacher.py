"""Multi-teacher distillation helpers vs manual references, and TeacherEnsemble inside TRL's own loss path."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

from research_rl.models.tiny import build_tiny_llama
from research_rl.trainers.multi_teacher import (
    MultiTeacherBatch,
    TeacherEnsemble,
    TeacherSpec,
    aggregate_teacher_logprobs,
    masked_forward_kl,
    masked_reverse_kl,
)


def rand_logprobs(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(2.0 * torch.randn(*shape, generator=g, dtype=torch.float64), dim=-1)


# ------------------------------------------------------------------------------------------- aggregation
def test_mixture_and_product_match_manual_loops():
    k, b, t, v = 3, 2, 4, 5
    lp = rand_logprobs(k, b, t, v, seed=0)
    w = [0.5, 0.2, 0.3]
    mix = aggregate_teacher_logprobs(lp, w, "mixture")
    prod = aggregate_teacher_logprobs(lp, w, "product")
    for i in range(b):
        for j in range(t):
            p = [sum(w[q] * math.exp(float(lp[q, i, j, x])) for q in range(k)) for x in range(v)]
            for x in range(v):
                assert float(mix[i, j, x]) == pytest.approx(math.log(p[x]), rel=1e-12)
            un = [math.exp(sum(w[q] * float(lp[q, i, j, x]) for q in range(k))) for x in range(v)]
            z = sum(un)
            for x in range(v):
                assert float(prod[i, j, x]) == pytest.approx(math.log(un[x] / z), rel=1e-10)
    # both are normalised distributions
    torch.testing.assert_close(mix.exp().sum(-1), torch.ones(b, t, dtype=torch.float64))
    torch.testing.assert_close(prod.exp().sum(-1), torch.ones(b, t, dtype=torch.float64))
    # unnormalised weights are normalised; a single teacher reproduces itself in both modes
    torch.testing.assert_close(aggregate_teacher_logprobs(lp, [5.0, 2.0, 3.0], "mixture"), mix)
    for mode in ("mixture", "product"):
        torch.testing.assert_close(aggregate_teacher_logprobs(lp, [0.0, 1.0, 0.0], mode), lp[1])


def test_per_sample_routing_weights():
    k, b, t, v = 2, 3, 2, 4
    lp = rand_logprobs(k, b, t, v, seed=1)
    w = torch.tensor([[1.0, 0.0, 0.25], [0.0, 1.0, 0.75]])  # [K, B]
    mix = aggregate_teacher_logprobs(lp, w, "mixture")
    torch.testing.assert_close(mix[0], lp[0, 0])
    torch.testing.assert_close(mix[1], lp[1, 1])
    torch.testing.assert_close(mix[2], torch.log(0.25 * lp[0, 2].exp() + 0.75 * lp[1, 2].exp()))
    with pytest.raises(ValueError):
        aggregate_teacher_logprobs(lp, torch.ones(k, b + 1), "mixture")
    with pytest.raises(ValueError):
        aggregate_teacher_logprobs(lp, [1.0, -1.0], "mixture")


def test_masked_kl_helpers_vs_manual():
    b, t, v = 2, 4, 6
    ps, pt = rand_logprobs(b, t, v, seed=2), rand_logprobs(b, t, v, seed=3)
    mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]])
    fwd_tok = torch.zeros(b, t, dtype=torch.float64)
    rev_tok = torch.zeros(b, t, dtype=torch.float64)
    for i in range(b):
        for j in range(t):
            for x in range(v):
                a, c = float(pt[i, j, x]), float(ps[i, j, x])
                fwd_tok[i, j] += math.exp(a) * (a - c)
                rev_tok[i, j] += math.exp(c) * (c - a)
    m = mask.bool()
    assert float(masked_forward_kl(pt, ps, mask)) == pytest.approx(float(fwd_tok[m].mean()), rel=1e-12)
    assert float(masked_reverse_kl(ps, pt, mask)) == pytest.approx(float(rev_tok[m].mean()), rel=1e-12)
    assert float(masked_reverse_kl(ps, pt, mask, "sum")) == pytest.approx(float(rev_tok[m].sum()), rel=1e-12)
    torch.testing.assert_close(masked_forward_kl(pt, ps, mask, "none"), fwd_tok * mask)
    # forward and reverse KL differ (not vacuous) and a zero-probability teacher entry is handled
    assert abs(float(fwd_tok[m].mean() - rev_tok[m].mean())) > 1e-3
    pt0 = pt.clone()
    pt0[..., 0] = -torch.inf
    pt0 = torch.log_softmax(pt0, -1)
    assert torch.isfinite(masked_forward_kl(pt0, ps, mask))


def test_multi_teacher_batch():
    k, b, t, v = 2, 2, 3, 5
    lp = rand_logprobs(k, b, t, v, seed=4)
    ids = torch.tensor([[0, 3, 4], [2, 2, 1]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    batch = MultiTeacherBatch(ids, mask, lp, torch.tensor([0.7, 0.3]), ["math", "code"])
    tok = batch.teacher_token_logprobs()
    for q in range(k):
        for i in range(b):
            for j in range(t):
                assert tok[q, i, j] == lp[q, i, j, ids[i, j]]
    student = rand_logprobs(b, t, v, seed=5)
    ref_mix = torch.log(0.7 * lp[0].exp() + 0.3 * lp[1].exp())
    m = mask.bool()
    ref = (student.exp() * (student - ref_mix)).sum(-1)[m].mean()
    torch.testing.assert_close(batch.reverse_kl(student, "mixture"), ref)
    with pytest.raises(ValueError):
        MultiTeacherBatch(ids, mask[:, :2], lp, torch.ones(k))


# ------------------------------------------------------------------------------------------- ensemble
V = 40


def make_teachers():
    return [build_tiny_llama(V, seed=1), build_tiny_llama(V, seed=2)]


def ids_and_mask():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, V, (2, 9), generator=g)
    mask = torch.ones_like(ids)
    mask[0, :2] = 0  # left padding on row 0
    return ids, mask


@pytest.mark.parametrize("mode", ["mixture", "product"])
def test_ensemble_forward_matches_manual_aggregation(mode):
    teachers = make_teachers()
    ens = TeacherEnsemble(teachers, [0.3, 0.7], mode)
    ids, mask = ids_and_mask()
    with torch.no_grad():
        out = ens(input_ids=ids, attention_mask=mask).logits
        l1 = torch.log_softmax(teachers[0](input_ids=ids, attention_mask=mask).logits.float(), -1)
        l2 = torch.log_softmax(teachers[1](input_ids=ids, attention_mask=mask).logits.float(), -1)
    if mode == "mixture":
        ref = torch.log(0.3 * l1.exp() + 0.7 * l2.exp())
    else:
        s = 0.3 * l1 + 0.7 * l2
        ref = s - torch.logsumexp(s, -1, keepdim=True)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert all(not p.requires_grad for p in ens.parameters())


@pytest.mark.parametrize("mode", ["mixture", "product"])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_ensemble_through_trl_chunked_loss_path(mode, beta):
    """Drive TRL's own `_get_last_hidden_state` + `_chunked_divergence_loss` with the ensemble."""
    from trl.trainer.distillation_trainer import DistillationTrainer, _chunked_divergence_loss

    teachers = make_teachers()
    student = build_tiny_llama(V, seed=7)
    ens = TeacherEnsemble(teachers, [0.4, 0.6], mode)
    ids, attn = ids_and_mask()
    k = 5  # completion length
    completion_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    fake = SimpleNamespace(model_kwarg_keys={"input_ids", "attention_mask", "logits_to_keep"}, _is_vlm=False)
    get_h = DistillationTrainer._get_last_hidden_state
    with torch.no_grad():
        hs = get_h(fake, student, ids, attn, k)
        ht = get_h(fake, ens, ids, attn, k)
        s_head, t_head = student.get_output_embeddings(), ens.get_output_embeddings()
        temp = 1.5
        loss, _, n_valid = _chunked_divergence_loss(
            hs,
            ht,
            s_head.weight,
            t_head.weight,
            completion_mask,
            beta,
            student_lm_head_bias=s_head.bias,
            teacher_lm_head_bias=t_head.bias,
            temperature=temp,
        )
        # independent reference: full forwards, manual aggregation, manual KL
        s_logits = student(input_ids=ids, attention_mask=attn).logits[:, -k - 1 : -1].float()
        t_lp = [
            torch.log_softmax(t(input_ids=ids, attention_mask=attn).logits[:, -k - 1 : -1].float(), -1)
            for t in teachers
        ]
        if mode == "mixture":
            agg = torch.log(0.4 * t_lp[0].exp() + 0.6 * t_lp[1].exp())
        else:
            agg = 0.4 * t_lp[0] + 0.6 * t_lp[1]
        ls = torch.log_softmax(s_logits / temp, -1)
        lt = torch.log_softmax(agg / temp, -1)
        ref = (
            masked_forward_kl(lt, ls, completion_mask)
            if beta == 0.0
            else masked_reverse_kl(ls, lt, completion_mask)
        )
        # the ensemble's TRL view equals its forward() up to a per-position constant
        view = torch.log_softmax(t_head(ht).float(), -1)
        torch.testing.assert_close(
            view,
            torch.log_softmax(ens(input_ids=ids, attention_mask=attn).logits[:, -k - 1 : -1], -1),
            rtol=1e-5,
            atol=1e-5,
        )
    assert int(n_valid) == 8
    assert float(ref) > 1e-3
    torch.testing.assert_close(loss, ref.to(loss.dtype), rtol=1e-4, atol=1e-6)


def test_mixture_identity_head_is_guarded():
    ens = TeacherEnsemble(make_teachers(), mode="mixture", max_identity_vocab=V - 1)
    with pytest.raises(ValueError, match="identity"):
        ens.get_output_embeddings()
    with pytest.raises(ValueError, match="vocabulary"):
        TeacherEnsemble([build_tiny_llama(V, seed=1), build_tiny_llama(V + 1, seed=2)])


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["mixture", "product"])
def test_distillation_trainer_with_teacher_ensemble(tmp_path, tiny_model_dir, tokenizer, mode):
    from trl import DistillationConfig, DistillationTrainer

    teacher_b = build_tiny_llama(len(tokenizer), seed=3)
    ens = TeacherEnsemble.from_specs(
        [TeacherSpec("a", tiny_model_dir, 1.0), TeacherSpec("b", teacher_b, 3.0)], mode=mode
    )
    ref_a = AutoModelForCausalLM.from_pretrained(tiny_model_dir)  # string path was loaded
    assert all(torch.equal(p, q) for p, q in zip(ens.teachers[0].parameters(), ref_a.parameters()))
    torch.testing.assert_close(ens.weights, torch.tensor([0.25, 0.75], dtype=torch.float64))
    student = build_tiny_llama(len(tokenizer), seed=11)
    before = {n: p.detach().clone() for n, p in student.named_parameters()}
    teacher_before = {n: p.detach().clone() for n, p in ens.named_parameters()}
    args = DistillationConfig(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        max_completion_length=6,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        use_cpu=True,
        seed=0,
        beta=1.0,
        bf16=False,
    )
    ds = Dataset.from_dict({"prompt": ["1+1=", "2*3=", "abc", "xyz"]})
    trainer = DistillationTrainer(
        model=student, teacher_model=ens, args=args, train_dataset=ds, processing_class=tokenizer
    )
    out = trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    assert trainer.state.global_step == 2 and len(losses) == 2
    assert all(math.isfinite(x) and x > 0 for x in losses)
    assert out.training_loss > 0
    assert any(not torch.equal(before[n], p) for n, p in student.named_parameters())
    assert all(torch.equal(teacher_before[n], p) for n, p in ens.named_parameters())
