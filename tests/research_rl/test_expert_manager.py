"""Expert Manager: freezing, gradient flow, routing, expert files, vanilla export and TRL resume.

Every numerical check compares against an independent reference: an untouched vanilla copy of the
model (autograd / forward), per-row forward passes, or bit-exact parameter snapshots.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import safetensors.torch as st
import torch
from datasets import Dataset
from transformers import LlamaForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer

from research_rl.core.checkpoint import (
    COMPONENTS_FILE,
    checkpoint_inventory,
    load_components,
    save_components,
)
from research_rl.models.expert_manager import (
    BackboneChangedError,
    ExpertCompatibilityError,
    ExpertError,
    ExpertFFN,
    ExpertManager,
    config_hash,
    load_model_with_experts,
)
from research_rl.models.model_interface import fingerprint, snapshot
from research_rl.models.tiny import build_tiny_llama, tiny_llama_config
from research_rl.trainers.base import ResearchCallback, research_trainer_class

MLP = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------- helpers
def perturb(manager: ExpertManager, name: str, seed: int, scale: float = 0.2) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in manager.expert_parameters(name):
            p.add_(torch.randn(p.shape, generator=gen) * scale)


def batch(n: int = 4, length: int = 9, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(4, 90, (n, length), generator=gen)


def logits(model, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask).logits


def managed(
    tokenizer, layers=(0, 1), experts=("a", "b"), seed: int = 0
) -> tuple[torch.nn.Module, ExpertManager]:
    model = build_tiny_llama(len(tokenizer), seed=seed)
    mgr = ExpertManager(model)
    mgr.attach(layers)
    for i, name in enumerate(experts):
        mgr.create_expert(name)
        perturb(mgr, name, seed=100 + i)
    return model, mgr


def sft_data() -> Dataset:
    rows = [{"prompt": f"{a}+{b}=", "completion": str(a + b)} for a in range(4) for b in range(4)]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------- structure / selection
def test_attach_copies_are_exact_and_selection_validates(tokenizer):
    model = build_tiny_llama(len(tokenizer), seed=0)
    vanilla = copy.deepcopy(model)
    ids = batch()
    ref = logits(vanilla, ids)
    base_params = {n: p for n, p in model.named_parameters() if ".mlp." in n}

    mgr = ExpertManager(model)
    assert mgr.attach([1]) == [1]
    assert isinstance(model.model.layers[1].mlp, ExpertFFN)
    assert not isinstance(model.model.layers[0].mlp, ExpertFFN)
    # the original module is kept as "base": same parameter objects, no copy
    for proj in MLP:
        assert (
            getattr(model.model.layers[1].mlp.experts["base"], proj).weight
            is base_params[f"model.layers.1.mlp.{proj}.weight"]
        )
    mgr.create_expert("a")
    assert mgr.expert_names == ["base", "a"]
    for proj in MLP:
        wa = getattr(model.model.layers[1].mlp.experts["a"], proj).weight
        wb = getattr(model.model.layers[1].mlp.experts["base"], proj).weight
        assert torch.equal(wa, wb) and wa.data_ptr() != wb.data_ptr()
    assert torch.equal(logits(model, ids), ref)
    with mgr.using("a"):
        assert torch.equal(logits(model, ids), ref)

    with pytest.raises(ExpertError):
        mgr.activate("nope")
    with pytest.raises(ExpertError), mgr.using("nope"):
        pass
    with pytest.raises(ExpertError):
        mgr.set_routing(["a", "nope"])
    with pytest.raises(ExpertError):
        mgr.create_expert("c", source="nope")
    with pytest.raises(ExpertError):
        mgr.create_expert("bad.name")
    with pytest.raises(ExpertError):
        mgr.create_expert("a")
    with pytest.raises(ExpertError):
        mgr.attach([0])
    with pytest.raises(ExpertError):
        mgr.set_trainable("base")


def test_context_managers_restore_selection_on_error(tokenizer):
    _, mgr = managed(tokenizer)
    mgr.activate("b")
    with pytest.raises(RuntimeError), mgr.using("a"):
        assert mgr.active == "a"
        raise RuntimeError
    assert mgr.active == "b" and mgr.routing is None
    with pytest.raises(RuntimeError), mgr.routed(["a", "base"]):
        assert mgr.routing == ("a", "base")
        raise RuntimeError
    assert mgr.active == "b" and mgr.routing is None


# ---------------------------------------------------------------------- (a) gradient through frozen modules
def test_gradient_flows_through_frozen_backbone_and_matches_vanilla_autograd(tokenizer):
    model = build_tiny_llama(len(tokenizer), seed=0)
    vanilla = copy.deepcopy(model)  # untouched reference, every parameter trainable
    mgr = ExpertManager(model)
    mgr.attach([0])  # expert in layer 0: layer 1, final norm and lm_head (all frozen) come after it
    mgr.create_expert("math")
    mgr.set_trainable("math")
    expected_names = [f"model.layers.0.mlp.experts.math.{p}.weight" for p in MLP]
    assert sorted(mgr.trainable_parameter_names()) == sorted(expected_names)
    assert mgr.active == "math"

    ids = batch(3, 11, seed=1)
    model(input_ids=ids, labels=ids).loss.backward()
    vanilla(input_ids=ids, labels=ids).loss.backward()

    vparams = dict(vanilla.named_parameters())
    for proj in MLP:
        g = dict(model.named_parameters())[f"model.layers.0.mlp.experts.math.{proj}.weight"].grad
        g_ref = vparams[f"model.layers.0.mlp.{proj}.weight"].grad
        assert g is not None and g_ref is not None
        assert g.abs().sum() > 0
        torch.testing.assert_close(g, g_ref, rtol=1e-5, atol=1e-7)
    for name, p in model.named_parameters():
        if name not in expected_names:
            assert p.grad is None, name


# ---------------------------------------------------------------------- (b) independent training
def test_torch_loop_with_routing_only_moves_trainable_expert(tokenizer):
    model, mgr = managed(tokenizer, layers=(0, 1))
    mgr.set_trainable("a")
    ref_backbone = mgr.backbone_fingerprint()
    before = snapshot(model)
    fp_b, fp_base = mgr.expert_fingerprint("b"), mgr.expert_fingerprint("base")

    # The optimizer gets *all* parameters and weight decay; only requires_grad keeps "b" fixed even
    # though "b" is in the forward path of half the rows.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    ids = batch(4, 10, seed=2)
    mgr.set_routing(["a", "b", "a", "b"])
    for _ in range(3):
        opt.zero_grad()
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()

    after = snapshot(model)
    assert mgr.backbone_fingerprint() == ref_backbone
    mgr.assert_backbone_unchanged()
    assert mgr.expert_fingerprint("b") == fp_b and mgr.expert_fingerprint("base") == fp_base
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    assert changed == set(mgr.expert_parameter_names("a"))

    with torch.no_grad():
        model.model.norm.weight[0] += 1e-3
    with pytest.raises(BackboneChangedError):
        mgr.assert_backbone_unchanged()


class StepCounter(TrainerCallback):
    def __init__(self) -> None:
        self.steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.steps += 1


def make_expert_trainer(tokenizer, out_dir, max_steps, *, model_seed=0, save_steps=None, create=True):
    model = build_tiny_llama(len(tokenizer), seed=model_seed)
    mgr = ExpertManager(model)
    mgr.attach([0])
    if create:  # a resumed run starts with only "base": the components file recreates the rest
        mgr.create_expert("math")
        mgr.create_expert("other")
        perturb(mgr, "other", seed=5)
        mgr.set_trainable("math")
    args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        learning_rate=5e-3,
        weight_decay=0.1,
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        seed=7,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
    )
    callback, counter = ResearchCallback(), StepCounter()
    trainer = research_trainer_class(SFTTrainer)(
        model=model,
        args=args,
        train_dataset=sft_data(),
        processing_class=tokenizer,
        components={"experts": mgr},
        callbacks=[callback, counter],
    )
    callback.bind(trainer)
    return trainer, mgr, counter


@pytest.mark.integration
def test_sft_trainer_trains_only_the_designated_expert(tokenizer, tmp_path):
    trainer, mgr, _ = make_expert_trainer(tokenizer, tmp_path / "run", max_steps=3)
    model = trainer.model
    ref_backbone = mgr.backbone_fingerprint()
    fp_other, fp_base = mgr.expert_fingerprint("other"), mgr.expert_fingerprint("base")
    before = snapshot(model, mgr.expert_parameter_names("math"))
    trainer.train()

    opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert opt_ids == {id(p) for p in mgr.expert_parameters("math")}
    mgr.assert_backbone_unchanged()
    assert mgr.backbone_fingerprint() == ref_backbone
    assert mgr.expert_fingerprint("other") == fp_other and mgr.expert_fingerprint("base") == fp_base
    after = snapshot(model, before)
    assert all(not torch.equal(before[n], after[n]) for n in before)


@pytest.mark.integration
def test_trainer_checkpoint_resume_continues_training_the_expert(tokenizer, tmp_path):
    full, full_mgr, _ = make_expert_trainer(tokenizer, tmp_path / "full", max_steps=4)
    full.train()
    full_params = snapshot(full.model)

    part, _, _ = make_expert_trainer(tokenizer, tmp_path / "part", max_steps=4, save_steps=2)
    part.train()
    ckpt = tmp_path / "part" / "checkpoint-2"
    inv = checkpoint_inventory(ckpt)
    assert inv["components"] and inv["model_weights"] and inv["optimizer"]
    assert (ckpt / COMPONENTS_FILE).exists()

    # Different seed and only the base expert: only a genuine resume reproduces the full run.
    resumed, mgr, counter = make_expert_trainer(
        tokenizer, tmp_path / "part", max_steps=4, save_steps=2, model_seed=123, create=False
    )
    assert mgr.expert_names == ["base"]
    resumed.train(resume_from_checkpoint=str(ckpt))
    assert counter.steps == 2
    assert mgr.expert_names == ["base", "math", "other"] and mgr.trainable_expert == "math"
    resumed_params = snapshot(resumed.model)
    assert resumed_params.keys() == full_params.keys()
    for name, tensor in full_params.items():
        assert torch.equal(tensor, resumed_params[name]), name
    # the restored reference is the original backbone, not the seed-123 one present at load time
    mgr.assert_backbone_unchanged()
    assert mgr.backbone_fingerprint() == full_mgr.backbone_fingerprint()

    # an HF Trainer checkpoint (experts attached, no sidecar) re-loads via the expert-aware loader
    loaded, lmgr = load_model_with_experts(ckpt)
    assert lmgr.layers == [0] and lmgr.expert_names == ["base", "math", "other"]
    assert config_hash(loaded.config) == config_hash(full.model.config)
    saved = st.load_file(str(ckpt / "model.safetensors"))
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, saved[name]), name


# ---------------------------------------------------------------------- (c) expert files
def test_save_load_expert_roundtrip_and_switching(tokenizer, tmp_path):
    model, mgr = managed(tokenizer)
    ids = batch()
    ref = {}
    for name in ("base", "a", "b"):
        with mgr.using(name):
            ref[name] = logits(model, ids)
    assert not torch.allclose(ref["a"], ref["base"]) and not torch.allclose(ref["a"], ref["b"])
    mgr.save_expert("a", tmp_path / "a.safetensors")
    mgr.save_expert("b", tmp_path / "b.safetensors")
    _, meta = ExpertManager.read_expert_file(tmp_path / "a.safetensors")
    assert meta["expert_name"] == "a" and json.loads(meta["layers"]) == [0, 1]
    assert meta["backbone_fingerprint"] == mgr.backbone_fingerprint()

    fresh = build_tiny_llama(len(tokenizer), seed=0)
    fmgr = ExpertManager(fresh)
    fmgr.attach([0, 1])
    assert fmgr.load_expert(tmp_path / "a.safetensors") == "a"
    assert fmgr.load_expert(tmp_path / "b.safetensors") == "b"
    for name in ("base", "a", "b"):
        with fmgr.using(name):
            assert torch.equal(logits(fresh, ids), ref[name]), name
    assert fmgr.expert_fingerprint("a") == mgr.expert_fingerprint("a")
    with pytest.raises(ExpertError):
        fmgr.load_expert(tmp_path / "a.safetensors")  # exists, no overwrite
    fmgr.load_expert(tmp_path / "b.safetensors", name="a", overwrite=True)
    with fmgr.using("a"):
        assert torch.equal(logits(fresh, ids), ref["b"])


def _flip_last_byte(src: Path, dst: Path) -> None:
    raw = bytearray(src.read_bytes())
    raw[-1] ^= 0x01  # tensor payload, not the header: safetensors still parses it
    dst.write_bytes(bytes(raw))


def test_corrupted_or_mismatched_expert_files_are_rejected(tokenizer, tmp_path):
    _, mgr = managed(tokenizer)
    good = mgr.save_expert("a", tmp_path / "a.safetensors")

    def target(seed=0, layers=(0, 1), **cfg):
        m = build_tiny_llama(len(tokenizer), seed=seed, **cfg)
        tm = ExpertManager(m)
        tm.attach(layers)
        return tm

    flipped = tmp_path / "flipped.safetensors"
    _flip_last_byte(good, flipped)
    st.load_file(str(flipped))  # proves the file itself is still readable
    tm = target()
    with pytest.raises(ExpertCompatibilityError, match="corrupted"):
        tm.load_expert(flipped)
    assert tm.expert_names == ["base"]  # nothing half-created

    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(good.read_bytes()[:40])
    with pytest.raises(ExpertCompatibilityError):
        tm.load_expert(truncated)

    with pytest.raises(ExpertCompatibilityError, match="fingerprint"):
        target(seed=1).load_expert(good)
    # identical backbone, different base FFN: only the base-expert check catches it
    other_base = target()
    perturb(other_base, "base", seed=9, scale=1e-3)
    assert other_base.backbone_fingerprint() == mgr.backbone_fingerprint()
    with pytest.raises(ExpertCompatibilityError, match="base_expert_fingerprint"):
        other_base.load_expert(good, check_backbone=False)
    with pytest.raises(ExpertCompatibilityError, match="layers"):
        target(layers=(0,)).load_expert(good)
    with pytest.raises(ExpertCompatibilityError, match="config_hash"):
        target(intermediate_size=48).load_expert(good)

    # same base FFN but a different backbone elsewhere: only the backbone check catches it
    other = target()
    with torch.no_grad():
        other.model.model.norm.weight.mul_(1.5)
    with pytest.raises(ExpertCompatibilityError, match="backbone_fingerprint"):
        other.load_expert(good)
    other.load_expert(good, check_backbone=False)
    assert other.expert_names == ["base", "a"]


# ---------------------------------------------------------------------- (d) per-sample routing
def test_per_sample_routing_matches_rowwise_reference(tokenizer):
    model, mgr = managed(tokenizer)
    # Unsorted with repeats; grouped order [0, 3, 1, 4, 2] is not its own inverse, so a scatter that
    # uses the grouping order instead of its inverse permutation is caught.
    routing = ["b", "a", "base", "b", "a"]
    ids = batch(5, 9, seed=3)
    with mgr.routed(routing):
        out = logits(model, ids)
    for i, name in enumerate(routing):
        with mgr.using(name):
            torch.testing.assert_close(out[i : i + 1], logits(model, ids[i : i + 1]), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out, logits(model, ids), atol=1e-3)  # differs from all-"base"

    # left padding: each reference row is run with its own attention mask
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    mask[3, :5] = 0
    ids_p = ids.masked_fill(mask == 0, 0)
    with mgr.routed(routing):
        out_p = logits(model, ids_p, mask)
    for i, name in enumerate(routing):
        with mgr.using(name):
            ref = logits(model, ids_p[i : i + 1], mask[i : i + 1])
        keep = mask[i].bool()
        torch.testing.assert_close(out_p[i, keep], ref[0, keep], rtol=1e-5, atol=1e-5)

    # Every row routed to ONE expert that differs from `active` (the single-group shortcut): the
    # routing, not `active`, must decide.
    mgr.activate("base")
    base_out = logits(model, ids[:3])
    for rows, name in ((3, "b"), (1, "a")):
        with mgr.using(name):
            ref_uniform = logits(model, ids[:rows])
        with mgr.routed([name] * rows):
            out_uniform = logits(model, ids[:rows])
        assert mgr.active == "base" and mgr.routing is None
        torch.testing.assert_close(out_uniform, ref_uniform, rtol=1e-5, atol=1e-5)
        assert not torch.allclose(out_uniform, base_out[:rows], atol=1e-3)

    with mgr.routed(["a", "b"]), pytest.raises(ExpertError, match="batch size"):
        logits(model, ids)


def test_routing_holds_for_the_whole_generated_response(tokenizer):
    model, mgr = managed(tokenizer, experts=("a", "b"))
    perturb(mgr, "a", seed=11, scale=1.0)
    perturb(mgr, "b", seed=12, scale=1.0)
    ids = batch(4, 6, seed=4)
    routing = ["a", "base", "b", "a"]
    kw = {"max_new_tokens": 6, "do_sample": False, "eos_token_id": None, "pad_token_id": 0}
    with mgr.routed(routing):
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
    rows = []
    for i, name in enumerate(routing):
        with mgr.using(name):
            rows.append(
                model.generate(input_ids=ids[i : i + 1], attention_mask=torch.ones_like(ids[i : i + 1]), **kw)
            )
    assert torch.equal(gen, torch.cat(rows))
    assert len({tuple(r[0, 6:].tolist()) for r in rows}) > 1  # experts really generate differently


# ---------------------------------------------------------------------- (e) vanilla export and the pitfall
def test_merged_export_loads_as_vanilla_llama_with_identical_logits(tokenizer, tmp_path):
    model, mgr = managed(tokenizer)
    ids = batch()
    with mgr.using("a"):
        ref = logits(model, ids)
    out = mgr.export_merged("a", tmp_path / "merged")
    vanilla_keys = set(build_tiny_llama(len(tokenizer)).state_dict())
    assert set(st.load_file(str(out / "model.safetensors"))) == vanilla_keys
    loaded, info = LlamaForCausalLM.from_pretrained(str(out), output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert torch.equal(logits(loaded, ids), ref)
    # the attached model is untouched by the export
    assert mgr.expert_names == ["base", "a", "b"] and isinstance(model.model.layers[0].mlp, ExpertFFN)

    merged = mgr.merged_model("b")
    assert not any(isinstance(m, ExpertFFN) for m in merged.modules())
    with mgr.using("b"):
        assert torch.equal(logits(merged, ids), logits(model, ids))

    mgr.detach("a")
    assert set(model.state_dict()) == vanilla_keys
    assert torch.equal(logits(model, ids), ref)


def test_attached_checkpoint_pitfall_and_expert_aware_loader(tokenizer, tmp_path):
    model, mgr = managed(tokenizer)
    mgr.activate("b")
    ids = batch()
    ref = {}
    for name in ("base", "a", "b"):
        with mgr.using(name):
            ref[name] = logits(model, ids)
    path = mgr.save_pretrained(tmp_path / "attached")

    # Pitfall: plain from_pretrained silently re-initialises the FFNs of attached layers.
    plain, info = LlamaForCausalLM.from_pretrained(str(path), output_loading_info=True)
    assert "model.layers.0.mlp.gate_proj.weight" in info["missing_keys"]
    assert "model.layers.0.mlp.experts.a.gate_proj.weight" in info["unexpected_keys"]
    plain_logits = logits(plain, ids)
    assert all(not torch.allclose(plain_logits, r, atol=1e-4) for r in ref.values())

    loaded, lmgr = load_model_with_experts(path)
    assert lmgr.active == "b"  # selection restored from the sidecar
    for name in ("base", "a", "b"):
        with lmgr.using(name):
            assert torch.equal(logits(loaded, ids), ref[name]), name

    (path / "research_rl_experts.json").unlink()  # structure inferred from keys alone
    loaded2, lmgr2 = load_model_with_experts(path)
    assert lmgr2.expert_names == ["base", "a", "b"] and lmgr2.active == "base"
    with lmgr2.using("a"):
        assert torch.equal(logits(loaded2, ids), ref["a"])

    with pytest.raises(ExpertCompatibilityError, match="no expert keys"):
        load_model_with_experts(mgr.export_merged("a", tmp_path / "vanilla"))

    # a sidecar that disagrees with the weights is rejected (unexpected / missing expert keys)
    sidecar = mgr.state_dict()
    (path / "research_rl_experts.json").write_text(
        json.dumps(dict(sidecar, experts=["base", "a"], active="a"))
    )
    with pytest.raises(ExpertCompatibilityError, match="unexpected keys"):
        load_model_with_experts(path)
    (path / "research_rl_experts.json").write_text(json.dumps(dict(sidecar, experts=["base", "a", "b", "c"])))
    with pytest.raises(ExpertCompatibilityError, match="missing keys"):
        load_model_with_experts(path)
    detached_state = ExpertManager(build_tiny_llama(len(tokenizer))).state_dict()
    (path / "research_rl_experts.json").write_text(json.dumps(detached_state))
    with pytest.raises(ExpertCompatibilityError, match="detached"):
        load_model_with_experts(path)
    (path / "research_rl_experts.json").unlink()
    weights = st.load_file(str(path / "model.safetensors"))
    weights["model.norm.weight"] = weights["model.norm.weight"].half()
    st.save_file(weights, str(path / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ExpertCompatibilityError, match="mixed floating dtypes"):
        load_model_with_experts(path)


def test_tied_embeddings_sharded_checkpoint_and_custom_factory(tokenizer, tmp_path):
    cfg = tiny_llama_config(len(tokenizer))
    cfg.tie_word_embeddings = True
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    mgr = ExpertManager(model)
    mgr.attach([1])
    mgr.create_expert("a")
    perturb(mgr, "a", seed=3)
    ids = batch()
    ref = {}
    for name in ("base", "a"):
        with mgr.using(name):
            ref[name] = logits(model, ids)

    path = mgr.save_pretrained(tmp_path / "sharded", max_shard_size="20KB")
    assert (path / "model.safetensors.index.json").exists() and not (path / "model.safetensors").exists()
    calls = []

    def factory(config, dtype):
        calls.append(dtype)
        return LlamaForCausalLM(config).to(dtype)

    loaded, lmgr = load_model_with_experts(path, model_factory=factory)
    assert calls == [torch.float32]
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight  # tied key absent from file
    assert lmgr.backbone_fingerprint() == mgr.backbone_fingerprint()
    for name in ("base", "a"):
        with lmgr.using(name):
            assert torch.equal(logits(loaded, ids), ref[name]), name

    out = mgr.export_merged("a", tmp_path / "tied_vanilla")
    vanilla, info = LlamaForCausalLM.from_pretrained(str(out), output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert torch.equal(logits(vanilla, ids), ref["a"])


def test_delete_expert_and_freeze_controls(tokenizer):
    model, mgr = managed(tokenizer)
    with pytest.raises(ExpertError, match="no backbone reference"):
        mgr.assert_backbone_unchanged()
    mgr.assert_backbone_unchanged(reference=fingerprint(model, mgr.backbone_parameter_names()))

    fp = mgr.freeze_backbone()  # no trainable expert: everything frozen
    assert fp == mgr.backbone_fingerprint() and mgr.trainable_parameter_names() == []
    mgr.set_trainable("a")
    assert sorted(mgr.trainable_parameter_names()) == sorted(mgr.expert_parameter_names("a"))
    mgr.set_trainable(None)
    assert mgr.trainable_parameter_names() == [] and mgr.trainable_expert is None
    mgr.assert_backbone_unchanged()

    with pytest.raises(ExpertError, match="base"):
        mgr.delete_expert("base")
    mgr.set_trainable("a")
    with pytest.raises(ExpertError, match="trainable"):
        mgr.delete_expert("a")
    mgr.set_trainable(None)
    mgr.activate("b")
    with pytest.raises(ExpertError, match="active or routed"):
        mgr.delete_expert("b")
    mgr.set_routing(["base", "b"])
    with pytest.raises(ExpertError, match="active or routed"):
        mgr.delete_expert("b")
    mgr.clear_routing()
    mgr.activate("base")
    mgr.delete_expert("a")
    assert mgr.expert_names == ["base", "b"]
    assert not any(".experts.a." in n for n, _ in model.named_parameters())
    with pytest.raises(ExpertError):
        mgr.activate("a")


# ---------------------------------------------------------------------- state / hashing
def test_manager_state_roundtrip_and_config_hash_is_stable(tokenizer, tmp_path):
    model, mgr = managed(tokenizer, layers=(1,))
    mgr.set_trainable("b")
    torch.save({"experts": mgr.state_dict()}, tmp_path / "c.pt")
    state = torch.load(tmp_path / "c.pt", weights_only=True)["experts"]

    other = build_tiny_llama(len(tokenizer), seed=0)
    omgr = ExpertManager(other)
    omgr.load_state_dict(state)
    assert omgr.layers == [1] and omgr.expert_names == ["base", "a", "b"]
    assert omgr.active == "b" and omgr.trainable_expert == "b"
    assert sorted(omgr.trainable_parameter_names()) == sorted(omgr.expert_parameter_names("b"))
    # recorded fingerprint is the original one (backbones are identical here, seed 0)
    omgr.assert_backbone_unchanged()

    bad = dict(state, layers=[0])
    with pytest.raises(ExpertCompatibilityError):
        omgr.load_state_dict(bad)

    model.save_pretrained(tmp_path / "rt")
    assert config_hash(LlamaForCausalLM.from_pretrained(str(tmp_path / "rt")).config) == config_hash(
        model.config
    )
    assert config_hash(model.config) != config_hash(
        build_tiny_llama(len(tokenizer), intermediate_size=48).config
    )
    assert fingerprint(model, mgr.backbone_parameter_names()) == mgr.backbone_fingerprint()


def test_find_swiglu_ffns_resolves_experts_for_channel_tools(tiny_model):
    """Core model_interface sees inside ExpertFFN, so FFN channel ops can target one expert."""
    from research_rl.models.ffn_channels import ChannelSelection, channel_mask
    from research_rl.models.model_interface import ModelInterfaceError, find_swiglu_ffns

    mgr = ExpertManager(tiny_model)
    mgr.attach([0])
    mgr.create_expert("a")
    params = dict(tiny_model.named_parameters())
    refs = find_swiglu_ffns(tiny_model, [0], expert="a")
    assert refs[0].module_path == "model.layers.0.mlp.experts.a"
    for full in refs[0].param_names().values():
        assert full in params
    assert refs[0].gate.weight is params["model.layers.0.mlp.experts.a.gate_proj.weight"]
    mask = channel_mask(tiny_model, ChannelSelection({0: [1, 5]}))
    assert set(mask.names()) == {
        f"model.layers.0.mlp.experts.base.{k}.weight" for k in ("gate_proj", "up_proj", "down_proj")
    }
    with pytest.raises(ModelInterfaceError):
        find_swiglu_ffns(tiny_model, [0], expert="missing")
    with pytest.raises(ModelInterfaceError):
        find_swiglu_ffns(tiny_model, [1], expert="a")


# ---------------------------------------------------------------------- review regressions
def test_switching_experts_never_rebaselines_a_corrupted_backbone(tokenizer, tmp_path):
    model, mgr = managed(tokenizer, layers=(0,))
    ref = mgr.freeze_backbone()
    mgr.set_trainable("a")
    mgr.set_trainable("b")  # clean switch: same reference, verified, not re-recorded
    assert mgr.freeze_backbone() == ref and mgr._backbone_ref == ref
    mgr.set_trainable("a")

    with torch.no_grad():
        model.model.norm.weight[0] += 1.0  # backbone corrupted while training "a"
    with pytest.raises(BackboneChangedError):
        mgr.set_trainable("b")
    # the failed switch changed nothing
    assert mgr.trainable_expert == "a" and mgr.active == "a"
    assert sorted(mgr.trainable_parameter_names()) == sorted(mgr.expert_parameter_names("a"))
    with pytest.raises(BackboneChangedError):
        mgr.freeze_backbone()
    with pytest.raises(BackboneChangedError):
        mgr.set_trainable(None)
    with pytest.raises(BackboneChangedError):
        mgr.assert_backbone_unchanged()  # still detected after the failed attempts

    # explicit, intentional reset
    mgr.set_trainable("b", rebaseline=True)
    assert mgr.trainable_expert == "b" and mgr._backbone_ref != ref
    mgr.assert_backbone_unchanged()
    with torch.no_grad():
        model.model.norm.weight[0] -= 1.0
    assert mgr.freeze_backbone(rebaseline=True) == ref

    # a restored reference (sidecar) is checked against the loaded weights, not recorded anew
    path = mgr.save_pretrained(tmp_path / "attached")
    loaded, lmgr = load_model_with_experts(path)
    lmgr.assert_backbone_unchanged()
    lmgr.set_trainable("a")
    assert lmgr._backbone_ref == ref
    with torch.no_grad():
        loaded.model.embed_tokens.weight[3, 0] += 1.0
    with pytest.raises(BackboneChangedError):
        lmgr.set_trainable("b")


def test_load_state_dict_restores_unfrozen_state_routing_and_validates(tokenizer):
    _, src = managed(tokenizer, layers=(0,))
    src.set_routing(["a", "base", "b"])
    unfrozen = src.state_dict()
    assert unfrozen["frozen"] is False and unfrozen["routing"] == ["a", "base", "b"]

    # into a manager that froze everything for "a": the manager's own freezing is undone
    model, mgr = managed(tokenizer, layers=(0,))
    mgr.set_trainable("a")
    mgr.load_state_dict(unfrozen)
    assert mgr.trainable_expert is None and mgr._frozen is False
    assert all(p.requires_grad for p in model.parameters())
    assert mgr.routing == ("a", "base", "b") and mgr.active == "base"
    mgr.create_expert("c")  # not frozen any more: nothing is re-frozen
    assert all(p.requires_grad for p in model.parameters())

    # into a fresh manager: a user's own freezing is left alone
    fresh = build_tiny_llama(len(tokenizer), seed=0)
    fresh.model.embed_tokens.weight.requires_grad_(False)
    fmgr = ExpertManager(fresh)
    fmgr.load_state_dict(unfrozen)
    assert fmgr.expert_names == ["base", "a", "b"] and fmgr.routing == ("a", "base", "b")
    assert not fresh.model.embed_tokens.weight.requires_grad
    assert all(p.requires_grad for n, p in fresh.named_parameters() if "embed_tokens" not in n)

    # a frozen state restores requires_grad exactly
    src.clear_routing()
    src.set_trainable("b")
    fmgr.load_state_dict(src.state_dict())
    assert sorted(fmgr.trainable_parameter_names()) == sorted(fmgr.expert_parameter_names("b"))

    # invalid states are rejected before anything changes
    good = src.state_dict()
    before = fmgr.state_dict()
    for bad, match in (
        (dict(good, trainable="nope"), "trainable"),
        (dict(good, trainable="base"), "trainable"),
        (dict(good, frozen=False), "trainable"),
        (dict(good, active="nope"), "unknown experts"),
        (dict(good, routing=["a", "nope"]), "unknown experts"),
        (dict(good, experts=["a", "base", "b"]), "expert list"),
        (dict(good, format="other"), "format"),
    ):
        with pytest.raises(ExpertCompatibilityError, match=match):
            fmgr.load_state_dict(bad)
        assert fmgr.state_dict() == before
    # the model has an expert the state does not know
    fmgr.create_expert("extra")
    with pytest.raises(ExpertCompatibilityError, match="does not know"):
        fmgr.load_state_dict(good)


def test_detached_manager_state_and_reset(tokenizer, tmp_path):
    _, mgr = managed(tokenizer, layers=(0,))
    mgr.set_trainable("a")
    trained = mgr.expert_parameters("a")
    mgr.detach("a")
    assert mgr.trainable_expert is None and not mgr.attached
    # still usable as a registered checkpoint component
    out = save_components(tmp_path, {"experts": mgr})
    assert out is not None
    state = torch.load(out, weights_only=True)["experts"]
    assert state["layers"] == [] and state["experts"] == []
    assert load_components(tmp_path, {"experts": ExpertManager(build_tiny_llama(len(tokenizer)))}) == [
        "experts"
    ]
    mgr.load_state_dict(state)  # detached -> detached: no-op
    assert not mgr.attached
    with pytest.raises(ExpertError, match="no experts attached"):
        mgr.save_pretrained(tmp_path / "x")

    # re-attach: no stale reference / frozen flag from the previous session
    mgr.attach([0])
    with pytest.raises(ExpertError, match="no backbone reference"):
        mgr.assert_backbone_unchanged()
    assert all(p.requires_grad for p in trained)  # the trained FFN is the new "base"
    mgr.create_expert("c")  # a stale frozen flag would re-freeze everything here
    assert all(p.requires_grad for p in trained)
    with pytest.raises(ExpertCompatibilityError, match="detached"):
        mgr.load_state_dict(state)


def test_config_hash_ignores_token_ids_and_dropout(tokenizer, tmp_path):
    model, mgr = managed(tokenizer, layers=(0,))
    good = mgr.save_expert("a", tmp_path / "a.safetensors")
    h = config_hash(model.config)
    target = build_tiny_llama(len(tokenizer), seed=0)
    target.config.pad_token_id = 5
    target.config.eos_token_id = [2, 3]
    target.config.bos_token_id = None
    target.config.attention_dropout = 0.1
    target.config.initializer_range = 0.5
    assert config_hash(target.config) == h
    tmgr = ExpertManager(target)
    tmgr.attach([0])
    assert tmgr.load_expert(good) == "a"
    assert tmgr.expert_fingerprint("a") == mgr.expert_fingerprint("a")
    for key, value in (("intermediate_size", 48), ("tie_word_embeddings", True), ("rms_norm_eps", 1e-5)):
        cfg = copy.deepcopy(model.config)
        setattr(cfg, key, value)
        assert config_hash(cfg) != h, key


def test_model_class_mismatch_is_rejected(tokenizer, tmp_path):
    _, mgr = managed(tokenizer, layers=(0,))
    good = mgr.save_expert("a", tmp_path / "a.safetensors")
    tensors, meta = ExpertManager.read_expert_file(good)
    meta["model_class"] = "transformers.models.mistral.modeling_mistral.MistralForCausalLM"
    edited = tmp_path / "edited.safetensors"
    st.save_file(tensors, str(edited), metadata=meta)  # tensors unchanged: checksum still valid
    _, tmgr = managed(tokenizer, layers=(0,), experts=())
    with pytest.raises(ExpertCompatibilityError, match="model_class"):
        tmgr.load_expert(edited)
    assert tmgr.load_expert(good) == "a"


def test_bf16_attached_checkpoint_reloads_bit_exact_in_eval_mode(tokenizer, tmp_path):
    model, mgr = managed(tokenizer)
    model.to(torch.bfloat16)
    model.generation_config.max_new_tokens = 7
    routing = ["a", "b", "base", "a"]
    mgr.set_routing(routing)
    path = mgr.save_pretrained(tmp_path / "bf16")
    assert (path / "generation_config.json").exists()

    loaded, lmgr = load_model_with_experts(path)
    assert {p.dtype for p in loaded.parameters()} == {torch.bfloat16}
    src_sd = model.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, src_sd[name]), name
    assert not loaded.training
    assert loaded.generation_config.max_new_tokens == 7
    assert lmgr.routing == tuple(routing)  # restored from the sidecar

    # Independent reference: HF's own loader on the vanilla export of each expert. (The source model's
    # ``.to(bf16)`` also cast the non-persistent RoPE ``inv_freq`` buffer, which a fresh load keeps in
    # fp32, so the source model itself is not a bit-exact logits reference.)
    ids = batch()
    refs = {}
    for name in ("base", "a", "b"):
        vanilla = LlamaForCausalLM.from_pretrained(str(mgr.export_merged(name, tmp_path / f"v_{name}")))
        assert vanilla.dtype == torch.bfloat16
        refs[name] = logits(vanilla, ids)
        with lmgr.using(name):
            assert torch.equal(logits(loaded, ids), refs[name]), name
    out = logits(loaded, ids)
    assert not torch.allclose(refs["a"], refs["b"], atol=0.05) and not torch.allclose(
        refs["a"], refs["base"], atol=0.05
    )
    for row, name in enumerate(routing):  # rows are batched per expert: close, not necessarily bit-equal
        torch.testing.assert_close(out[row], refs[name][row], rtol=1e-2, atol=1e-2)
