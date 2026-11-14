"""Tests of the committed core: config, registry, experiment records, checkpoints, masks, model interface."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from torch import nn

from research_rl.core import checkpoint as ckpt
from research_rl.core.config import (
    CONFIG_VERSION,
    ConfigError,
    DataSpec,
    ExperimentConfig,
    ModelSpec,
    OptimSpec,
    TrainerSpec,
)
from research_rl.core.experiment import (
    NOT_TESTED,
    ExperimentRecorder,
    dataset_identity,
    environment_info,
    model_identity,
)
from research_rl.core.registry import Registry, RegistryError
from research_rl.models.model_interface import (
    ModelInterfaceError,
    count_parameters,
    find_swiglu_ffns,
    fingerprint,
    get_decoder_layers,
    snapshot,
)
from research_rl.models.parameter_mask import MaskError, ParameterMask


def make_config(tmp_path: Path | None = None, **overrides) -> ExperimentConfig:
    kwargs = {
        "study_id": "study-1",
        "model": ModelSpec("tiny-llama", revision="abc123", dtype="bfloat16"),
        "data": DataSpec("gsm8k", split="test", revision="r1", notes="n"),
        "trainer": TrainerSpec(
            "grpo", args={"learning_rate": 1e-5, "num_generations": 4, "nested": {"a": [1, 2]}}
        ),
        "seed": 7,
        "output_dir": str(tmp_path) if tmp_path is not None else "runs",
        "run_name": "r",
        "optim": OptimSpec(hooked=True, clip_in_wrapper=1.0, skip_zero_advantage_steps=True),
        "tags": ["a", "b"],
        "extra": {"k": {"x": 1.5}},
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


# ================================================================================ config
@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_config_round_trip(tmp_path, suffix):
    cfg = make_config(tmp_path)
    path = cfg.save(tmp_path / "sub" / f"cfg{suffix}")
    loaded = ExperimentConfig.load(path)
    assert loaded == cfg
    assert isinstance(loaded.model, ModelSpec) and isinstance(loaded.optim, OptimSpec)
    assert loaded.to_dict()["config_version"] == CONFIG_VERSION
    if suffix == ".json":
        assert json.loads(path.read_text())["trainer"]["args"]["nested"] == {"a": [1, 2]}


def test_config_defaults_and_paths(tmp_path):
    cfg = ExperimentConfig.from_dict(
        {
            "study_id": "s",
            "model": {"name_or_path": "m"},
            "data": {"name": "d"},
            "trainer": {"algorithm": "sft"},
        }
    )
    assert cfg.optim == OptimSpec() and cfg.seed == 0
    assert cfg.resolved_run_name == "sft-seed0"
    assert cfg.run_dir == Path("runs") / "s" / "sft-seed0"
    assert make_config(tmp_path).run_dir == tmp_path / "study-1" / "r"


def test_config_version_check(tmp_path):
    raw = make_config().to_dict()
    raw["config_version"] = "0.9"
    with pytest.raises(ConfigError, match="config_version"):
        ExperimentConfig.from_dict(raw)
    p = tmp_path / "old.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        ExperimentConfig.load(p)


def test_config_rejects_unknown_and_missing_keys(tmp_path):
    raw = make_config().to_dict()
    with pytest.raises(ConfigError, match="unknown top-level"):
        ExperimentConfig.from_dict({**raw, "lr": 1.0})
    for section in ("model", "data", "trainer", "optim"):
        bad = json.loads(json.dumps(raw))
        bad[section]["typo_key"] = 1
        with pytest.raises(ConfigError):
            ExperimentConfig.from_dict(bad)
    missing = dict(raw)
    del missing["trainer"]
    with pytest.raises(ConfigError, match="missing"):
        ExperimentConfig.from_dict(missing)
    with pytest.raises(ConfigError, match="dtype"):
        ModelSpec("m", dtype="float64")
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        ExperimentConfig.load(p)


@pytest.mark.parametrize("study_id", ["", "a/b", " lead", "trail ", "/abs"])
def test_config_invalid_study_id(study_id):
    with pytest.raises(ConfigError, match="study_id"):
        make_config(study_id=study_id)


# ================================================================================ registry
def test_registry_register_lookup_and_errors():
    reg: Registry[object] = Registry("widget")

    @reg.register()
    def alpha() -> str:
        return "a"

    @reg.register("beta")
    class Beta:
        def __init__(self, x: int) -> None:
            self.x = x

    assert reg.get("alpha") is alpha and reg.build("beta", 3).x == 3
    assert reg.names() == ["alpha", "beta"] and list(reg) == ["alpha", "beta"] and len(reg) == 2
    assert "alpha" in reg and "gamma" not in reg
    with pytest.raises(RegistryError, match="already registered"):
        reg.add("alpha", lambda: None)
    reg.add("alpha", alpha)  # re-registering the same object is idempotent
    reg.add("alpha", Beta, overwrite=True)
    assert reg.get("alpha") is Beta
    with pytest.raises(RegistryError, match=r"unknown widget 'gamma'; known: alpha, beta"):
        reg.get("gamma")
    with pytest.raises(KeyError):  # RegistryError is a KeyError
        reg.get("gamma")
    reg.add("const", 5)
    with pytest.raises(RegistryError, match="not callable"):
        reg.build("const")
    with pytest.raises(RegistryError, match="cannot infer"):
        reg.register()(7)
    assert Registry("empty").names() == []
    with pytest.raises(RegistryError, match="<none>"):
        Registry("empty").get("x")


# ================================================================================ experiment records
ROWS = [{"prompt": "1+1=", "answer": "2"}, {"prompt": "2+2=", "answer": "4"}, {"prompt": "é", "answer": "x"}]


def reference_rows_hash(rows) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def test_dataset_identity_hash_is_stable_and_content_sensitive():
    ident = dataset_identity(ROWS, name="toy")
    assert ident["num_rows"] == 3 and ident["name"] == "toy"
    assert ident["content_sha256"] == reference_rows_hash(ROWS)
    reordered_keys = [{"answer": r["answer"], "prompt": r["prompt"]} for r in ROWS]
    assert dataset_identity(reordered_keys)["content_sha256"] == ident["content_sha256"]

    hf = Dataset.from_list(ROWS)
    hf_ident = dataset_identity(hf)
    assert hf_ident["content_sha256"] == ident["content_sha256"]  # same rows -> same hash
    assert hf_ident["columns"] == ["prompt", "answer"] and "hf_fingerprint" in hf_ident
    assert dataset_identity(Dataset.from_list(ROWS))["content_sha256"] == hf_ident["content_sha256"]

    changed = [dict(r) for r in ROWS]
    changed[1]["answer"] = "5"
    assert dataset_identity(changed)["content_sha256"] != ident["content_sha256"]
    assert dataset_identity(ROWS[::-1])["content_sha256"] != ident["content_sha256"]  # order matters
    assert dataset_identity(ROWS[:2])["content_sha256"] != ident["content_sha256"]

    big = dataset_identity(ROWS, max_rows_for_hash=2)
    assert big["content_sha256"] is None and big["num_rows"] == 3
    unsized = dataset_identity(iter(ROWS))
    assert unsized["num_rows"] is None and unsized["content_sha256"] is None


def reference_fingerprint(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        t = p.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def test_model_identity_weights_hash_and_counts(tiny_model):
    ident = model_identity(tiny_model, revision="rev")
    manual_count = sum(p.numel() for p in tiny_model.parameters())
    assert ident["num_parameters"] == manual_count == ident["num_trainable_parameters"]
    assert ident["weights_sha256"] == reference_fingerprint(tiny_model)
    assert ident["class"] == "LlamaForCausalLM" and ident["requested_revision"] == "rev"
    tiny_model.lm_head.weight.requires_grad_(False)
    ident2 = model_identity(tiny_model, hash_weights=False)
    assert ident2["weights_sha256"] is None
    assert ident2["num_trainable_parameters"] == manual_count - tiny_model.lm_head.weight.numel()
    with torch.no_grad():
        tiny_model.model.norm.weight[0] += 1e-3
    assert model_identity(tiny_model)["weights_sha256"] != ident["weights_sha256"]


def test_recorder_metrics_finalize_and_files(tmp_path, tiny_model):
    cfg = make_config(tmp_path)
    rec = ExperimentRecorder(cfg)
    assert rec.run_dir == tmp_path / "study-1" / "r"
    assert ExperimentConfig.load(rec.run_dir / "config.json") == cfg
    rec.set_model(tiny_model)
    rec.set_dataset(ROWS)
    rec.log_metrics(1, {"loss": 2.0, "reward": 0.1, "num_tokens": 100, "rewards/acc": 0.2, "other": 9.0})
    rec.log_metrics(2, {"tensor_metric": torch.tensor(1.5), "reward": 0.3, "eval_loss": float("nan")})
    rec.log_metrics(
        3, {"loss": 1.25, "grad_norm": 0.7, "learning_rate": 1e-5, "num_tokens": 260, "flag": True}
    )
    rec.set_steps(nominal=3, applied=2, skipped=1, global_step=3)
    rec.add_checkpoint(tmp_path / "checkpoint-3", 3)
    rec.add_checkpoint(tmp_path / "checkpoint-3", 3)  # deduplicated
    rec.add_evaluation("probe", {"acc": torch.tensor(0.5), "per": torch.tensor([1.0, 0.0])}, step=3)

    lines = [json.loads(line) for line in (rec.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [line["step"] for line in lines] == [1, 2, 3]
    assert lines[1]["tensor_metric"] == 1.5 and lines[1]["eval_loss"] == "nan"  # tensors/NaN made JSON-safe
    assert rec.history("loss") == [2.0, 1.25]
    assert rec.history("flag") == []  # bools are not metrics

    final = rec.finalize()
    on_disk = ExperimentRecorder.load(rec.run_dir)
    assert final["final_metrics"]["loss"] == 1.25 and final["status"] == "completed"
    assert on_disk["final_metrics"] == {
        "loss": 1.25,
        "reward": 0.3,
        "grad_norm": 0.7,
        "num_tokens": 260.0,
        "learning_rate": 1e-5,
        "rewards/acc": 0.2,
        "eval_loss": "nan",  # NaN is stored as a string by _jsonable
    }
    assert on_disk["status"] == "completed" and on_disk["training_tokens"] == 260
    assert on_disk["steps"] == {
        "nominal_optimizer_steps": 3,
        "applied_updates": 2,
        "skipped_steps": 1,
        "global_step": 3,
    }
    assert on_disk["checkpoints"] == [{"path": str(tmp_path / "checkpoint-3"), "global_step": 3}]
    assert on_disk["evaluations"] == [
        {"name": "probe", "step": 3, "results": {"acc": 0.5, "per": [1.0, 0.0]}}
    ]
    assert on_disk["model"]["weights_sha256"] == reference_fingerprint(tiny_model)
    assert on_disk["trainable_parameters"] == count_parameters(tiny_model, trainable_only=True)
    assert on_disk["dataset"]["content_sha256"] == reference_rows_hash(ROWS)
    assert on_disk["dataset"]["name"] == "gsm8k"  # falls back to the configured data name
    assert on_disk["seed"] == 7 and on_disk["algorithm"] == "grpo"


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU refusal path")
def test_record_gpu_metrics_refused_on_cpu(tmp_path):
    rec = ExperimentRecorder(make_config(tmp_path))
    with pytest.raises(RuntimeError, match="NOT_TESTED"):
        rec.record_gpu_metrics({"max_memory_allocated": 123})
    assert rec.record["hardware"]["gpu_metrics"] == NOT_TESTED
    assert ExperimentRecorder.load(rec.run_dir)["hardware"]["gpu_metrics"] == "NOT_TESTED"
    rec.finalize("failed")
    assert ExperimentRecorder.load(rec.run_dir)["hardware"]["gpu_metrics"] == "NOT_TESTED"
    assert environment_info()["device"] == "cpu" and environment_info()["cuda_device_name"] is None


# ================================================================================ checkpoint helpers
def test_research_state_round_trip_and_version_mismatch(tmp_path):
    assert ckpt.load_research_state(tmp_path / "none") == {}
    state = {"policy_version": 5, "data_progress": {"epoch": 1, "index": 17}, "algo": {"beta": 0.1}}
    out = ckpt.save_research_state(tmp_path / "c1", state)
    assert out.name == ckpt.RESEARCH_STATE_FILE
    loaded = ckpt.load_research_state(tmp_path / "c1")
    assert loaded == {"config_version": CONFIG_VERSION, **state}
    raw = json.loads(out.read_text())
    raw["config_version"] = "0.1"
    out.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="config_version"):
        ckpt.load_research_state(tmp_path / "c1")


class TensorComponent:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.extra: object = 3

    def state_dict(self) -> dict:
        return {"value": self.value.clone(), "extra": self.extra}

    def load_state_dict(self, state) -> None:
        self.value = state["value"].clone()
        self.extra = state["extra"]


class Opaque:
    """A custom class: allowed by plain pickle, refused by torch.load(weights_only=True)."""


def test_components_round_trip_and_weights_only(tmp_path):
    assert ckpt.save_components(tmp_path, {}) is None
    a = TensorComponent(torch.arange(6.0).reshape(2, 3))
    b = TensorComponent(torch.tensor([1, 2], dtype=torch.int64))
    path = ckpt.save_components(tmp_path, {"a": a, "b": b})
    assert path is not None and path.name == ckpt.COMPONENTS_FILE
    a2, b2, c2 = (
        TensorComponent(torch.zeros(1)),
        TensorComponent(torch.zeros(1)),
        TensorComponent(torch.ones(1)),
    )
    restored = ckpt.load_components(tmp_path, {"a": a2, "b": b2, "c": c2})
    assert restored == ["a", "b"]
    assert torch.equal(a2.value, a.value) and torch.equal(b2.value, b.value) and b2.value.dtype == torch.int64
    assert torch.equal(c2.value, torch.ones(1))  # not in the file: untouched
    assert ckpt.load_components(tmp_path / "missing", {"a": a2}) == []
    assert isinstance(a, ckpt.StatefulComponent)

    evil = TensorComponent(torch.zeros(1))
    evil.extra = Opaque()
    (tmp_path / "e").mkdir()
    ckpt.save_components(tmp_path / "e", {"x": evil})
    with pytest.raises(pickle.UnpicklingError, match="Weights only"):
        ckpt.load_components(tmp_path / "e", {"x": TensorComponent(torch.zeros(1))})


def test_checkpoint_inventory_and_latest(tmp_path):
    inv = ckpt.checkpoint_inventory(tmp_path)
    assert inv == {
        "optimizer": False,
        "scheduler": False,
        "rng_state": False,
        "trainer_state": False,
        "model_weights": False,
        "research_state": False,
        "components": False,
    }
    for fname in ("optimizer.pt", "trainer_state.json", "adapter_model.safetensors"):
        (tmp_path / fname).write_text("x")
    ckpt.save_research_state(tmp_path, {})
    inv = ckpt.checkpoint_inventory(tmp_path)
    assert inv["optimizer"] and inv["trainer_state"] and inv["model_weights"] and inv["research_state"]
    assert not inv["scheduler"] and not inv["rng_state"] and not inv["components"]

    assert ckpt.latest_checkpoint(tmp_path / "nope") is None
    out = tmp_path / "out"
    assert ckpt.latest_checkpoint(out) is None
    for step in (2, 10, 9):
        (out / f"checkpoint-{step}").mkdir(parents=True)
    assert ckpt.latest_checkpoint(out) == str(out / "checkpoint-10")


# ================================================================================ parameter masks
def two_layer() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))


def test_parameter_mask_algebra_matches_boolean_reference():
    g = torch.Generator().manual_seed(0)
    a0, b0 = torch.rand(3, 4, generator=g) > 0.5, torch.rand(3, 4, generator=g) > 0.5
    a1 = torch.rand(2, 3, generator=g) > 0.5
    b2 = torch.rand(2, generator=g) > 0.5
    A = ParameterMask({"0.weight": a0, "1.weight": a1})
    B = ParameterMask({"0.weight": b0, "1.bias": b2})
    uni, inter = A.union(B), A.intersection(B)
    assert torch.equal(uni.get("0.weight"), a0 | b0)
    assert torch.equal(inter.get("0.weight"), a0 & b0)
    # absent = unconstrained (all True): intersection keeps the one-sided mask
    assert torch.equal(inter.get("1.weight"), a1) and torch.equal(inter.get("1.bias"), b2)
    assert sorted(inter.names()) == ["0.weight", "1.bias", "1.weight"]
    assert torch.equal(A.get("0.weight"), a0) and "1.bias" not in A  # inputs not mutated
    assert A.num_trainable() == int(a0.sum() + a1.sum())
    assert A.num_frozen() == a0.numel() + a1.numel() - A.num_trainable()
    assert len(A) == 2 and "0.weight" in A and A.get("nope") is None
    with pytest.raises(MaskError, match="bool"):
        ParameterMask({"x": torch.ones(2)})
    back = ParameterMask.from_state_dict(A.state_dict())
    assert all(torch.equal(back.get(n), m) for n, m in A.items())


def test_parameter_mask_validate_restore_and_changed_outside():
    model = two_layer()
    params = dict(model.named_parameters())
    frozen = ParameterMask.all_frozen(model, ["0.weight"])
    assert frozen.num_trainable() == 0 and frozen.num_frozen() == 12
    assert ParameterMask.all_trainable(model, ["0.bias"]).num_frozen() == 0
    with pytest.raises(MaskError, match="unknown parameter"):
        ParameterMask({"2.weight": torch.ones(1, dtype=torch.bool)}).validate(model)
    with pytest.raises(MaskError, match="shape"):
        ParameterMask({"0.weight": torch.ones(4, 3, dtype=torch.bool)}).validate(model)

    m = torch.zeros(3, 4, dtype=torch.bool)
    m[0, 1] = m[2, 3] = True
    mask = ParameterMask({"0.weight": m})
    mask.validate(model)
    before = snapshot(model)
    with torch.no_grad():
        params["0.weight"].add_(1.0)
        params["0.bias"].add_(1.0)  # not in mask: unconstrained
    after = snapshot(model)
    assert mask.changed_outside(before, after) == {"0.weight": 10}
    mask.restore_frozen(model, before)
    expected = torch.where(m, before["0.weight"] + 1.0, before["0.weight"])
    assert torch.equal(params["0.weight"], expected)
    assert torch.equal(params["0.bias"], before["0.bias"] + 1.0)
    assert mask.changed_outside(before, snapshot(model)) == {"0.weight": 0}


# ================================================================================ model interface
def test_find_swiglu_ffns_on_tiny_llama(tiny_model):
    refs = find_swiglu_ffns(tiny_model)
    assert [r.layer_index for r in refs] == [0, 1]
    assert [r.module_path for r in refs] == ["model.layers.0.mlp", "model.layers.1.mlp"]
    params = dict(tiny_model.named_parameters())
    for r in refs:
        assert r.module is tiny_model.model.layers[r.layer_index].mlp
        assert r.intermediate_size == 64
        assert tuple(r.gate.weight.shape) == (64, 32) and tuple(r.up.weight.shape) == (64, 32)
        assert tuple(r.down.weight.shape) == (32, 64)
        names = r.param_names()
        assert names["down_proj"] == f"model.layers.{r.layer_index}.mlp.down_proj.weight"
        assert params[names["gate_proj"]] is r.gate.weight and params[names["down_proj"]] is r.down.weight
    assert [r.layer_index for r in find_swiglu_ffns(tiny_model, layers=[1])] == [1]
    with pytest.raises(ModelInterfaceError, match=r"out of range: \[2, 5\]"):
        find_swiglu_ffns(tiny_model, layers=[0, 2, 5])
    with pytest.raises(ModelInterfaceError, match="out of range"):
        find_swiglu_ffns(tiny_model, layers=[-1])
    assert get_decoder_layers(tiny_model)[0] == "model.layers"
    with pytest.raises(ModelInterfaceError, match="cannot locate"):
        find_swiglu_ffns(two_layer())


def test_find_swiglu_ffns_rejects_non_swiglu_layer(tiny_model):
    tiny_model.model.layers[1].mlp = nn.Sequential(nn.Linear(32, 32))
    with pytest.raises(ModelInterfaceError, match="layer 1 has no SwiGLU"):
        find_swiglu_ffns(tiny_model)
    assert len(find_swiglu_ffns(tiny_model, layers=[0])) == 1


def test_fingerprint_reference_and_one_ulp_sensitivity(tiny_model):
    fp = fingerprint(tiny_model)
    assert fp == reference_fingerprint(tiny_model)
    assert fingerprint(snapshot(tiny_model)) == fp
    w = tiny_model.model.layers[1].mlp.up_proj.weight
    old = w.detach().clone()
    with torch.no_grad():
        w[3, 7] = torch.nextafter(w[3, 7], torch.tensor(float("inf")))
    assert not torch.equal(w, old) and (w - old).abs().max() < 1e-6  # really just one ulp
    assert fingerprint(tiny_model) != fp
    with torch.no_grad():
        w.copy_(old)
    assert fingerprint(tiny_model) == fp
    up_name = "model.layers.1.mlp.up_proj.weight"
    sub = fingerprint(tiny_model, names=[up_name])
    assert sub == fingerprint({up_name: old, "unrelated": torch.zeros(1)}, names=[up_name])
    assert sub != fingerprint({up_name: old.double()})  # dtype is part of the identity
    assert sub != fingerprint({"renamed": old})  # name is part of the identity
    snap = snapshot(tiny_model, [up_name])
    with torch.no_grad():
        w.add_(1.0)
    assert torch.equal(snap[up_name], old)  # snapshot is a copy
    assert fingerprint({"e": torch.zeros(0)}) == fingerprint({"e": torch.zeros(0)})
    assert fingerprint({"b": torch.ones(2, dtype=torch.bfloat16)}) != fingerprint({"b": torch.ones(2)})


class ScalarParamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 0-dim parameter
        self.proj = nn.Linear(2, 2)


def test_fingerprint_handles_scalar_parameters():
    model = ScalarParamModel()
    assert len(fingerprint(model)) == 64


def test_model_identity_handles_scalar_parameters():
    assert model_identity(ScalarParamModel())["weights_sha256"] is not None


# ----------------------------------------------------------------------------- regressions for core review fixes
def test_union_treats_missing_name_as_unconstrained():
    a = ParameterMask({"w": torch.tensor([True, False]), "v": torch.tensor([False])})
    b = ParameterMask({"w": torch.tensor([False, False])})
    u = a.union(b)
    assert torch.equal(u.get("w"), torch.tensor([True, False]))
    assert "v" not in u  # unconstrained in b -> unconstrained in the union
    assert len(a.union(ParameterMask())) == 0


@pytest.mark.parametrize("bad", ["..", ".hidden", "a/b", "../../evil", "a..b", "", " x", "x\\y"])
def test_study_id_and_run_name_cannot_escape_output_dir(bad):
    base = dict(
        model=ModelSpec(name_or_path="m"), data=DataSpec(name="d"), trainer=TrainerSpec(algorithm="sft")
    )
    with pytest.raises(ConfigError):
        ExperimentConfig(study_id=bad, **base)
    with pytest.raises(ConfigError):
        ExperimentConfig(study_id="ok", run_name=bad, **base)


def test_recorder_history_accepts_scalar_tensors(tmp_path):
    cfg = ExperimentConfig(
        study_id="s",
        model=ModelSpec(name_or_path="m"),
        data=DataSpec(name="d"),
        trainer=TrainerSpec(algorithm="sft"),
        output_dir=str(tmp_path),
    )
    rec = ExperimentRecorder(cfg)
    rec.log_metrics(1, {"loss": torch.tensor(1.5)})
    assert rec.finalize()["final_metrics"]["loss"] == 1.5
