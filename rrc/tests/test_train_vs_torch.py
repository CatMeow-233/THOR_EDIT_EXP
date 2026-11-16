"""rrc_train (rrc/src/train.c, rrc/src/data.c) against PyTorch + transformers + PEFT on tiny models.

Order (each group builds on the previous one):
  a. optimizer semantics: the LR sequence against transformers.get_scheduler, clipping against
     torch.nn.utils.clip_grad_norm_, and an N-step trajectory (gradient accumulation, active clipping,
     warmup + cosine, weight decay, an epoch boundary) against a torch.optim.AdamW + PEFT loop fed the
     batches rrc trained on (rrc/tools/torch_reference.py); fp32 tight, bf16 anchored to HF's own error;
  b. bit-identical resume (adapter, optimizer state, metrics) with the prefetch thread on;
  c. the saved checkpoint loads with peft.PeftModel.from_pretrained and gives rrc's logits;
  d. data: prepare_sft.py round trip with a tokenizer trained in the test, every malformed .rrcbin,
     shuffling / packing determinism (repeated runs, prefetch on / off) and the packing spec.
Nothing is downloaded (HF_HUB_OFFLINE=1); tiny models, adapters and tokenizers live in tmp dirs.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import re
import struct
import subprocess
import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
safetensors_torch = pytest.importorskip("safetensors.torch")
import make_tiny_model as mtm  # noqa: E402  (rrc/tools is put on sys.path by conftest)
import prepare_sft  # noqa: E402
import torch_reference as tref  # noqa: E402
from safetensors import safe_open  # noqa: E402

RRC_DIR = pathlib.Path(__file__).resolve().parents[1]
VOCAB = 128
MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --------------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def train_exe(rrc_build: pathlib.Path) -> pathlib.Path:
    exe = rrc_build / "rrc_train"
    assert exe.is_file()
    return exe


@pytest.fixture(scope="session")
def data_exe(rrc_build: pathlib.Path) -> pathlib.Path:
    exe = rrc_build / "rrc_data"
    assert exe.is_file()
    return exe


@pytest.fixture(scope="session")
def fwdbwd_exe(rrc_build: pathlib.Path) -> pathlib.Path:
    exe = rrc_build / "rrc_fwdbwd"
    assert exe.is_file()
    return exe


_MODELS: dict[str, pathlib.Path] = {}


@pytest.fixture(scope="session")
def tiny(tmp_path_factory: pytest.TempPathFactory):
    def get(name: str) -> pathlib.Path:
        if name not in _MODELS:
            out = tmp_path_factory.mktemp(f"tiny_{name}")
            mtm.make_variant(name, str(out), seed=1)
            _MODELS[name] = out
        return _MODELS[name]

    return get


def make_docs(n: int, seed: int, lmin: int = 3, lmax: int = 20, long_every: int = 0) -> list:
    """n random documents (tokens, mask): a prompt prefix with mask 0, the rest 1. Every
    `long_every`-th document is 45 tokens long (longer than the seq_len 32 used below)."""
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        ln = 45 if long_every and i % long_every == long_every - 1 else rng.randint(lmin, lmax)
        p = rng.randint(1, ln - 1)
        docs.append(([rng.randrange(VOCAB) for _ in range(ln)], [0] * p + [1] * (ln - p)))
    return docs


def make_init_adapter(mdir, out, r=4, alpha=8.0, targets=MODS, rslora=False, layers=None, seed=5):
    """A PEFT adapter with randomized, non-zero A and B (B = 0 would leave dA = 0 on step 1)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32)
    pm = get_peft_model(
        base,
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=list(targets),
            lora_dropout=0.0,
            use_rslora=rslora,
            layers_to_transform=layers,
        ),
    )
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_A" in n:
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) / p.shape[1] ** 0.5)
            elif "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    pm.save_pretrained(str(out))
    return pathlib.Path(out)


BASE_CFG: dict[str, Any] = dict(
    dtype="f32",
    seq_len=32,
    micro_batch_size=2,
    grad_accum_steps=2,
    max_steps=8,
    learning_rate=1e-2,
    lr_scheduler_type="cosine",
    warmup_steps=2,
    max_grad_norm=0.55,
    weight_decay=0.01,
    seed=3,
    lora=dict(r=4, alpha=8, target_modules="all-linear"),
    attn_chunk=7,
    ce_chunk=5,
)


def write_cfg(path: pathlib.Path, **kw) -> pathlib.Path:
    cfg = json.loads(json.dumps(BASE_CFG))
    for k, v in kw.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    path.write_text(json.dumps(cfg, indent=1))
    return path


def run_train(exe, cfg_path, expect_ok=True, extra=()) -> subprocess.CompletedProcess:
    p = subprocess.run([str(exe), "--config", str(cfg_path), *extra], capture_output=True, text=True)
    if expect_ok:
        assert p.returncode == 0, f"rrc_train failed ({p.returncode}): {p.stderr}"
    return p


def read_metrics(out_dir: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]


def load_windows(out_dir: pathlib.Path, n: int) -> list[dict[str, Any]]:
    return [
        safetensors_torch.load_file(str(out_dir / f"batches/step-{s:06d}.safetensors"))
        for s in range(1, n + 1)
    ]


def rrc_order(names) -> list[str]:
    """PEFT names in rrc's buffer layout order (layer, module q..down, A before B)."""

    def key(n: str):
        layer = int(n.split(".layers.")[1].split(".")[0])
        mod = next(i for i, m in enumerate(MODS) if f".{m}." in n)
        return (layer, mod, 0 if ".lora_A." in n else 1)

    return sorted(names, key=key)


def read_state(path: pathlib.Path, adapter: dict[str, Any]) -> dict[str, Any]:
    """Parse rrc_state.bin (layout: rrc/include/rrc/train.h); m / v split into PEFT-named tensors."""
    data = path.read_bytes()
    assert data[:8] == b"RRCSTAT1"
    f = struct.unpack_from("<11Q", data, 8)
    version, step, n, seed, epoch, index, mbs, tokens, phash, dhash, cfg_len = f
    off = 96 + cfg_len
    flat_m = torch.frombuffer(bytearray(data[off : off + 4 * n]), dtype=torch.float32)
    flat_v = torch.frombuffer(bytearray(data[off + 4 * n : off + 8 * n]), dtype=torch.float32)
    assert off + 8 * n + 8 == len(data)
    m, v, o = {}, {}, 0
    for name in rrc_order(adapter):
        k = adapter[name].numel()
        m[name] = flat_m[o : o + k].reshape(adapter[name].shape)
        v[name] = flat_v[o : o + k].reshape(adapter[name].shape)
        o += k
    assert o == n
    return dict(
        version=version,
        step=step,
        seed=seed,
        epoch=epoch,
        index=index,
        micro_batches=mbs,
        tokens_seen=tokens,
        params_hash=phash,
        data_hash=dhash,
        config=data[96 : 96 + cfg_len].decode(),
        m=m,
        v=v,
    )


def torch_clip_coef(norm: float, max_norm: float) -> float:
    """torch.nn.utils.clip_grad_norm_'s coefficient for an f32 total norm (exactly as torch computes it)."""
    if max_norm <= 0:
        return 1.0
    return torch.clamp(max_norm / (torch.tensor(norm, dtype=torch.float32) + 1e-6), max=1.0).item()


def rel_err(a, b) -> float:
    a, b = a.double(), b.double()
    nb = b.norm().item()
    return (a - b).norm().item() / nb if nb > 0 else (a - b).norm().item()


@pytest.fixture(scope="module")
def setup_fp32(tiny, tmp_path_factory):
    """Data (30 short documents: the 8-step run crosses two epoch boundaries) + init adapter."""
    d = tmp_path_factory.mktemp("train_fp32")
    mdir = tiny("llama")
    prepare_sft.write_rrcbin(str(d / "data.rrcbin"), make_docs(30, seed=0), VOCAB)
    make_init_adapter(mdir, d / "init")
    return dict(dir=d, model=mdir, data=d / "data.rrcbin", init=d / "init")


# --------------------------------------------------------------------------------------------
# a. LR schedule, clipping, trajectory
# --------------------------------------------------------------------------------------------

SCHED_CASES = [
    # (type, warmup_steps, max_steps); transformers 5: warmup_steps < 1 is a ratio of max_steps
    ("constant", 0, 7),
    ("constant", 3, 7),  # transformers' "constant" ignores warmup
    ("constant_with_warmup", 3, 9),
    ("constant_with_warmup", 0, 4),
    ("linear", 0, 10),
    ("linear", 4, 10),
    ("linear", 12, 10),  # warmup >= total
    ("linear", 0, 1),  # total 1
    ("linear", 0.25, 9),  # ratio -> ceil(2.25) = 3
    ("linear", 2.7, 9),  # int(2.7) = 2
    ("cosine", 0, 13),
    ("cosine", 5, 13),
    ("cosine", 13, 13),  # warmup == total
    ("cosine", 1, 1),
    ("cosine", 0.1, 37),
]


@pytest.mark.parametrize("sched,warmup,total", SCHED_CASES)
def test_lr_schedule_matches_transformers(sched, warmup, total, train_exe, tmp_path):
    from transformers import TrainingArguments, get_scheduler

    lr = 3e-4
    cfg = write_cfg(
        tmp_path / "c.json",
        model_dir="unused",
        data_path="unused",
        output_dir=str(tmp_path / "o"),
        learning_rate=lr,
        lr_scheduler_type=sched,
        warmup_steps=warmup,
        max_steps=total,
    )
    p = run_train(train_exe, cfg, extra=["--print-schedule"])
    got = json.loads(p.stdout)
    targs = TrainingArguments(
        output_dir=str(tmp_path / "hf"), warmup_steps=warmup, max_steps=total, report_to=[]
    )
    hf_warmup = targs.get_warmup_steps(total)
    assert got["warmup_steps"] == hf_warmup
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    s = get_scheduler(sched, opt, num_warmup_steps=hf_warmup, num_training_steps=total)
    want = []
    for _ in range(total):
        want.append(s.get_last_lr()[0])
        opt.step()
        s.step()
    assert got["lr"] == want  # exact (both evaluate the same double formulas, cos == math.cos)


@pytest.mark.parametrize("max_norm", [0.0, 0.2, 0.55, 5.0], ids=["off", "clip_hard", "clip_some", "no_clip"])
def test_clipping_matches_torch_clip_grad_norm(max_norm, setup_fp32, train_exe, tmp_path):
    """One step with a constant LR: grad_norm and the clip coefficient of the metrics, and the AdamW
    first moment m = (1 - beta1) * clipped grad from rrc_state.bin, against clip_grad_norm_."""
    from transformers import AutoModelForCausalLM

    s = setup_fp32
    cfg = write_cfg(
        tmp_path / "c.json",
        model_dir=str(s["model"]),
        data_path=str(s["data"]),
        output_dir=str(tmp_path / "o"),
        init_adapter=str(s["init"]),
        max_steps=1,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_grad_norm=max_norm,
        dump_batches=True,
    )
    run_train(train_exe, cfg)
    met = read_metrics(tmp_path / "o")[0]
    w = load_windows(tmp_path / "o", 1)[0]
    base = AutoModelForCausalLM.from_pretrained(s["model"], dtype=torch.float32, attn_implementation="eager")
    pm = peft.PeftModel.from_pretrained(base, str(s["init"]), is_trainable=True)
    params = [p for p in pm.parameters() if p.requires_grad]
    docs = [tref.split_documents(w["tokens"][m], w["labels"][m], w["doc_ids"][m]) for m in range(2)]
    total = sum(int((lab[1:] != -100).sum()) for d in docs for _, lab in d)
    for d in docs:
        for ids, lab in d:
            if int((lab[1:] != -100).sum()):
                lg = pm(input_ids=ids[None]).logits[0]
                (torch.nn.functional.cross_entropy(lg[:-1], lab[1:], reduction="sum") / total).backward()
    if max_norm > 0:
        tn = torch.nn.utils.clip_grad_norm_(params, max_norm)
        coef = torch.clamp(max_norm / (tn + 1e-6), max=1.0).item()
    else:
        tn = torch.nn.utils.get_total_norm([p.grad for p in params])
        coef = 1.0
    assert abs(met["grad_norm"] - tn.item()) <= 2e-6 * tn.item()
    assert abs(met["clip_coef"] - coef) <= 2e-6
    # the coefficient formula itself, bit for bit (eps 1e-6, reciprocal * max_norm, clamp) on rrc's norm
    assert met["clip_coef"] == torch_clip_coef(met["grad_norm"], max_norm)
    if max_norm == 0.2:
        assert met["clip_coef"] < 0.5
    if max_norm == 5.0 or max_norm == 0.0:
        assert met["clip_coef"] == 1.0
    adapter = safetensors_torch.load_file(str(tmp_path / "o/checkpoint-1/adapter_model.safetensors"))
    st = read_state(tmp_path / "o/checkpoint-1/rrc_state.bin", adapter)
    grads = {n.replace(".default", ""): p.grad for n, p in pm.named_parameters() if p.requires_grad}
    for k, g in grads.items():
        assert rel_err(st["m"][k], 0.1 * g) < 1e-5, k


# (backend, loss_reduction) -> tolerances (loss, grad_norm, params, m / v), each about 10x the
# measured maximum relative error (printed by the test; DESIGN.md section 10 item 25)
TRAJ_CASES = {
    ("cpu", "window_tokens"): (1e-6, 3e-6, 5e-6, 2e-5),
    ("cpu-f32acc", "window_tokens"): (1e-6, 1e-5, 5e-6, 2e-5),
    ("cpu", "microbatch_mean"): (1e-6, 3e-6, 5e-6, 2e-5),
}


@pytest.mark.parametrize("backend,reduction", list(TRAJ_CASES), ids=["-".join(c) for c in TRAJ_CASES])
def test_trajectory_fp32_matches_torch_peft(backend, reduction, setup_fp32, train_exe, tmp_path):
    s = setup_fp32
    out = tmp_path / "o"
    cfg = write_cfg(
        tmp_path / "c.json",
        model_dir=str(s["model"]),
        data_path=str(s["data"]),
        output_dir=str(out),
        init_adapter=str(s["init"]),
        dump_batches=True,
        backend=backend,
        loss_reduction=reduction,
    )
    run_train(train_exe, cfg)
    met = read_metrics(out)
    N = BASE_CFG["max_steps"]
    assert [m["step"] for m in met] == list(range(1, N + 1))
    windows = load_windows(out, N)
    # the batches rrc trained on follow the documented packing spec (independent Python implementation)
    _, docs = prepare_sft.read_rrcbin(str(s["data"]))
    spec, cursors = tref.pack_reference(docs, 32, 2, 2 * N, seed=BASE_CFG["seed"])
    for step, w in enumerate(windows):
        for k in ("tokens", "labels", "position_ids", "doc_ids"):
            assert w[k].tolist() == spec[k][2 * step : 2 * step + 2], (step, k)
    assert cursors[-1][0] >= 2 and met[-1]["epoch"] == cursors[-1][0]  # crossed two epoch boundaries
    ref = tref.train_reference(
        str(s["model"]),
        str(s["init"]),
        windows,
        tref.RefConfig(
            learning_rate=BASE_CFG["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_steps=2,
            max_steps=N,
            max_grad_norm=BASE_CFG["max_grad_norm"],
            weight_decay=BASE_CFG["weight_decay"],
            loss_reduction=reduction,
        ),
    )
    # clipping active on some steps and not on others; warmup starts at lr 0
    clipped = [m["clip_coef"] < 1.0 for m in met]
    assert any(clipped) and not all(clipped), [m["grad_norm"] for m in met]
    assert met[0]["lr"] == 0.0
    assert [m["lr"] for m in met] == ref["lr"]
    assert all(m["clip_coef"] == torch_clip_coef(m["grad_norm"], BASE_CFG["max_grad_norm"]) for m in met)
    assert [m["tokens"] for m in met] == ref["tokens"]
    loss_err = max(abs(m["loss"] - r) / r for m, r in zip(met, ref["loss"]))
    norm_err = max(abs(m["grad_norm"] - r) / r for m, r in zip(met, ref["grad_norm"]))
    final = safetensors_torch.load_file(str(out / f"checkpoint-{N}/adapter_model.safetensors"))
    assert set(final) == set(ref["params"])
    init = safetensors_torch.load_file(str(s["init"] / "adapter_model.safetensors"))
    perr = {k: rel_err(final[k], v) for k, v in ref["params"].items()}
    moved = min(rel_err(ref["params"][k], init[k]) for k in init)
    st = read_state(out / f"checkpoint-{N}/rrc_state.bin", final)
    merr = max(rel_err(st["m"][k], ref["m"][k]) for k in final)
    verr = max(rel_err(st["v"][k], ref["v"][k]) for k in final)
    print(
        f"fp32 trajectory {backend} {reduction}: max rel loss err {loss_err:.3g}, grad_norm {norm_err:.3g}, "
        f"params {max(perr.values()):.3g}, m {merr:.3g}, v {verr:.3g}, min param movement {moved:.3g}, "
        f"clipped {clipped}"
    )
    tol_loss, tol_norm, tol_param, tol_mv = TRAJ_CASES[(backend, reduction)]
    assert loss_err < tol_loss
    assert norm_err < tol_norm
    assert max(perr.values()) < tol_param
    assert merr < tol_mv and verr < tol_mv
    assert moved > 1e-3  # the parameters really moved (the comparison is not trivially satisfied)
    assert st["step"] == N and st["tokens_seen"] == sum(ref["tokens"])


def make_edge_docs(n: int, seed: int, seq_len: int) -> list:
    """Documents with every packing edge: length 1 (never a target), longer than seq_len (cut or
    dropped), exactly seq_len and seq_len + 1, and ordinary ones with a prompt prefix."""
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        ln = {3: 1, 4: 1, 8: seq_len + 7, 11: seq_len, 15: seq_len + 1, 17: 1}.get(i % 19, rng.randint(2, 9))
        p = rng.randint(0, ln - 1)
        docs.append(([rng.randrange(VOCAB) for _ in range(ln)], [0] * p + [1] * (ln - p)))
    return docs


def run_and_compare_trajectory(tiny, train_exe, tmp_path, docs, cfg_kw, ref_kw, tol, init_kw=None):
    """rrc_train vs torch.optim.AdamW + PEFT on the windows rrc dumped (which are checked against the
    Python packing spec first). Returns (metrics, reference, errors)."""
    mdir = tiny("llama")
    data = tmp_path / "data.rrcbin"
    prepare_sft.write_rrcbin(str(data), docs, VOCAB)
    init = make_init_adapter(mdir, tmp_path / "init", **(init_kw or {}))
    out = tmp_path / "o"
    cfg = write_cfg(
        tmp_path / "c.json",
        model_dir=str(mdir),
        data_path=str(data),
        output_dir=str(out),
        init_adapter=str(init),
        dump_batches=True,
        **cfg_kw,
    )
    run_train(train_exe, cfg)
    c = json.loads(cfg.read_text())
    N, M, B, T = c["max_steps"], c["grad_accum_steps"], c["micro_batch_size"], c["seq_len"]
    met = read_metrics(out)
    assert [m["step"] for m in met] == list(range(1, N + 1))
    windows = load_windows(out, N)
    spec, _ = tref.pack_reference(
        docs,
        T,
        B,
        M * N,
        seed=c["seed"],
        shuffle=c.get("shuffle", True),
        truncation=c.get("truncation", "truncate"),
    )
    for step, w in enumerate(windows):
        for k in ("tokens", "labels", "position_ids", "doc_ids"):
            assert w[k].tolist() == spec[k][M * step : M * step + M], (step, k)
    ref = tref.train_reference(
        str(mdir),
        str(init),
        windows,
        tref.RefConfig(
            learning_rate=c["learning_rate"],
            lr_scheduler_type=c["lr_scheduler_type"],
            # transformers 5: warmup_steps >= 1 is a step count, < 1 a ratio (DESIGN.md 10 item 19)
            warmup_steps=int(wu) if (wu := c["warmup_steps"]) >= 1 else math.ceil(wu * N),
            max_steps=N,
            max_grad_norm=c["max_grad_norm"],
            weight_decay=c["weight_decay"],
            betas=(c.get("adam_beta1", 0.9), c.get("adam_beta2", 0.999)),
            eps=c.get("adam_epsilon", 1e-8),
            loss_reduction=c.get("loss_reduction", "window_tokens"),
            **ref_kw,
        ),
    )
    assert [m["lr"] for m in met] == ref["lr"]
    assert [m["tokens"] for m in met] == ref["tokens"]
    loss_err = max(abs(m["loss"] - r) / r if r else abs(m["loss"]) for m, r in zip(met, ref["loss"]))
    norm_err = max(
        abs(m["grad_norm"] - r) / r if r else abs(m["grad_norm"]) for m, r in zip(met, ref["grad_norm"])
    )
    final = safetensors_torch.load_file(str(out / f"checkpoint-{N}/adapter_model.safetensors"))
    assert set(final) == set(ref["params"])
    init_sd = safetensors_torch.load_file(str(init / "adapter_model.safetensors"))
    perr = max(rel_err(final[k], v) for k, v in ref["params"].items())
    moved = min(rel_err(ref["params"][k], init_sd[k]) for k in init_sd)
    st = read_state(out / f"checkpoint-{N}/rrc_state.bin", final)
    merr = max(rel_err(st["m"][k], ref["m"][k]) for k in final)
    verr = max(rel_err(st["v"][k], ref["v"][k]) for k in final)
    errs = dict(loss=loss_err, grad_norm=norm_err, params=perr, m=merr, v=verr, moved=moved)
    print(f"trajectory {tmp_path.name}: " + ", ".join(f"{k} {v:.3g}" for k, v in errs.items()))
    tol_loss, tol_norm, tol_param, tol_mv = tol
    assert loss_err < tol_loss and norm_err < tol_norm
    assert perr < tol_param and merr < tol_mv and verr < tol_mv
    assert moved > 1e-3
    assert st["step"] == N and st["tokens_seen"] == sum(ref["tokens"])
    return met, ref, errs


# id -> (rrc config overrides, data seed, tolerances (loss, grad_norm, params, m / v)). Each covers
# settings the BASE_CFG trajectory does not: grad_accum_steps 3, micro_batch_size 1 / 2, non-default
# betas / eps / weight decay, a warmup ratio with a linear schedule, constant without clipping, both
# truncation policies with documents of length 1, seq_len, seq_len + 1 and longer. Tolerances are
# about 10x the measured errors (printed; DESIGN.md section 10).
TRAJ_EXTRA = {
    "truncate_accum3_mbs1_betas_linear_ratio": (
        dict(
            seq_len=16,
            micro_batch_size=1,
            grad_accum_steps=3,
            max_steps=7,
            lr_scheduler_type="linear",
            warmup_steps=0.3,
            adam_beta1=0.8,
            adam_beta2=0.95,
            adam_epsilon=1e-6,
            weight_decay=0.3,
            max_grad_norm=0.95,
            truncation="truncate",
        ),
        11,
        (1e-6, 3e-6, 5e-6, 2e-5),
    ),
    "drop_accum3_mbmean_constant_noclip": (
        dict(
            seq_len=16,
            micro_batch_size=2,
            grad_accum_steps=3,
            max_steps=6,
            lr_scheduler_type="constant",
            warmup_steps=0,
            adam_beta1=0.85,
            adam_epsilon=1e-4,  # no element with |g| ~ eps: Adam's g / (|g| + eps) stays well conditioned
            max_grad_norm=0.0,
            truncation="drop",
            loss_reduction="microbatch_mean",
        ),
        12,
        (1e-6, 3e-6, 5e-6, 2e-5),
    ),
}


@pytest.mark.parametrize("case", list(TRAJ_EXTRA))
def test_trajectory_fp32_other_settings_match_torch(case, tiny, train_exe, tmp_path_factory):
    cfg_kw, seed, tol = TRAJ_EXTRA[case]
    tmp = tmp_path_factory.mktemp(case)
    docs = make_edge_docs(40, seed, cfg_kw["seq_len"])
    met, ref, _ = run_and_compare_trajectory(tiny, train_exe, tmp, docs, cfg_kw, {}, tol)
    if cfg_kw["max_grad_norm"] == 0:
        assert all(m["clip_coef"] == 1.0 for m in met)
    else:
        clipped = [m["clip_coef"] < 1.0 for m in met]
        assert any(clipped) and not all(clipped), [m["grad_norm"] for m in met]
    if cfg_kw["lr_scheduler_type"] == "linear":
        assert met[0]["lr"] == 0.0 and met[2]["lr"] > 0  # ceil(0.3 * 7) = 3 warmup steps
    # the generated data has the packing edges (length 1 and longer than seq_len)
    lens = [len(t) for t, _ in docs]
    assert 1 in lens and max(lens) > cfg_kw["seq_len"]


@pytest.mark.parametrize("reduction", ["window_tokens", "microbatch_mean"])
def test_all_ignored_window_matches_adamw_with_zero_grads(reduction, tiny, train_exe, tmp_path):
    """DESIGN.md section 10 item 21: a window without labelled tokens logs loss 0 and still takes the
    AdamW step with g = 0 (moments decay, the parameters move by the remaining momentum and weight
    decay). Compared with torch.optim.AdamW fed explicit zero gradients; with shuffle off the fully
    masked documents 8..15 make one whole window (not the first one)."""
    rng = random.Random(4)
    docs = []
    for i in range(24):
        ln = rng.randint(3, 8)
        masked = 8 <= i < 16
        docs.append(([rng.randrange(VOCAB) for _ in range(ln)], [0] * ln if masked else [0] + [1] * (ln - 1)))
    cfg_kw = dict(
        seq_len=16,
        micro_batch_size=1,
        grad_accum_steps=2,
        max_steps=6,
        shuffle=False,
        loss_reduction=reduction,
        weight_decay=0.1,
        warmup_steps=0,
        lr_scheduler_type="constant",
        # microbatch_mean at eps 1e-8 measured params 1.5e-5 (m 6.1e-6), 7.7e-6 at eps 1e-6 and 5.1e-7 at
        # eps 1e-4: inferred to be elements with |g| near eps (Adam's m / (sqrt(v) + eps) ill-conditioned)
        adam_epsilon=1e-8 if reduction == "window_tokens" else 1e-4,
    )
    met, ref, _ = run_and_compare_trajectory(
        tiny, train_exe, tmp_path, docs, cfg_kw, dict(zero_grad_if_empty=True), (1e-6, 3e-6, 5e-6, 2e-5)
    )
    tokens = [m["tokens"] for m in met]
    empty = [i for i, t in enumerate(tokens) if t == 0]
    assert empty and empty[0] > 0, tokens  # an empty window after the optimizer state exists
    for i in empty:
        assert met[i]["loss"] == 0.0 and met[i]["grad_norm"] == 0.0 and ref["loss"][i] == 0.0


def test_trajectory_bf16_anchored_to_hf_bf16(setup_fp32, train_exe, tmp_path):
    """bf16 base weights: rrc's error against an fp32 torch run on the same bf16-rounded weights is
    at most twice HF-bf16's own error (+ slack), per step loss and per final LoRA tensor."""
    from transformers import AutoModelForCausalLM

    s = setup_fp32
    out = tmp_path / "o"
    cfg = write_cfg(
        tmp_path / "c.json",
        model_dir=str(s["model"]),
        data_path=str(s["data"]),
        output_dir=str(out),
        init_adapter=str(s["init"]),
        dtype="bf16",
        dump_batches=True,
    )
    run_train(train_exe, cfg)
    met = read_metrics(out)
    N = BASE_CFG["max_steps"]
    windows = load_windows(out, N)
    rounded = tmp_path / "rounded"
    m32 = AutoModelForCausalLM.from_pretrained(s["model"], dtype=torch.float32)
    with torch.no_grad():
        for p in m32.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    m32.save_pretrained(str(rounded))
    rc = tref.RefConfig(
        learning_rate=BASE_CFG["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_steps=N,
        max_grad_norm=BASE_CFG["max_grad_norm"],
        weight_decay=BASE_CFG["weight_decay"],
    )
    truth = tref.train_reference(str(rounded), str(s["init"]), windows, rc, dtype="f32")
    hf16 = tref.train_reference(str(s["model"]), str(s["init"]), windows, rc, dtype="bf16")
    e_hf = max(abs(a - b) for a, b in zip(hf16["loss"], truth["loss"]))
    e_rrc = max(abs(m["loss"] - b) for m, b in zip(met, truth["loss"]))
    print(f"bf16 loss error vs fp32 truth: rrc {e_rrc:.3g}, HF-bf16 {e_hf:.3g}")
    assert e_rrc <= 2.0 * e_hf + 1e-3
    final = safetensors_torch.load_file(str(out / f"checkpoint-{N}/adapter_model.safetensors"))
    worst = 0.0
    for k in truth["params"]:
        a = rel_err(hf16["params"][k], truth["params"][k])
        b = rel_err(final[k], truth["params"][k])
        worst = max(worst, b / (2.0 * a + 0.005))
        assert b <= 2.0 * a + 0.005, (k, b, a)
    print(f"bf16 params: worst rrc error / allowed = {worst:.3g}")


# --------------------------------------------------------------------------------------------
# b. bit-identical resume
# --------------------------------------------------------------------------------------------

CKPT_FILES = ["adapter_model.safetensors", "adapter_config.json", "rrc_state.bin"]


def test_resume_is_bit_identical(setup_fp32, train_exe, tmp_path):
    """Uninterrupted N steps (prefetch off) vs K steps + stop + resume to N (prefetch on): every
    checkpoint file byte for byte, the metrics' loss / lr / grad_norm / tokens. K = 5 is mid-epoch
    and the resumed part crosses an epoch boundary."""
    s = setup_fp32
    N, K = BASE_CFG["max_steps"], 5
    common = dict(model_dir=str(s["model"]), data_path=str(s["data"]))
    full = tmp_path / "full"
    run_train(
        train_exe,
        write_cfg(
            tmp_path / "full.json",
            **common,
            output_dir=str(full),
            init_adapter=str(s["init"]),
            save_steps=K,
            prefetch=False,
        ),
    )
    part = tmp_path / "part"
    run_train(
        train_exe,
        write_cfg(
            tmp_path / "a.json",
            **common,
            output_dir=str(part),
            init_adapter=str(s["init"]),
            stop_after_step=K,
        ),
    )
    assert not (part / f"checkpoint-{N}").exists()
    for f in CKPT_FILES:
        assert (part / f"checkpoint-{K}" / f).read_bytes() == (full / f"checkpoint-{K}" / f).read_bytes(), f
    st = read_state(
        part / f"checkpoint-{K}/rrc_state.bin",
        safetensors_torch.load_file(str(part / f"checkpoint-{K}/adapter_model.safetensors")),
    )
    assert st["step"] == K and st["index"] > 0 and st["micro_batches"] == 2 * K
    # simulate a crash after step K: a later record and a partial line in the metrics file
    with open(part / "metrics.jsonl", "a") as f:
        f.write('{"step": 7, "loss": 1.0}\n{"step": 8, "lo')
    run_train(
        train_exe,
        write_cfg(
            tmp_path / "b.json", **common, output_dir=str(part), resume_from=str(part / f"checkpoint-{K}")
        ),
    )
    for f in CKPT_FILES:
        assert (part / f"checkpoint-{N}" / f).read_bytes() == (full / f"checkpoint-{N}" / f).read_bytes(), f
    mf, mp = read_metrics(full), read_metrics(part)
    assert [m["step"] for m in mp] == list(range(1, N + 1))
    for a, b in zip(mf, mp):
        for key in ("loss", "lr", "grad_norm", "clip_coef", "tokens", "input_tokens", "tokens_seen", "epoch"):
            assert a[key] == b[key], (a["step"], key)
    assert mp[K]["epoch"] > st["epoch"]  # the resumed segment crosses an epoch boundary


def test_resume_refusals(setup_fp32, train_exe, tmp_path):
    s = setup_fp32
    common = dict(model_dir=str(s["model"]), data_path=str(s["data"]))
    out = tmp_path / "o"
    run_train(
        train_exe,
        write_cfg(
            tmp_path / "a.json",
            **common,
            output_dir=str(out),
            init_adapter=str(s["init"]),
            max_steps=4,
            stop_after_step=2,
        ),
    )
    ck = out / "checkpoint-2"
    # a trajectory-relevant setting changed
    p = run_train(
        train_exe,
        write_cfg(
            tmp_path / "b.json",
            **common,
            output_dir=str(out),
            resume_from=str(ck),
            max_steps=4,
            learning_rate=2e-2,
        ),
        expect_ok=False,
    )
    assert p.returncode == 2 and "learning_rate" in p.stderr
    # other data
    other = tmp_path / "other.rrcbin"
    prepare_sft.write_rrcbin(str(other), make_docs(30, seed=1), VOCAB)
    p = run_train(
        train_exe,
        write_cfg(
            tmp_path / "c.json",
            model_dir=str(s["model"]),
            data_path=str(other),
            output_dir=str(out),
            resume_from=str(ck),
            max_steps=4,
        ),
        expect_ok=False,
    )
    assert p.returncode == 2 and "not the data file" in p.stderr
    # an existing run is not overwritten without resume / overwrite_output_dir
    p = run_train(
        train_exe,
        write_cfg(tmp_path / "d.json", **common, output_dir=str(out), max_steps=4),
        expect_ok=False,
    )
    assert p.returncode == 2 and "already holds a run" in p.stderr
    # corrupted optimizer state and a swapped adapter are detected
    bad = tmp_path / "bad"
    bad.mkdir()
    for f in CKPT_FILES:
        (bad / f).write_bytes((ck / f).read_bytes())
    data = bytearray((bad / "rrc_state.bin").read_bytes())
    data[200] ^= 1
    (bad / "rrc_state.bin").write_bytes(bytes(data))
    p = run_train(
        train_exe,
        write_cfg(tmp_path / "e.json", **common, output_dir=str(out), resume_from=str(bad), max_steps=4),
        expect_ok=False,
    )
    assert p.returncode == 2 and "checksum" in p.stderr
    (bad / "rrc_state.bin").write_bytes((ck / "rrc_state.bin").read_bytes())
    (bad / "adapter_model.safetensors").write_bytes((s["init"] / "adapter_model.safetensors").read_bytes())
    p = run_train(
        train_exe,
        write_cfg(tmp_path / "f.json", **common, output_dir=str(out), resume_from=str(bad), max_steps=4),
        expect_ok=False,
    )
    assert p.returncode == 2 and "does not hold the parameters" in p.stderr


def test_init_adapter_config_must_match_lora_settings(setup_fp32, train_exe, tmp_path):
    """A PEFT adapter directory whose adapter_config.json disagrees with the config's lora section
    (here lora_alpha, use_rslora, targets, layers) is refused: the weights alone would load and the
    scaling would silently differ."""
    s = setup_fp32  # init adapter: r 4, alpha 8, all seven modules, all layers, no rslora
    common = dict(
        model_dir=str(s["model"]), data_path=str(s["data"]), init_adapter=str(s["init"]), max_steps=1
    )
    cases = [
        (dict(r=4, alpha=16, target_modules="all-linear"), "lora_alpha differs"),
        (dict(r=4, alpha=8, use_rslora=True, target_modules="all-linear"), "use_rslora differs"),
        (dict(r=4, alpha=8, target_modules=["q_proj", "v_proj"]), "target_modules differs"),
        (
            dict(r=4, alpha=8, target_modules="all-linear", layers_to_transform=[1]),
            "layers_to_transform differs",
        ),
    ]
    for i, (lora, msg) in enumerate(cases):
        p = run_train(
            train_exe,
            write_cfg(tmp_path / f"c{i}.json", **common, output_dir=str(tmp_path / f"o{i}"), lora=lora),
            expect_ok=False,
        )
        assert p.returncode == 2 and msg in p.stderr, (lora, p.stderr)
    # the matching settings train
    run_train(train_exe, write_cfg(tmp_path / "ok.json", **common, output_dir=str(tmp_path / "ok")))


def test_init_adapter_with_peft_patterns_is_refused(setup_fp32, train_exe, tmp_path):
    """A PEFT-saved adapter with alpha_pattern (review finding: ignored, so the adapter computed with
    lora_alpha everywhere and the pattern was dropped on save), rank_pattern or modules_to_save is
    refused before training; the same adapter without them trains."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    s = setup_fp32
    variants = {
        "plain": {},
        "alpha_pattern": dict(alpha_pattern={"q_proj": 64}),
        "rank_pattern": dict(rank_pattern={"v_proj": 2}),
        "modules_to_save": dict(modules_to_save=["lm_head"]),
    }
    for name, extra in variants.items():
        base = AutoModelForCausalLM.from_pretrained(s["model"], dtype=torch.float32)
        pm = get_peft_model(
            base, LoraConfig(r=4, lora_alpha=8, target_modules=list(MODS), lora_dropout=0.0,
                             init_lora_weights=False, **extra)
        )
        ad = tmp_path / f"ad_{name}"
        pm.save_pretrained(str(ad))
        cfg = json.loads((ad / "adapter_config.json").read_text())
        for k in extra:
            assert cfg[k], (k, cfg[k])  # PEFT wrote the field
        p = run_train(
            train_exe,
            write_cfg(tmp_path / f"{name}.json", model_dir=str(s["model"]), data_path=str(s["data"]),
                      init_adapter=str(ad), max_steps=1, output_dir=str(tmp_path / f"o_{name}"),
                      lora=dict(r=4, alpha=8, target_modules="all-linear")),
            expect_ok=name == "plain",
        )
        if name != "plain":
            assert p.returncode == 2 and f'"{name}" is set' in p.stderr, (name, p.stderr)


# --------------------------------------------------------------------------------------------
# c. the saved adapter loads in PEFT and reproduces rrc's forward
# --------------------------------------------------------------------------------------------


def test_peft_loads_checkpoint_and_matches_rrc_logits(tiny, train_exe, fwdbwd_exe, tmp_path):
    """Non-default r, alpha, rslora, a target subset and layers_to_transform: a wrong field in
    adapter_config.json would change PEFT's logits."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    mdir = tiny("qwen3_hd32_bias")
    targets = ["q_proj", "v_proj", "o_proj", "up_proj"]
    init = make_init_adapter(
        mdir, tmp_path / "init", r=6, alpha=10.0, targets=targets, rslora=True, layers=[1]
    )
    data = tmp_path / "d.rrcbin"
    prepare_sft.write_rrcbin(str(data), make_docs(20, seed=4), VOCAB)
    out = tmp_path / "o"
    cfg = write_cfg(
        tmp_path / "c.json",
        model_dir=str(mdir),
        data_path=str(data),
        output_dir=str(out),
        init_adapter=str(init),
        max_steps=3,
        lora=dict(r=6, alpha=10.0, use_rslora=True, target_modules=targets, layers_to_transform=[1]),
        dump_batches=True,
    )
    run_train(train_exe, cfg)
    ck = out / "checkpoint-3"
    ours = json.loads((ck / "adapter_config.json").read_text())
    # our adapter_config.json is a subset of what PEFT itself writes for the same LoraConfig
    base = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32)
    lc = LoraConfig(
        r=6,
        lora_alpha=10,
        target_modules=targets,
        lora_dropout=0.0,
        use_rslora=True,
        layers_to_transform=[1],
        task_type="CAUSAL_LM",
    )
    get_peft_model(base, lc).save_pretrained(str(tmp_path / "peft_own"))
    theirs = json.loads((tmp_path / "peft_own/adapter_config.json").read_text())
    for k, v in ours.items():
        assert k in theirs, k
        if k == "target_modules":
            assert sorted(v) == sorted(theirs[k])
        elif k != "base_model_name_or_path":
            assert v == theirs[k], (k, v, theirs[k])
    with safe_open(str(ck / "adapter_model.safetensors"), "pt") as f:
        assert f.metadata() == {"format": "pt"}
        keys = set(f.keys())
    assert keys == {
        f"base_model.model.model.layers.1.{'self_attn' if m in MODS[:4] else 'mlp'}.{m}.lora_{ab}.weight"
        for m in targets
        for ab in "AB"
    }
    # PEFT's loader
    base = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32, attn_implementation="eager")
    pm = PeftModel.from_pretrained(base, str(ck)).eval()
    cfg_loaded = pm.peft_config["default"]
    assert cfg_loaded.r == 6 and cfg_loaded.lora_alpha == 10 and cfg_loaded.use_rslora
    assert list(cfg_loaded.layers_to_transform) == [1] and set(cfg_loaded.target_modules) == set(targets)
    w = load_windows(out, 1)[0]
    batch = tmp_path / "batch.safetensors"
    safetensors_torch.save_file(
        {k: w[k][:1].contiguous() for k in ("tokens", "labels", "doc_ids")}, str(batch)
    )
    r = subprocess.run(
        [
            str(fwdbwd_exe),
            "--model",
            str(mdir),
            "--batch",
            str(batch),
            "--out",
            str(tmp_path / "fb"),
            "--dtype",
            "f32",
            "--lora",
            str(ck / "adapter_model.safetensors"),
            "--r",
            "6",
            "--alpha",
            "10",
            "--rslora",
            "--targets",
            ",".join(targets),
            "--layers",
            "1",
            "--logits",
            "--no-grad",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    got = safetensors_torch.load_file(str(tmp_path / "fb"))["logits"][0]
    T = w["tokens"].shape[2]
    worst = 0.0
    adapter_effect = 0.0
    # per document of the first micro-batch: PEFT logits vs rrc's packed logits
    for row in range(w["tokens"].shape[1]):
        d = w["doc_ids"][0, row]
        start = 0
        for t in range(1, T + 1):
            if t == T or d[t] != d[start]:
                ids = w["tokens"][0, row, start:t].long()[None]
                with torch.no_grad():
                    ref = pm(input_ids=ids).logits[0]
                    with pm.disable_adapter():
                        plain = pm(input_ids=ids).logits[0]
                mine = got[row * T + start : row * T + t]
                worst = max(worst, (mine - ref).abs().max().item())
                adapter_effect = max(adapter_effect, (ref - plain).abs().max().item())
                start = t
    print(f"PEFT-loaded vs rrc logits: max abs diff {worst:.3g}; adapter effect {adapter_effect:.3g}")
    assert worst < 2e-5
    assert adapter_effect > 1e-2  # a wrong scaling / layer / target would be visible


# --------------------------------------------------------------------------------------------
# d. data: prepare_sft.py, malformed files, determinism
# --------------------------------------------------------------------------------------------

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)


def build_tokenizer(out: pathlib.Path) -> Any:
    """A WordLevel tokenizer trained in the test (whitespace split), with a chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [
        "the cat sat on the mat",
        "a dog ran in the park",
        "what is two plus two",
        "it is four",
        "hello there how are you",
        "i am fine thanks",
        "translate cat to french",
        "chat",
    ]
    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    specials = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    tk.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=specials))
    # tokenize(prompt, add_special_tokens=True) adds BOS (prompt/completion examples)
    tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(str(out))
    return fast


def test_prepare_sft_round_trip(data_exe, tmp_path):
    from transformers import AutoTokenizer

    build_tokenizer(tmp_path / "tok")
    tok = AutoTokenizer.from_pretrained(str(tmp_path / "tok"), local_files_only=True)
    assert len(tok) < VOCAB
    examples = [
        {
            "messages": [
                {"role": "user", "content": "what is two plus two"},
                {"role": "assistant", "content": "it is four"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "chat"},
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "i am fine thanks"},
                {"role": "user", "content": "translate cat to french"},
                {"role": "assistant", "content": "chat"},
            ]
        },
        {"prompt": "the cat sat", "completion": "on the mat"},
    ]
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(e) for e in examples) + "\n")
    out = tmp_path / "d.rrcbin"
    p = subprocess.run(
        [
            sys.executable,
            str(RRC_DIR / "tools/prepare_sft.py"),
            "--tokenizer",
            str(tmp_path / "tok"),
            "--input",
            str(src),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    stats = json.loads(p.stdout)
    vocab, docs = prepare_sft.read_rrcbin(str(out))
    assert vocab == len(tok) and stats["num_docs"] == 3 and len(docs) == 3

    # tokens are exactly the tokenizer's; the masked tokens are exactly the assistant turns (+ <|end|>)
    def targets(i):
        toks, mask = docs[i]
        return tok.convert_ids_to_tokens([t for t, m in zip(toks, mask) if m])

    t0 = tok.apply_chat_template(examples[0]["messages"], tokenize=False)
    assert docs[0][0] == tok(t0, add_special_tokens=False)["input_ids"]
    assert targets(0) == ["it", "is", "four", "<|end|>"]
    assert targets(1) == ["i", "am", "fine", "thanks", "<|end|>", "chat", "<|end|>"]
    assert tok.convert_ids_to_tokens(docs[1][0])[:3] == ["<s>", "<|system|>", "chat"]
    assert tok.convert_ids_to_tokens(docs[2][0]) == ["<s>", "the", "cat", "sat", "on", "the", "mat", "</s>"]
    assert targets(2) == ["on", "the", "mat", "</s>"]
    # the C reader agrees
    c = subprocess.run(
        [str(data_exe), "check", str(out), "--vocab", str(VOCAB)], capture_output=True, text=True
    )
    assert c.returncode == 0, c.stderr
    info = json.loads(c.stdout)
    assert info["num_docs"] == 3 and info["vocab_size"] == len(tok)
    assert info["total_tokens"] == sum(len(t) for t, _ in docs)
    assert info["label_tokens"] == sum(sum(m) for _, m in docs) == stats["label_tokens"]
    # refusals: a template that is not prefix-stable, an unknown example shape
    bad = tmp_path / "bad.jinja"
    bad.write_text("{% for m in messages|reverse %}{{ m['content'] }} {% endfor %}")
    p = subprocess.run(
        [
            sys.executable,
            str(RRC_DIR / "tools/prepare_sft.py"),
            "--tokenizer",
            str(tmp_path / "tok"),
            "--input",
            str(src),
            "--output",
            str(tmp_path / "x.rrcbin"),
            "--chat-template",
            str(bad),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0 and "prefix-stable" in p.stderr
    src2 = tmp_path / "in2.jsonl"
    src2.write_text(json.dumps({"text": "the cat"}) + "\n")
    p = subprocess.run(
        [
            sys.executable,
            str(RRC_DIR / "tools/prepare_sft.py"),
            "--tokenizer",
            str(tmp_path / "tok"),
            "--input",
            str(src2),
            "--output",
            str(tmp_path / "y.rrcbin"),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0 and "needs 'messages'" in p.stderr


# Qwen3's stock template in miniature: an empty think block only in assistant turns after the last
# user message, so a multi-turn conversation's prefixes are not prefixes of the full rendering.
THINK_TEMPLATE = (
    "{%- set ns = namespace(last=-1) -%}"
    "{%- for m in messages -%}{%- if m['role'] == 'user' -%}{%- set ns.last = loop.index0 -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- for m in messages -%}<|{{ m['role'] }}|> "
    "{%- if m['role'] == 'assistant' and loop.index0 > ns.last %} <|system|> <|end|>{% endif %}"
    " {{ m['content'] }} <|end|> {% endfor -%}"
    "{%- if add_generation_prompt %}<|assistant|> {% endif -%}"
)


def test_prepare_sft_multi_turn_with_think_template(tmp_path):
    """Review finding: one multi-turn conversation under a Qwen3-style template aborted the whole
    conversion with a traceback. Now: exit 2 with a clean message naming the fix, nothing written;
    --skip-refused converts the rest and counts the refused ones; single-turn still works."""
    build_tokenizer(tmp_path / "tok")
    (tmp_path / "think.jinja").write_text(THINK_TEMPLATE)
    single = {"messages": [{"role": "user", "content": "hello there"}, {"role": "assistant", "content": "i am fine"}]}
    multi = {
        "messages": [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "i am fine thanks"},
            {"role": "user", "content": "translate cat to french"},
            {"role": "assistant", "content": "chat"},
        ]
    }
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(e) for e in (single, multi, single)) + "\n")

    def run(*extra):
        return subprocess.run(
            [sys.executable, str(RRC_DIR / "tools/prepare_sft.py"), "--tokenizer", str(tmp_path / "tok"),
             "--input", str(src), "--output", str(tmp_path / "o.rrcbin"), "--chat-template",
             str(tmp_path / "think.jinja"), *extra],
            capture_output=True, text=True,
        )

    p = run()
    assert p.returncode == 2, (p.returncode, p.stderr)
    assert "Traceback" not in p.stderr and "prepare_sft: error: example 1:" in p.stderr, p.stderr
    assert "prefix-stable" in p.stderr and "qwen3_training.jinja" in p.stderr and "--skip-refused" in p.stderr
    assert not (tmp_path / "o.rrcbin").exists() and not (tmp_path / "o.rrcbin.tmp").exists()
    p = run("--skip-refused")
    assert p.returncode == 0, p.stderr
    stats = json.loads(p.stdout)
    assert stats["num_docs"] == 2 and stats["skipped_refused"] == 1, stats
    assert "skipped 1 example" in p.stderr
    _, docs = prepare_sft.read_rrcbin(str(tmp_path / "o.rrcbin"))
    assert len(docs) == 2 and docs[0] == docs[1]


def _hdr(n, vocab, total, magic=b"RRCBIN01"):
    return struct.pack("<8sQQQ", magic, n, vocab, total)


def _doc(toks, mask):
    return struct.pack("<I", len(toks)) + struct.pack(f"<{len(toks)}I", *toks) + bytes(mask)


GOOD = _hdr(2, 100, 5) + _doc([1, 2, 3], [0, 1, 1]) + _doc([4, 5], [1, 1])

MALFORMED = {
    "empty": (b"", "header alone"),
    "short_header": (GOOD[:20], "header alone"),
    "bad_magic": (b"RRCBIN02" + GOOD[8:], "bad magic"),
    "num_docs_zero": (_hdr(0, 100, 0) + b"\0" * 9, "num_docs is 0"),
    "num_docs_too_large": (_hdr(1000, 100, 5) + GOOD[32:], "can hold at most"),
    "vocab_zero": (_hdr(2, 0, 5) + GOOD[32:], "vocab_size = 0"),
    "len_zero": (_hdr(2, 100, 3) + _doc([1, 2, 3], [0, 1, 1]) + struct.pack("<I", 0) + b"\0" * 5, "len 0"),
    "truncated_tokens": (GOOD[:-6], "truncated"),
    "truncated_mask": (GOOD[:-1], "truncated"),
    "truncated_len": (_hdr(2, 100, 5) + _doc([1, 2, 3], [0, 1, 1]) + b"\x02\x00" + b"\0" * 7, "truncated"),
    "trailing_bytes": (GOOD + b"\0", "trailing bytes"),
    "total_mismatch": (_hdr(2, 100, 6) + GOOD[32:], "total_tokens"),
    "token_ge_vocab": (_hdr(2, 100, 5) + _doc([1, 100, 3], [0, 1, 1]) + _doc([4, 5], [1, 1]), "out of range"),
    "mask_not_bool": (_hdr(2, 100, 5) + _doc([1, 2, 3], [0, 2, 1]) + _doc([4, 5], [1, 1]), "must be 0 or 1"),
}


def test_good_rrcbin_reference(data_exe, tmp_path):
    f = tmp_path / "good.rrcbin"
    f.write_bytes(GOOD)
    p = subprocess.run([str(data_exe), "check", str(f)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["label_tokens"] == 4
    assert prepare_sft.read_rrcbin(str(f)) == (100, [([1, 2, 3], [0, 1, 1]), ([4, 5], [1, 1])])


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_rrcbin_refused(case, data_exe, tmp_path):
    data, msg = MALFORMED[case]
    f = tmp_path / "bad.rrcbin"
    f.write_bytes(data)
    p = subprocess.run([str(data_exe), "check", str(f)], capture_output=True, text=True)
    assert p.returncode == 2, (case, p.stdout)
    assert msg in p.stderr, (case, p.stderr)


def test_rrcbin_vocab_against_model(data_exe, setup_fp32, train_exe, tmp_path):
    f = tmp_path / "big_vocab.rrcbin"
    prepare_sft.write_rrcbin(str(f), [([1, 2, 200], [0, 1, 1])], 300)
    p = subprocess.run(
        [str(data_exe), "check", str(f), "--vocab", str(VOCAB)], capture_output=True, text=True
    )
    assert p.returncode == 2 and "larger than the model's vocabulary" in p.stderr
    # rrc_train validates the data against the model too
    p = run_train(
        train_exe,
        write_cfg(
            tmp_path / "c.json",
            model_dir=str(setup_fp32["model"]),
            data_path=str(f),
            output_dir=str(tmp_path / "o"),
        ),
        expect_ok=False,
    )
    assert p.returncode == 2 and "larger than the model's vocabulary" in p.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["--seq-len", 2**32, "--rows", 2**32, "--count", 1],
        ["--seq-len", 2**31, "--rows", 1, "--count", 1],
        ["--seq-len", 4, "--rows", 2**62, "--count", 1],
        ["--seq-len", 4, "--rows", 1, "--count", 2, "--start-epoch", 2**63 - 1],
    ],
    ids=["TxB_wraps", "T_2^31", "B_2^62", "epoch_max"],
)
def test_rrc_data_refuses_out_of_range_sizes(args, data_exe, tmp_path):
    """Untrusted CLI sizes are checked before they are multiplied (no overflow, no SIGFPE)."""
    f = tmp_path / "d.rrcbin"
    f.write_bytes(GOOD)
    p = subprocess.run(
        [str(data_exe), "pack", str(f), "--out", str(tmp_path / "o.st"), *map(str, args)],
        capture_output=True,
        text=True,
    )
    assert p.returncode in (1, 2), (p.returncode, p.stderr)
    assert not (tmp_path / "o.st").exists()


def pack(exe, data, out, *args) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    p = subprocess.run(
        [str(exe), "pack", str(data), "--out", str(out), *map(str, args)], capture_output=True, text=True
    )
    assert p.returncode == 0, p.stderr
    t = safetensors_torch.load_file(str(out))
    with safe_open(str(out), "pt") as f:
        cur = [tuple(map(int, c.split(":"))) for c in f.metadata()["cursors"].split(",")]
    return t, cur


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("truncation", ["truncate", "drop"])
def test_packing_deterministic_and_matches_spec(shuffle, truncation, data_exe, tmp_path):
    docs = make_docs(25, seed=7, long_every=6)  # some documents are longer than seq_len = 32
    data = tmp_path / "d.rrcbin"
    prepare_sft.write_rrcbin(str(data), docs, VOCAB)
    args = ["--seq-len", 32, "--rows", 3, "--seed", 11, "--truncation", truncation, "--pad-id", 5]
    if not shuffle:
        args.append("--no-shuffle")
    args_all = [*args, "--count", 12]
    a, ca = pack(data_exe, data, tmp_path / "a", *args_all)
    runs = {
        "repeat": pack(data_exe, data, tmp_path / "b", *args_all),
        "no_prefetch": pack(data_exe, data, tmp_path / "c", *args_all, "--no-prefetch"),
        "hold_4": pack(data_exe, data, tmp_path / "d", *args_all, "--hold", 4),
        "hold_4_no_prefetch": pack(data_exe, data, tmp_path / "e", *args_all, "--hold", 4, "--no-prefetch"),
    }
    for name, (t, c) in runs.items():
        assert c == ca, name
        for k in a:
            assert torch.equal(a[k], t[k]), (name, k)
    spec, cs = tref.pack_reference(docs, 32, 3, 12, seed=11, shuffle=shuffle, truncation=truncation, pad_id=5)
    assert cs == ca
    for k in ("tokens", "labels", "position_ids", "doc_ids"):
        assert a[k].tolist() == spec[k], k
    # starting from a saved cursor reproduces the tail exactly (what resume relies on)
    e, i = ca[4]
    tail, ct = pack(
        data_exe, data, tmp_path / "f", *args, "--count", 7, "--start-epoch", e, "--start-index", i
    )
    assert ct == ca[5:]
    for k in a:
        assert torch.equal(tail[k], a[k][5:]), k
    # invariants: every epoch visits each usable document once; rows hold whole (truncated) documents
    usable = [j for j, (t, _) in enumerate(docs) if truncation == "truncate" or len(t) <= 32]
    if shuffle:
        assert tref.permutation(11, 0, len(usable)) != list(range(len(usable)))
    seen: list[list[int]] = []
    for m in range(12):
        for r in range(3):
            tok, lab, pos, did = (
                a[k][m, r].tolist() for k in ("tokens", "labels", "position_ids", "doc_ids")
            )
            start = 0
            for t in range(1, 33):
                if t == 32 or did[t] != did[start]:
                    seg = tok[start:t]
                    assert pos[start:t] == list(range(t - start))
                    match = [
                        j
                        for j in usable
                        if docs[j][0][: t - start] == seg and len(seg) == min(32, len(docs[j][0]))
                    ]
                    if match:
                        seen.append(seg)
                        j = match[0]
                        want = [x if mk else -100 for x, mk in zip(docs[j][0][:32], docs[j][1][:32])]
                        assert lab[start:t] == want
                    else:  # padding
                        assert seg == [5] * (t - start) and lab[start:t] == [-100] * (t - start)
                    start = t
    if truncation == "drop":
        assert all(len(s) <= 32 for s in seen)


# --------------------------------------------------------------------------------------------
# config strictness and documentation
# --------------------------------------------------------------------------------------------


def test_config_unknown_and_invalid_fields(train_exe, tmp_path):
    base = dict(model_dir="m", data_path="d", output_dir=str(tmp_path / "o"))
    cases = [
        (dict(learning_rat=1e-3), 'unknown key "learning_rat"'),
        (dict(lora=dict(r=4, alpah=8)), 'unknown key "lora.alpah"'),
        (dict(lora=dict(dropout=0.1)), "lora.dropout"),
        (dict(lora=dict(target_modules=["q_proj", "lm_head"])), "lm_head"),
        (dict(lr_scheduler_type="polynomial"), "polynomial"),
        (dict(warmup_ratio=0.1), 'unknown key "warmup_ratio"'),
        (dict(warmup_steps=-1), "warmup_steps must be"),
        (dict(seq_len=None), 'missing required key "seq_len"'),
        (dict(max_steps=0), '"max_steps" = 0'),
        (dict(grad_accum_steps=1.5), "must be an integer"),
        (dict(stop_after_step=99), "stop_after_step"),
        (dict(dtype="fp16"), '"dtype" = "fp16"'),
    ]
    for over, msg in cases:
        cfg = write_cfg(tmp_path / "c.json", **base, **over)
        p = run_train(train_exe, cfg, expect_ok=False, extra=["--print-schedule"])
        assert p.returncode == 2 and msg in p.stderr, (over, p.stderr)


def test_every_config_field_is_documented(train_exe):
    p = subprocess.run([str(train_exe), "--print-default-config"], capture_output=True, text=True)
    assert p.returncode == 0
    cfg = json.loads(p.stdout)
    keys = {k for k in cfg if k != "lora"} | {f"lora.{k}" for k in cfg["lora"]}
    readme = (RRC_DIR / "docs/README.md").read_text()
    section = readme.split("<!-- rrc_train config fields -->")[1].split("<!-- end of config fields -->")[0]
    documented = set(re.findall(r"^\| `([a-z_.0-9]+)` \|", section, flags=re.M))
    assert keys == documented, (
        f"undocumented: {sorted(keys - documented)}, documented but not a field: {sorted(documented - keys)}"
    )


# --------------------------------------------------------------------------------------------
# e. off-device test backend (rrc/docs/DESIGN.md section 14): device pointers are not host-mapped,
#    so any host access to backend memory in the trainer / model / LoRA / checkpoint code faults;
#    every output must be the plain CPU run's byte for byte
# --------------------------------------------------------------------------------------------

OFFDEV_STATS = re.compile(
    r"rrc_train: test-offdev backend: (\d+) allocations, (\d+) frees, (\d+) live, peak \d+ bytes, (\d+) bad frees, "
    r"(\d+) refused translations"
)
WALL_CLOCK = ("step_time_s", "elapsed_s", "input_tokens_per_s")


def run_offdev(exe, cfg_path) -> subprocess.CompletedProcess:
    p = run_train(exe, cfg_path, extra=["--test-offdev"])
    m = OFFDEV_STATS.search(p.stderr)
    assert m, p.stderr
    allocs, frees, live, bad_frees, bad_ptrs = map(int, m.groups())
    assert allocs > 0 and allocs == frees and live == 0 and bad_frees == 0 and bad_ptrs == 0, p.stderr
    return p


def assert_same_run(a: pathlib.Path, b: pathlib.Path, steps: list[int], n_batches: int = 0) -> None:
    for s in steps:
        for f in CKPT_FILES:
            assert (a / f"checkpoint-{s}" / f).read_bytes() == (b / f"checkpoint-{s}" / f).read_bytes(), (
                s,
                f,
            )
    for s in range(1, n_batches + 1):
        name = f"batches/step-{s:06d}.safetensors"
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ma, mb = read_metrics(a), read_metrics(b)
    assert len(ma) == len(mb)
    for x, y in zip(ma, mb):
        for k in WALL_CLOCK:
            x.pop(k), y.pop(k)
        assert x == y


OFFDEV_TRAIN_CASES = {
    "f32": dict(dtype="f32", prefetch=False),
    "bf16_f32acc_mbmean_norecompute": dict(
        dtype="bf16", backend="cpu-f32acc", loss_reduction="microbatch_mean", recompute="none", prefetch=True
    ),
}


@pytest.mark.parametrize("case", list(OFFDEV_TRAIN_CASES))
def test_offdev_backend_trajectory_and_resume_bit_identical(case, setup_fp32, train_exe, tmp_path):
    """The 8-step trajectory (PEFT init adapter, accumulation, clipping, warmup + cosine, weight
    decay, two epoch boundaries, batch dumps) on the CPU backend and on its off-device wrapper:
    every checkpoint file (adapter, adapter_config.json, optimizer state), every batch dump and the
    metrics (minus wall-clock fields). Then a stop at step 5 + resume on the off-device backend from
    the CPU run's checkpoint-5 (and the CPU backend from the off-device run's) reach the same bytes."""
    s = setup_fp32
    N, K = BASE_CFG["max_steps"], 5
    common = dict(model_dir=str(s["model"]), data_path=str(s["data"]), **OFFDEV_TRAIN_CASES[case])
    runs = {}
    for name in ("cpu", "offdev"):
        out = tmp_path / name
        cfg = write_cfg(
            tmp_path / f"{name}.json",
            **common,
            output_dir=str(out),
            init_adapter=str(s["init"]),
            save_steps=K,
            dump_batches=True,
        )
        if name == "cpu":
            p = run_train(train_exe, cfg)
            assert OFFDEV_STATS.search(p.stderr) is None
        else:
            run_offdev(train_exe, cfg)
        runs[name] = out
    assert_same_run(runs["cpu"], runs["offdev"], [K, N], N)
    for name, src in (("resume_offdev", runs["cpu"]), ("resume_cpu", runs["offdev"])):
        out = tmp_path / name
        cfg = write_cfg(
            tmp_path / f"{name}.json", **common, output_dir=str(out), resume_from=str(src / f"checkpoint-{K}")
        )
        if name == "resume_offdev":
            run_offdev(train_exe, cfg)
        else:
            run_train(train_exe, cfg)
        for f in CKPT_FILES:
            assert (out / f"checkpoint-{N}" / f).read_bytes() == (
                runs["cpu"] / f"checkpoint-{N}" / f
            ).read_bytes(), (
                name,
                f,
            )


def test_offdev_backend_refuses_cuda_config(setup_fp32, train_exe, tmp_path):
    """--test-offdev wraps a CPU backend only; nothing is created for a "cuda" config."""
    s = setup_fp32
    out = tmp_path / "o"
    cfg = write_cfg(
        tmp_path / "c.json",
        model_dir=str(s["model"]),
        data_path=str(s["data"]),
        output_dir=str(out),
        backend="cuda",
    )
    p = run_train(train_exe, cfg, expect_ok=False, extra=["--test-offdev"])
    assert p.returncode == 2 and "wraps a CPU backend" in p.stderr, p.stderr
    assert not out.exists()
