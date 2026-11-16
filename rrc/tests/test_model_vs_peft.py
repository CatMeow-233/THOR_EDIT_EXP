"""rrc model forward/backward with LoRA compared with transformers + PEFT on tiny random models.

build/rrc_fwdbwd (rrc/src/rrc_fwdbwd.c) runs one optimizer window (forward + backward of M packed
micro-batches) and writes logits, the window loss and every LoRA gradient. The reference is
transformers (eager attention) + PEFT with the SAME randomized, non-zero LoRA A and B (written by
PEFT's own save_pretrained, so the key names are PEFT's). Packing is checked against running
every document SEPARATELY through HF/PEFT and summing the per-token losses, not against a
hand-built 4D mask.

Precision: fp32 base weights are compared with tight tolerances (logic), bf16 base weights with
tolerances anchored to the measured error of HF itself in bf16 (both against an fp32 run on the
same bf16-rounded weights).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import subprocess
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
safetensors_torch = pytest.importorskip("safetensors.torch")
import make_tiny_model as mtm  # noqa: E402  (rrc/tools, put on sys.path by conftest)
from safetensors import safe_open  # noqa: E402

VOCAB = 128
T = 23  # prime: no attention chunk divides it
ATTN_CHUNK = 5
CE_CHUNK = 7
ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BACKENDS = ["cpu", "cpu-f32acc"]

# --------------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fwdbwd(rrc_build: pathlib.Path) -> pathlib.Path:
    exe = rrc_build / "rrc_fwdbwd"
    assert exe.is_file(), f"{exe} was not built"
    return exe


_MODELS: dict[str, pathlib.Path] = {}


@pytest.fixture(scope="session")
def tiny(tmp_path_factory: pytest.TempPathFactory):
    """tiny(name) -> checkpoint dir (generated once per session into a tmp dir)."""

    def get(name: str) -> pathlib.Path:
        if name not in _MODELS:
            out = tmp_path_factory.mktemp(f"tiny_{name}")
            if name == "llama_3layers":
                mtm.make_tiny("llama", str(out), seed=3, num_hidden_layers=3)
            else:
                mtm.make_variant(name, str(out), seed=1)
            _MODELS[name] = out
        return _MODELS[name]

    return get


@dataclasses.dataclass
class Doc:
    mb: int
    row: int
    start: int
    length: int
    tokens: list[int]
    labels: list[int]


@dataclasses.dataclass
class Window:
    tokens: Any  # [M, B, T] int32
    labels: Any
    pos: Any
    doc: Any
    docs: list[Doc]


def build_window(layout: list[list[list[tuple[int, int] | int]]], seed: int = 0, t_len: int = T) -> Window:
    """layout[m][row] = list of documents; a document is (length, prompt_len) or an int length
    for padding (all labels -100). Rows are filled to t_len (default T) exactly."""
    T = t_len
    g = torch.Generator().manual_seed(1000 + seed)
    M, B = len(layout), len(layout[0])
    tok = torch.zeros(M, B, T, dtype=torch.int32)
    lab = torch.full((M, B, T), -100, dtype=torch.int32)
    pos = torch.zeros(M, B, T, dtype=torch.int32)
    did = torch.zeros(M, B, T, dtype=torch.int32)
    docs: list[Doc] = []
    for m in range(M):
        assert len(layout[m]) == B
        for r in range(B):
            t0 = 0
            for k, d in enumerate(layout[m][r]):
                if isinstance(d, int):
                    pad, n, prompt = True, d, d
                else:
                    pad, (n, prompt) = False, d
                ids = torch.randint(0, VOCAB, (n,), generator=g, dtype=torch.int32)
                labels = ids.clone()
                if pad:
                    labels[:] = -100
                else:
                    # label of the first token stays real: a shift leaking across documents or rows
                    # would then pick it up
                    labels[1:prompt] = -100
                tok[m, r, t0 : t0 + n] = ids
                lab[m, r, t0 : t0 + n] = labels
                pos[m, r, t0 : t0 + n] = torch.arange(n, dtype=torch.int32)
                did[m, r, t0 : t0 + n] = 7 * k + 3
                docs.append(Doc(m, r, t0, n, ids.tolist(), labels.tolist()))
                t0 += n
            assert t0 == T, f"row {m},{r} has {t0} tokens, expected {T}"
    return Window(tok, lab, pos, did, docs)


def save_batch(w: Window, path: pathlib.Path, with_pos: bool = True, with_doc: bool = True) -> None:
    t = {"tokens": w.tokens.contiguous(), "labels": w.labels.contiguous()}
    if with_pos:
        t["position_ids"] = w.pos.contiguous()
    if with_doc:
        t["doc_ids"] = w.doc.contiguous()
    safetensors_torch.save_file(t, str(path))


# two micro-batches, unequal labelled counts; the 10-token document crosses the attention chunk
# boundaries at 10 and 15; row 0 of micro-batch 0 ends with padding
MAIN_LAYOUT: list[list[list[tuple[int, int] | int]]] = [
    [[(9, 3), (10, 2), 4], [(23, 5)]],
    [[(6, 2), (6, 1), (11, 4)], [(15, 3), 8]],
]


def peft_model(
    mdir: pathlib.Path, dtype, targets, r=4, alpha=8.0, rslora=False, layers=None, seed=0, round_bf16=False
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32, attn_implementation="eager")
    if round_bf16:  # fp32 model on bf16-rounded weights (the reference for bf16 error budgets)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(torch.bfloat16).to(torch.float32))
    if dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(targets),
        lora_dropout=0.0,
        use_rslora=rslora,
        layers_to_transform=layers,
        bias="none",
    )
    pm = get_peft_model(model, cfg)
    # randomized NON-ZERO A and B (B = 0 at init would make dA and the LoRA dx path vanish). A has
    # the kaiming bound 1/sqrt(in); B is sized so that s * B A is comparable to the base weights
    # (std 0.02): a much larger delta saturates the attention softmax of these tiny models and makes
    # the gradients ill-conditioned (bf16 rounding of the base weights alone then moves them by 10%+).
    g = torch.Generator().manual_seed(500 + seed)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_A" in n:
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) / p.shape[1] ** 0.5)
            elif "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    pm.eval()
    return pm


def lora_grads(pm) -> dict[str, Any]:
    out = {}
    for n, p in pm.named_parameters():
        if p.requires_grad:
            assert p.dtype == torch.float32, f"{n} is {p.dtype}: PEFT adapters are expected in fp32"
            assert p.grad is not None and p.grad.dtype == torch.float32
            out[n.replace(".default", "")] = p.grad.detach().clone()
    return out


def reference(pm, w: Window, reduction: str = "window_tokens"):
    """Each document separately through HF/PEFT; per-token losses combined like the engine."""
    import torch.nn.functional as F

    pm.zero_grad(set_to_none=True)
    M = w.tokens.shape[0]
    sums = [torch.zeros((), dtype=torch.float32) for _ in range(M)]
    counts = [0] * M
    logits: dict[tuple[int, int, int], Any] = {}
    for d in w.docs:
        ids = torch.tensor([d.tokens], dtype=torch.long)
        lg = pm(input_ids=ids).logits[0]
        logits[(d.mb, d.row, d.start)] = lg.detach().float()
        lab = torch.tensor(d.labels[1:], dtype=torch.long)
        n = int((lab != -100).sum())
        if n:
            sums[d.mb] = sums[d.mb] + F.cross_entropy(
                lg[:-1].float(), lab, ignore_index=-100, reduction="sum"
            )
            counts[d.mb] += n
    total = sum(counts)
    if reduction == "window_tokens":
        loss = sum(sums) / total if total else torch.zeros(())
    else:
        loss = sum(s / c for s, c in zip(sums, counts) if c) / M
    if isinstance(loss, torch.Tensor) and loss.requires_grad:
        loss.backward()
    grads = lora_grads(pm) if total else {}
    return float(loss), counts, logits, grads


def run_rrc(
    exe,
    mdir,
    batch,
    out,
    *,
    lora=None,
    dtype="f32",
    backend="cpu",
    targets=None,
    r=4,
    alpha=8.0,
    rslora=False,
    layers=None,
    attn_chunk=ATTN_CHUNK,
    ce_chunk=CE_CHUNK,
    reduction="window_tokens",
    logits=True,
    extra=(),
    expect_ok=True,
):
    cmd = [
        str(exe),
        "--model",
        str(mdir),
        "--batch",
        str(batch),
        "--out",
        str(out),
        "--dtype",
        dtype,
        "--backend",
        backend,
        "--attn-chunk",
        str(attn_chunk),
        "--ce-chunk",
        str(ce_chunk),
        "--reduction",
        reduction,
    ]
    if lora == "none":
        cmd.append("--no-lora")
    else:
        if lora is not None:
            cmd += ["--lora", str(lora)]
        cmd += ["--r", str(r), "--alpha", repr(alpha), "--targets", ",".join(targets or ALL_TARGETS)]
        if rslora:
            cmd.append("--rslora")
        if layers is not None:
            cmd += ["--layers", ",".join(map(str, layers))]
    if logits:
        cmd.append("--logits")
    cmd += list(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not expect_ok:
        return proc
    assert proc.returncode == 0, f"rrc_fwdbwd failed ({proc.returncode}): {proc.stderr}"
    tensors = safetensors_torch.load_file(str(out))
    with safe_open(str(out), "pt") as f:
        meta = f.metadata()
    return tensors, meta


def rel_err(a, b) -> float:
    a, b = a.double(), b.double()
    nb = b.norm().item()
    return (a - b).norm().item() / nb if nb > 0 else (a - b).norm().item()


def check_logits(w: Window, out: dict[str, Any], ref_logits, atol: float, rtol: float) -> float:
    T = w.tokens.shape[2]
    worst = 0.0
    for d in w.docs:
        got = out["logits"][d.mb, d.row * T + d.start : d.row * T + d.start + d.length]
        want = ref_logits[(d.mb, d.row, d.start)]
        err = (got - want).abs().max().item()
        worst = max(worst, err / (atol + rtol * want.abs().max().item()))
    assert worst <= 1.0, f"logits differ: worst error / tolerance = {worst:.3g}"
    return worst


def check_grads(out: dict[str, Any], ref: dict[str, Any], rtol: float) -> dict[str, float]:
    got = {k[len("grad.") :]: v for k, v in out.items() if k.startswith("grad.")}
    assert set(got) == set(ref), (
        f"grad name sets differ: rrc-only {set(got) - set(ref)}, peft-only {set(ref) - set(got)}"
    )
    errs = {}
    for k, v in ref.items():
        assert v.norm() > 0, f"reference grad {k} is zero: the test cannot detect errors in it"
        assert got[k].shape == v.shape, k
        errs[k] = rel_err(got[k], v)
    bad = {k: e for k, e in errs.items() if not e <= rtol}
    assert not bad, f"LoRA grads differ (relative L2 error > {rtol}): {bad}"
    return errs


def write_adapter(pm, d: pathlib.Path) -> pathlib.Path:
    pm.save_pretrained(str(d))
    f = d / "adapter_model.safetensors"
    assert f.is_file()
    return f


def assert_window_preconditions(w: Window, counts: list[int]) -> None:
    assert T > ATTN_CHUNK and T % ATTN_CHUNK != 0
    assert all(c > CE_CHUNK for c in counts if c), counts
    # a document crosses an attention chunk boundary and rows hold >= 2 documents
    crosses = [
        d for d in w.docs if d.start // ATTN_CHUNK != (d.start + d.length - 1) // ATTN_CHUNK and d.start > 0
    ]
    assert crosses
    per_row: dict[tuple[int, int], int] = {}
    for d in w.docs:
        per_row[(d.mb, d.row)] = per_row.get((d.mb, d.row), 0) + 1
    assert max(per_row.values()) >= 2


# --------------------------------------------------------------------------------------------
# base model: logits / hidden states / HF's own loss
# --------------------------------------------------------------------------------------------

BASE_VARIANTS = ["llama", "llama_tied", "llama3_rope", "qwen2", "qwen2_tied", "qwen3", "qwen3_hd32_bias"]


@pytest.mark.parametrize("variant", BASE_VARIANTS)
def test_base_model_matches_hf_including_hf_loss(variant, tiny, fwdbwd, tmp_path):
    """No LoRA, fp32, one document per row: logits, every hidden state, and the loss against HF's own
    model(input_ids, labels=...).loss (pins the label shift and ignore_index on HF's side)."""
    from transformers import AutoModelForCausalLM

    mdir = tiny(variant)
    hf = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32, attn_implementation="eager").eval()
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, VOCAB, (2, T), generator=g)
    labels = ids.clone()
    labels[0, 1:6] = -100
    labels[1, 10:14] = -100
    with torch.no_grad():
        ref = hf(input_ids=ids, labels=labels, output_hidden_states=True)
    safetensors_torch.save_file(
        {"tokens": ids.to(torch.int32), "labels": labels.to(torch.int32)}, str(tmp_path / "b")
    )
    out, meta = run_rrc(fwdbwd, mdir, tmp_path / "b", tmp_path / "o", lora="none", extra=["--hidden"])
    assert (out["logits"][0].reshape(2, T, VOCAB) - ref.logits).abs().max().item() < 2e-5
    hid = out["hidden"][0]
    assert hid.shape[0] == len(ref.hidden_states)
    for i, h in enumerate(ref.hidden_states):
        assert (hid[i].reshape(2, T, -1) - h).abs().max().item() < 2e-5, f"hidden state {i}"
    assert abs(float(meta["loss"]) - ref.loss.item()) < 2e-6 * max(1.0, ref.loss.item())
    # one row = one micro-batch-mean with M = 1 gives the same value
    _, meta2 = run_rrc(
        fwdbwd, mdir, tmp_path / "b", tmp_path / "o2", lora="none", reduction="microbatch_mean", logits=False
    )
    assert abs(float(meta2["loss"]) - ref.loss.item()) < 2e-6 * max(1.0, ref.loss.item())


# --------------------------------------------------------------------------------------------
# LoRA, fp32 base: logits, loss, every LoRA gradient (both CPU backends)
# --------------------------------------------------------------------------------------------

# (variant, targets, rslora): attention-only, MLP-only and mixed target sets across families
LORA_CASES = [
    ("llama", ALL_TARGETS, False),
    ("llama_mha", ALL_TARGETS, False),
    ("llama_mqa", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj"], True),
    ("llama3_rope", ["q_proj", "v_proj", "down_proj"], False),
    ("llama_tied", ["k_proj", "o_proj", "gate_proj"], True),
    ("qwen2", ALL_TARGETS, False),
    ("qwen2_tied", ["q_proj", "k_proj", "up_proj"], False),
    ("qwen3", ["o_proj", "up_proj", "down_proj"], False),
    ("qwen3_hd32_bias", ALL_TARGETS, True),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant,targets,rslora", LORA_CASES, ids=[c[0] for c in LORA_CASES])
def test_lora_fp32_matches_peft(variant, targets, rslora, backend, tiny, fwdbwd, tmp_path):
    mdir = tiny(variant)
    cfg = json.loads((mdir / "config.json").read_text())
    pm = peft_model(mdir, torch.float32, targets, rslora=rslora, alpha=8.0)
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    hq, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    if variant == "llama_mha":
        assert hkv == hq  # G = 1
    elif variant == "llama_mqa":
        assert hkv == 1 and hq > 1  # G = Hq
    else:
        assert hq // hkv > 1 and hkv > 1  # GQA
    if variant.startswith("qwen3_hd32"):
        assert hd != cfg["hidden_size"] // cfg["num_attention_heads"]
    adapter = write_adapter(pm, tmp_path / "adapter")
    w = build_window(MAIN_LAYOUT, seed=1)
    loss, counts, ref_logits, ref_grads = reference(pm, w)
    assert_window_preconditions(w, counts)
    assert counts[0] != counts[1]
    save_batch(w, tmp_path / "batch")
    out, meta = run_rrc(
        fwdbwd,
        mdir,
        tmp_path / "batch",
        tmp_path / "out",
        lora=adapter,
        backend=backend,
        targets=targets,
        rslora=rslora,
    )
    assert out["n_tokens"].tolist() == counts
    tol = 2e-5 if backend == "cpu" else 1e-4
    check_logits(w, out, ref_logits, atol=tol, rtol=tol)
    assert abs(float(meta["loss"]) - loss) <= tol * loss
    check_grads(out, ref_grads, rtol=5e-5 if backend == "cpu" else 5e-4)
    # the parameters written back are exactly the ones PEFT used
    sd = safetensors_torch.load_file(str(adapter))
    for k, v in sd.items():
        assert torch.equal(out["param." + k], v.float()), k


# --------------------------------------------------------------------------------------------
# loss reductions, zero-labelled micro-batches and windows
# --------------------------------------------------------------------------------------------


def test_reductions_window_tokens_vs_microbatch_mean(tiny, fwdbwd, tmp_path):
    mdir = tiny("llama")
    pm = peft_model(mdir, torch.float32, ALL_TARGETS)
    adapter = write_adapter(pm, tmp_path / "adapter")
    # micro-batch 1 has only 3 labelled tokens, micro-batch 0 has 32: the two reductions weight them
    # very differently
    w = build_window([MAIN_LAYOUT[0], [[(23, 21)], [(12, 11), 11]]], seed=2)
    save_batch(w, tmp_path / "batch")
    results = {}
    for red in ["window_tokens", "microbatch_mean"]:
        loss, counts, _, ref_grads = reference(pm, w, red)
        assert counts == [32, 3]
        out, meta = run_rrc(
            fwdbwd,
            mdir,
            tmp_path / "batch",
            tmp_path / f"out_{red}",
            lora=adapter,
            reduction=red,
            logits=False,
        )
        assert abs(float(meta["loss"]) - loss) <= 2e-6 * loss
        check_grads(out, ref_grads, rtol=5e-5)
        results[red] = (loss, ref_grads)
    # the two reductions really differ on this window (unequal labelled counts)
    # (per-token losses of a random tiny model are all close to log(vocab), so the loss values differ
    # little; still by > 100x the tolerance used above)
    assert (
        abs(results["window_tokens"][0] - results["microbatch_mean"][0])
        > 100 * 2e-6 * results["window_tokens"][0]
    )
    for k in results["window_tokens"][1]:
        assert rel_err(results["window_tokens"][1][k], results["microbatch_mean"][1][k]) > 1e-2, k


@pytest.mark.parametrize("reduction", ["window_tokens", "microbatch_mean"])
def test_zero_labelled_microbatch_inside_window(reduction, tiny, fwdbwd, tmp_path):
    """Micro-batch 1 of 3 has no labelled token (padding and prompt-only documents): it contributes
    loss 0 and gradient 0, and still counts in num_micro_batches for microbatch_mean."""
    mdir = tiny("qwen2")
    pm = peft_model(mdir, torch.float32, ALL_TARGETS)
    adapter = write_adapter(pm, tmp_path / "adapter")
    layout = [MAIN_LAYOUT[0], [[(23, 23)], [5, (18, 18)]], MAIN_LAYOUT[1]]
    w = build_window(layout, seed=3)
    # the prompt-only documents still have a real label on their first token, which is never a target
    loss, counts, ref_logits, ref_grads = reference(pm, w, reduction)
    assert counts[1] == 0 and counts[0] > 0 and counts[2] > 0
    save_batch(w, tmp_path / "batch")
    out, meta = run_rrc(fwdbwd, mdir, tmp_path / "batch", tmp_path / "out", lora=adapter, reduction=reduction)
    assert out["n_tokens"].tolist() == counts
    assert float(out["loss_mb"][1]) == 0.0
    assert abs(float(meta["loss"]) - loss) <= 2e-5 * loss
    check_grads(out, ref_grads, rtol=5e-5)
    check_logits(w, out, ref_logits, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("reduction", ["window_tokens", "microbatch_mean"])
def test_fully_unlabelled_window_is_zero_not_nan(reduction, tiny, fwdbwd, tmp_path):
    mdir = tiny("llama")
    pm = peft_model(mdir, torch.float32, ALL_TARGETS)
    adapter = write_adapter(pm, tmp_path / "adapter")
    w = build_window([[[(23, 23)], [(10, 10), 13]], [[23], [(12, 12), (11, 11)]]], seed=4)
    save_batch(w, tmp_path / "batch")
    out, meta = run_rrc(fwdbwd, mdir, tmp_path / "batch", tmp_path / "out", lora=adapter, reduction=reduction)
    assert out["n_tokens"].tolist() == [0, 0]
    assert float(meta["loss"]) == 0.0 and meta["window_tokens"] == "0"
    grads = [v for k, v in out.items() if k.startswith("grad.")]
    assert grads and all(torch.isfinite(g).all() and (g == 0).all() for g in grads)


@pytest.mark.parametrize("shape", [(0, 2**32, 2**32), (0, 2, 3), (1, 0, 5), (2, 3, 0), (0, 7)], ids=str)
@pytest.mark.parametrize("logits", [False, True], ids=["no_logits", "logits"])
def test_fwdbwd_refuses_empty_batch_shapes(shape, logits, tiny, fwdbwd, tmp_path):
    """A batch file is untrusted input: an empty dimension is a shape error, not an empty output file
    (or, with huge other dimensions, a signed overflow in B * T)."""
    # numel is 0, so the file is just the header (torch cannot build the [0, 2^32, 2^32] tensor)
    hdr = {k: {"dtype": "I32", "shape": list(shape), "data_offsets": [0, 0]} for k in ("tokens", "labels")}
    raw = json.dumps(hdr).encode()
    (tmp_path / "b").write_bytes(len(raw).to_bytes(8, "little") + raw)
    proc = run_rrc(
        fwdbwd, tiny("llama"), tmp_path / "b", tmp_path / "o", lora="none", logits=logits, expect_ok=False
    )
    assert proc.returncode != 0 and proc.returncode > 0, (proc.returncode, proc.stderr)
    assert not (tmp_path / "o").exists()


# --------------------------------------------------------------------------------------------
# input-gradient path: LoRA on upper layers only vs on all layers
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layers",
    [[1, 2], [2], None, [1], [0], [0, 2]],
    ids=["upper2", "top", "all", "middle", "bottom", "gap"],
)
def test_upper_layers_only_vs_all_layers(layers, tiny, fwdbwd, tmp_path):
    """With LoRA only on upper layers the backward stops at the lowest adapted layer; with all layers
    the lower layers' grads require the input gradient propagated through every upper layer
    (non-zero B there). [1]: the backward stops at a middle layer below an unadapted one; [0]: the
    input gradient crosses two unadapted layers; [0, 2]: it crosses the unadapted gap layer 1."""
    mdir = tiny("llama_3layers")
    targets = ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    pm = peft_model(mdir, torch.float32, targets, layers=layers)
    adapter = write_adapter(pm, tmp_path / "adapter")
    w = build_window(MAIN_LAYOUT, seed=5)
    loss, _, ref_logits, ref_grads = reference(pm, w)
    save_batch(w, tmp_path / "batch")
    out, meta = run_rrc(
        fwdbwd, mdir, tmp_path / "batch", tmp_path / "out", lora=adapter, targets=targets, layers=layers
    )
    got_layers = {int(k.split(".layers.")[1].split(".")[0]) for k in out if k.startswith("grad.")}
    assert got_layers == (set(layers) if layers else {0, 1, 2})
    check_logits(w, out, ref_logits, atol=2e-5, rtol=2e-5)
    assert abs(float(meta["loss"]) - loss) <= 2e-5 * loss
    check_grads(out, ref_grads, rtol=5e-5)


# --------------------------------------------------------------------------------------------
# attention layouts at the ends of GQA, single-token documents, T = 1
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("attn_chunk", [1, 6, T], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("variant", ["llama_mha", "llama_mqa"])
def test_mha_mqa_attention_chunks(variant, attn_chunk, tiny, fwdbwd, tmp_path):
    """MHA (G = 1) and MQA (Hkv = 1) are the extreme strides of the batched dK / dV GEMMs; chunks of 1
    and 6 (< T) exercise the chunked attention against the unchunked one (T)."""
    mdir = tiny(variant)
    cfg = json.loads((mdir / "config.json").read_text())
    assert cfg["num_key_value_heads"] == (cfg["num_attention_heads"] if variant == "llama_mha" else 1)
    pm = peft_model(mdir, torch.float32, ALL_TARGETS, seed=7)
    adapter = write_adapter(pm, tmp_path / "adapter")
    w = build_window(MAIN_LAYOUT, seed=7)
    loss, counts, ref_logits, ref_grads = reference(pm, w)
    save_batch(w, tmp_path / "batch")
    out, meta = run_rrc(
        fwdbwd, mdir, tmp_path / "batch", tmp_path / "out", lora=adapter, attn_chunk=attn_chunk, ce_chunk=3
    )
    assert out["n_tokens"].tolist() == counts
    check_logits(w, out, ref_logits, atol=2e-5, rtol=2e-5)
    assert abs(float(meta["loss"]) - loss) <= 2e-5 * loss
    check_grads(out, ref_grads, rtol=5e-5)


# runs of length-1 documents (k0 == q0, a key chunk of one row, the shifted label always ignored),
# one starting exactly on each attention chunk boundary 5, 10, 15 (ATTN_CHUNK = 5) and a length-2
# document starting on 10; micro-batch 1 keeps both micro-batches labelled
SINGLE_TOKEN_LAYOUT: list[list[list[tuple[int, int] | int]]] = [
    [
        [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (4, 2), (1, 1), (2, 1), (2, 1), (1, 1), (7, 2)],
        [(9, 3), (1, 1), (2, 1), (11, 4)],
    ],
    MAIN_LAYOUT[1],
]


@pytest.mark.parametrize("attn_chunk", [1, 2, ATTN_CHUNK, T], ids=lambda c: f"chunk{c}")
def test_single_token_documents(attn_chunk, tiny, fwdbwd, tmp_path):
    mdir = tiny("qwen2")
    pm = peft_model(mdir, torch.float32, ALL_TARGETS, seed=8)
    adapter = write_adapter(pm, tmp_path / "adapter")
    w = build_window(SINGLE_TOKEN_LAYOUT, seed=8)
    ones = [d for d in w.docs if d.length == 1]
    assert len(ones) >= 8
    assert {d.start for d in ones if d.mb == 0 and d.row == 0} >= {0, 1, 2, 3, 4, 5, 10, 15}
    assert any(d.length == 2 and d.start == 10 for d in w.docs)
    loss, counts, ref_logits, ref_grads = reference(pm, w)
    assert all(counts)
    save_batch(w, tmp_path / "batch")
    for ce_chunk in (1, CE_CHUNK):
        out, meta = run_rrc(
            fwdbwd,
            mdir,
            tmp_path / "batch",
            tmp_path / f"out{ce_chunk}",
            lora=adapter,
            attn_chunk=attn_chunk,
            ce_chunk=ce_chunk,
        )
        assert out["n_tokens"].tolist() == counts
        check_logits(w, out, ref_logits, atol=2e-5, rtol=2e-5)
        assert abs(float(meta["loss"]) - loss) <= 2e-5 * loss
        check_grads(out, ref_grads, rtol=5e-5)


@pytest.mark.parametrize("attn_chunk", [1, ATTN_CHUNK])
def test_rows_of_one_token(attn_chunk, tiny, fwdbwd, tmp_path):
    """T = 1: every row is a single document whose only label is shifted away. Logits match HF, the
    counts are 0, the loss is 0 and every gradient is exactly 0 (not NaN)."""
    mdir = tiny("llama")
    pm = peft_model(mdir, torch.float32, ALL_TARGETS, seed=9)
    adapter = write_adapter(pm, tmp_path / "adapter")
    w = build_window([[[(1, 1)], [(1, 1)], [(1, 1)]], [[(1, 1)], [(1, 1)], [(1, 1)]]], seed=9, t_len=1)
    _, counts, ref_logits, _ = reference(pm, w)
    assert counts == [0, 0]
    save_batch(w, tmp_path / "batch")
    for red in ("window_tokens", "microbatch_mean"):
        out, meta = run_rrc(
            fwdbwd,
            mdir,
            tmp_path / "batch",
            tmp_path / f"out_{red}",
            lora=adapter,
            attn_chunk=attn_chunk,
            reduction=red,
        )
        assert out["n_tokens"].tolist() == [0, 0]
        assert float(meta["loss"]) == 0.0
        check_logits(w, out, ref_logits, atol=2e-5, rtol=2e-5)
        grads = [v for k, v in out.items() if k.startswith("grad.")]
        assert grads and all((g == 0).all() for g in grads)


# --------------------------------------------------------------------------------------------
# bf16 base weights (loose tolerances anchored to HF's own bf16 error)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", ["llama3_rope", "qwen2_tied", "qwen3_hd32_bias"])
def test_lora_bf16_matches_peft(variant, backend, tiny, fwdbwd, tmp_path):
    mdir = tiny(variant)
    w = build_window(MAIN_LAYOUT, seed=6)
    save_batch(w, tmp_path / "batch")
    # fp32 truth on the bf16-rounded weights, HF bf16, rrc bf16 -- all with identical LoRA weights
    pm32 = peft_model(mdir, torch.float32, ALL_TARGETS, seed=6, round_bf16=True)
    l32, counts, lg32, g32 = reference(pm32, w)
    pm16 = peft_model(mdir, torch.bfloat16, ALL_TARGETS, seed=6)
    for n, p in pm16.named_parameters():
        if "lora_" in n:
            assert p.dtype == torch.float32, n
    adapter = write_adapter(pm16, tmp_path / "adapter")
    l16, counts16, lg16, g16 = reference(pm16, w)
    assert counts16 == counts
    out, meta = run_rrc(
        fwdbwd, mdir, tmp_path / "batch", tmp_path / "out", lora=adapter, dtype="bf16", backend=backend
    )
    got = {k[len("grad.") :]: v for k, v in out.items() if k.startswith("grad.")}
    assert set(got) == set(g16) == set(g32)
    # logits: rrc-bf16 vs fp32 truth no worse than (2x HF-bf16 vs fp32 truth + small slack)
    for d in w.docs:
        key = (d.mb, d.row, d.start)
        mine = out["logits"][d.mb, d.row * T + d.start : d.row * T + d.start + d.length]
        e_hf = (lg16[key] - lg32[key]).abs().max().item()
        e_rrc = (mine - lg32[key]).abs().max().item()
        assert e_rrc <= 2.0 * e_hf + 0.005, (key, e_rrc, e_hf)
        assert (mine - lg16[key]).abs().max().item() <= 3.0 * e_hf + 0.005
    lrrc = float(meta["loss"])
    assert abs(lrrc - l32) <= 2.0 * abs(l16 - l32) + 2e-3 * l32, (lrrc, l16, l32)
    # gradients: per tensor, relative to the fp32 truth
    for k in g32:
        e_hf = rel_err(g16[k], g32[k])
        e_rrc = rel_err(got[k], g32[k])
        assert e_rrc <= 2.0 * e_hf + 0.005, (k, e_rrc, e_hf)
        assert rel_err(got[k], g16[k]) <= 3.0 * e_hf + 0.005, (k, rel_err(got[k], g16[k]), e_hf)


# --------------------------------------------------------------------------------------------
# engine-internal invariances (independent of PEFT)
# --------------------------------------------------------------------------------------------


def test_chunking_recompute_and_batch_derivation_invariance(tiny, fwdbwd, tmp_path):
    mdir = tiny("qwen3_hd32_bias")
    pm = peft_model(mdir, torch.float32, ALL_TARGETS)
    adapter = write_adapter(pm, tmp_path / "adapter")
    w = build_window(MAIN_LAYOUT, seed=8)
    save_batch(w, tmp_path / "full")
    save_batch(w, tmp_path / "nopos", with_pos=False)
    save_batch(w, tmp_path / "nodoc", with_doc=False)
    base, _ = run_rrc(
        fwdbwd, mdir, tmp_path / "full", tmp_path / "o0", lora=adapter, attn_chunk=1024, ce_chunk=4096
    )
    runs = {
        "small_chunks": dict(attn_chunk=3, ce_chunk=2),
        "chunk_1": dict(attn_chunk=1, ce_chunk=1),
        "recompute_none": dict(attn_chunk=1024, ce_chunk=4096, extra=["--recompute", "none"]),
    }
    for name, kw in runs.items():
        out, _ = run_rrc(fwdbwd, mdir, tmp_path / "full", tmp_path / f"o_{name}", lora=adapter, **kw)
        for k, v in base.items():
            if name == "recompute_none":
                assert torch.equal(out[k], v), f"{name}: {k} not bit-identical"
            else:
                assert rel_err(out[k], v) < 1e-5, (name, k)
    # position_ids derived from doc_ids, and doc_ids derived from position_ids: bit-identical
    for name in ["nopos", "nodoc"]:
        out, _ = run_rrc(
            fwdbwd,
            mdir,
            tmp_path / name,
            tmp_path / f"o_{name}",
            lora=adapter,
            attn_chunk=1024,
            ce_chunk=4096,
        )
        for k, v in base.items():
            assert torch.equal(out[k], v), f"{name}: {k}"


def test_sharded_bf16_checkpoint_loads_like_single_file(tiny, fwdbwd, tmp_path):
    """A bf16 checkpoint saved by HF in several shards: loaded as bf16 it is bit-identical to the
    fp32 single-file checkpoint rounded by the loader (RNE == torch's .to(bfloat16)); loaded as f32 it
    is bit-identical to an fp32 single-file checkpoint of the bf16-rounded weights."""
    from transformers import AutoModelForCausalLM

    src = tiny("qwen2_tied")
    m16 = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    sharded = tmp_path / "sharded"
    m16.save_pretrained(sharded, max_shard_size="100KB")
    assert (
        len(list(sharded.glob("*.safetensors"))) > 1 and (sharded / "model.safetensors.index.json").is_file()
    )
    rounded = tmp_path / "rounded_f32"
    m16.to(torch.float32).save_pretrained(rounded)
    pm = peft_model(src, torch.float32, ALL_TARGETS)
    adapter = write_adapter(pm, tmp_path / "adapter")
    w = build_window(MAIN_LAYOUT, seed=12)
    save_batch(w, tmp_path / "batch")
    for dtype, ref_dir in [("bf16", src), ("f32", rounded)]:
        a, ma = run_rrc(
            fwdbwd, sharded, tmp_path / "batch", tmp_path / f"s_{dtype}", lora=adapter, dtype=dtype
        )
        b, mb = run_rrc(
            fwdbwd, ref_dir, tmp_path / "batch", tmp_path / f"r_{dtype}", lora=adapter, dtype=dtype
        )
        assert ma["loss"] == mb["loss"], dtype
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), (dtype, k)


def test_init_b_zero_and_deterministic(tiny, fwdbwd, tmp_path):
    """Without --lora: A ~ U(-1/sqrt(in), 1/sqrt(in)) from the seeded generator, B = 0; the model
    output equals the base model exactly; dA = 0 exactly, dB != 0; same seed -> same A."""
    mdir = tiny("llama")
    w = build_window(MAIN_LAYOUT, seed=9)
    save_batch(w, tmp_path / "batch")
    base, bmeta = run_rrc(fwdbwd, mdir, tmp_path / "batch", tmp_path / "base", lora="none")
    o1, m1 = run_rrc(fwdbwd, mdir, tmp_path / "batch", tmp_path / "o1", extra=["--seed", "11"])
    o2, _ = run_rrc(fwdbwd, mdir, tmp_path / "batch", tmp_path / "o2", extra=["--seed", "11"])
    o3, _ = run_rrc(fwdbwd, mdir, tmp_path / "batch", tmp_path / "o3", extra=["--seed", "12"])
    assert torch.equal(o1["logits"], base["logits"]) and m1["loss"] == bmeta["loss"]
    n_a = 0
    for k, v in o1.items():
        if not k.startswith("param."):
            continue
        assert torch.equal(v, o2[k])
        if ".lora_B." in k:
            assert (v == 0).all()
            assert o1["grad." + k[len("param.") :]].abs().sum() > 0
        else:
            n_a += 1
            fan_in = v.shape[1]
            bound = 1.0 / fan_in**0.5
            assert v.abs().max() <= bound and v.abs().max() > 0.9 * bound
            assert abs(v.mean().item()) < 0.1 * bound
            assert abs(v.var().item() - bound**2 / 3) < 0.25 * bound**2 / 3
            assert (o1["grad." + k[len("param.") :]] == 0).all()
            assert not torch.equal(v, o3[k])
    assert n_a == 2 * 7


def test_refusals(tiny, fwdbwd, tmp_path):
    mdir = tiny("llama")
    w = build_window(MAIN_LAYOUT, seed=10)
    save_batch(w, tmp_path / "batch")
    p = run_rrc(fwdbwd, mdir, tmp_path / "batch", tmp_path / "o", extra=["--dropout", "0.1"], expect_ok=False)
    assert p.returncode == 2 and "lora_dropout" in p.stderr
    p = run_rrc(
        fwdbwd, mdir, tmp_path / "batch", tmp_path / "o", targets=["q_proj", "lm_head"], expect_ok=False
    )
    assert p.returncode == 2 and "lm_head" in p.stderr
    # adapter with a different target set than requested: exact tensor-set check
    pm = peft_model(mdir, torch.float32, ["q_proj", "v_proj"])
    adapter = write_adapter(pm, tmp_path / "adapter")
    p = run_rrc(
        fwdbwd, mdir, tmp_path / "batch", tmp_path / "o", lora=adapter, targets=["q_proj"], expect_ok=False
    )
    assert p.returncode == 2 and "not part of this adapter" in p.stderr
    p = run_rrc(
        fwdbwd,
        mdir,
        tmp_path / "batch",
        tmp_path / "o",
        lora=adapter,
        targets=["q_proj", "v_proj", "k_proj"],
        expect_ok=False,
    )
    assert p.returncode == 2 and "missing" in p.stderr
    p = run_rrc(
        fwdbwd,
        mdir,
        tmp_path / "batch",
        tmp_path / "o",
        lora=adapter,
        targets=["q_proj", "v_proj"],
        r=8,
        expect_ok=False,
    )
    assert p.returncode == 2 and "shape" in p.stderr
    # non-contiguous documents in a row are refused
    bad = build_window(MAIN_LAYOUT, seed=10)
    bad.doc[0, 0, 20:] = bad.doc[0, 0, 0]
    save_batch(bad, tmp_path / "bad")
    p = run_rrc(fwdbwd, mdir, tmp_path / "bad", tmp_path / "o", expect_ok=False)
    assert p.returncode == 2 and "contiguous" in p.stderr


# --------------------------------------------------------------------------------------------
# off-device test backend (rrc/docs/DESIGN.md section 14): device pointers are not host-mapped, so
# any host access to backend memory in the model / LoRA code faults; the output file must be the
# plain CPU backend's byte for byte
# --------------------------------------------------------------------------------------------

OFFDEV_STATS = re.compile(
    r"test-offdev backend: (\d+) allocations, (\d+) frees, (\d+) live, peak \d+ bytes, (\d+) bad frees, "
    r"(\d+) refused translations"
)


def run_both(fwdbwd, mdir, batch, out_dir: pathlib.Path, backend="cpu", expect_ok=True, **kw):
    """rrc_fwdbwd on `backend` and on `backend`-offdev; returns (bytes, bytes, stderr of offdev)."""
    res = []
    for be in (backend, backend + "-offdev"):
        out = out_dir / f"out_{be}.safetensors"
        p = run_rrc(fwdbwd, mdir, batch, out, backend=be, expect_ok=False, **kw)
        if expect_ok:
            assert p.returncode == 0, f"rrc_fwdbwd --backend {be} failed ({p.returncode}): {p.stderr}"
        res.append((p, out.read_bytes() if p.returncode == 0 else None))
    (pa, a), (pb, b) = res
    assert pa.returncode == pb.returncode, (pa.stderr, pb.stderr)
    m = OFFDEV_STATS.search(pb.stderr)
    assert m, pb.stderr
    allocs, frees, live, bad_frees, bad_ptrs = map(int, m.groups())
    assert allocs > 0 and allocs == frees and live == 0 and bad_frees == 0 and bad_ptrs == 0, pb.stderr
    assert OFFDEV_STATS.search(pa.stderr) is None
    return a, b, pa, pb


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("variant,targets,rslora", LORA_CASES, ids=[c[0] for c in LORA_CASES])
def test_offdev_backend_is_bit_identical(variant, targets, rslora, dtype, tiny, fwdbwd, tmp_path):
    """Every model variant (GQA / MHA / MQA, llama3 RoPE, tied heads, biases, q/k norm), f32 and
    bf16: window loss, per-micro-batch losses, every LoRA gradient, logits and every hidden state."""
    mdir = tiny(variant)
    pm = peft_model(mdir, torch.float32, targets, rslora=rslora, alpha=8.0)
    adapter = write_adapter(pm, tmp_path / "adapter")
    save_batch(build_window(MAIN_LAYOUT, seed=2), tmp_path / "batch")
    a, b, _, _ = run_both(
        fwdbwd,
        mdir,
        tmp_path / "batch",
        tmp_path,
        lora=adapter,
        dtype=dtype,
        targets=targets,
        rslora=rslora,
        extra=["--hidden"],
    )
    assert a == b
    out = safetensors_torch.load_file(str(tmp_path / "out_cpu.safetensors"))
    grads = [v for k, v in out.items() if k.startswith("grad.")]
    assert grads and all(g.abs().sum() > 0 for g in grads)  # a non-vacuous comparison


@pytest.mark.parametrize(
    "case",
    [
        "f32acc",
        "recompute_none",
        "microbatch_mean",
        "upper_layers",
        "no_lora",
        "no_grad",
        "seeded_init",
        "bad_token",
    ],
)
def test_offdev_backend_other_paths(case, tiny, fwdbwd, tmp_path):
    """The other code paths of rrc_fwdbwd on the off-device backend: fp32 accumulation, no
    recomputation, the microbatch_mean reduction, LoRA on upper layers only, base model only,
    loss only, rrc's seeded init, and a data-dependent error (same exit status and message)."""
    variant = "llama_3layers" if case == "upper_layers" else "qwen3_hd32_bias"
    mdir = tiny(variant)
    kw: dict[str, Any] = dict(dtype="bf16", extra=["--hidden"])
    backend = "cpu"
    if case not in ("no_lora", "seeded_init"):
        layers = [1, 2] if case == "upper_layers" else None
        pm = peft_model(mdir, torch.float32, ALL_TARGETS, layers=layers)
        kw["lora"] = write_adapter(pm, tmp_path / "adapter")
        kw["layers"] = layers
    if case == "f32acc":
        backend = "cpu-f32acc"
    elif case == "recompute_none":
        kw["extra"] += ["--recompute", "none"]
    elif case == "microbatch_mean":
        kw["reduction"] = "microbatch_mean"
    elif case == "no_lora":
        kw["lora"] = "none"
    elif case == "no_grad":
        kw["extra"] += ["--no-grad"]
    w = build_window(MAIN_LAYOUT, seed=3)
    if case == "bad_token":
        w.tokens[1, 0, 4] = VOCAB
    save_batch(w, tmp_path / "batch")
    a, b, pa, pb = run_both(
        fwdbwd, mdir, tmp_path / "batch", tmp_path, backend=backend, expect_ok=case != "bad_token", **kw
    )
    if case == "bad_token":
        assert pa.returncode == 2 and a is None and b is None
        assert "RRC_ERR_RANGE" in pa.stderr
        rest = [ln for ln in pb.stderr.splitlines() if "test-offdev backend:" not in ln]
        assert rest == pa.stderr.splitlines()
    else:
        assert a is not None and a == b
