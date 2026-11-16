"""End-to-end CPU smoke of the rrc SFT pipeline (NOT a GPU validation).

    tiny qwen3 checkpoint (make_tiny_model.py) + tiny tokenizer (trained here)
      -> prepare_sft.py (JSONL -> .rrcbin) -> build/rrc_train ("backend": "cpu") for a few steps
      -> metrics.jsonl and checkpoint-N/ -> peft.PeftModel.from_pretrained

Everything runs on the CPU backend; this checks that the pieces fit together, not that training is
correct numerically (test_train_vs_torch.py does that) and not anything about a GPU. The loss is not
required to decrease.

Also here:
* the CPU build of rrc_train refuses "backend": "cuda" with a message, before creating anything;
* HOST-ONLY, opt-in (skipped unless CUDA_HOME points at a toolkit with include/cublasLt.h and
  RRC_TEST_HOST_SHIM_TRAIN=1; it takes minutes): the same run with
  "backend": "cuda" through build/rrc_train_shim, i.e. the real cuda/*.cu sources compiled with g++
  against a fake CUDA runtime that executes the kernels on host threads (tests/cuda_host). It is
  compared with the CPU run. This is not a GPU run either: no device, no real cuBLASLt.

Nothing is downloaded (HF_HUB_OFFLINE=1); the model, tokenizer, data and outputs live in tmp dirs.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytest.importorskip("tokenizers")
import make_tiny_model as mtm  # noqa: E402  (rrc/tools is put on sys.path by conftest)
from safetensors.torch import load_file  # noqa: E402

RRC_DIR = pathlib.Path(__file__).resolve().parents[1]
MODEL_VOCAB = 128  # make_tiny_model's shared vocab_size

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)

CORPUS = [
    "the cat sat on the mat",
    "a dog ran in the park",
    "what is two plus two",
    "it is four",
    "hello there how are you",
    "i am fine thanks",
    "name a color",
    "blue is a color",
    "count to three",
    "one two three",
]


def build_tokenizer(out: pathlib.Path) -> Any:
    """A WordLevel tokenizer trained here (whitespace split), with a chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    specials = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    tk.train_from_iterator(CORPUS, trainers.WordLevelTrainer(special_tokens=specials))
    tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<unk>"
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(str(out))
    return fast


def examples() -> list[dict[str, Any]]:
    pairs = [
        ("what is two plus two", "it is four"),
        ("hello there how are you", "i am fine thanks"),
        ("name a color", "blue is a color"),
        ("count to three", "one two three"),
    ]
    ex: list[dict[str, Any]] = []
    for i in range(6):
        for q, a in pairs:
            if i % 2:
                ex.append({"prompt": q, "completion": a})
            else:
                ex.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    return ex


@pytest.fixture(scope="module")
def pipeline(rrc_build: pathlib.Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Tiny qwen3 model, tokenizer and .rrcbin data (prepare_sft.py run as a subprocess)."""
    from transformers import AutoTokenizer

    root = tmp_path_factory.mktemp("e2e_cpu")
    mdir = root / "model"
    mtm.make_variant("qwen3", str(mdir), seed=7)
    model_cfg = json.loads((mdir / "config.json").read_text())
    assert model_cfg["model_type"] == "qwen3" and model_cfg["vocab_size"] == MODEL_VOCAB
    build_tokenizer(root / "tok")
    tok = AutoTokenizer.from_pretrained(str(root / "tok"), local_files_only=True)
    # prepare_sft writes len(tok) as the .rrcbin vocab_size; rrc refuses one above the model's
    assert len(tok) <= MODEL_VOCAB
    assert tok.pad_token_id is not None and 0 <= tok.pad_token_id < MODEL_VOCAB
    src = root / "train.jsonl"
    src.write_text("".join(json.dumps(e) + "\n" for e in examples()))
    data = root / "train.rrcbin"
    p = subprocess.run(
        [
            sys.executable,
            str(RRC_DIR / "tools/prepare_sft.py"),
            "--tokenizer",
            str(root / "tok"),
            "--input",
            str(src),
            "--output",
            str(data),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert p.returncode == 0, p.stderr
    summary = json.loads(p.stdout)
    assert data.is_file()
    return {
        "root": root,
        "model": mdir,
        "data": data,
        "pad": int(tok.pad_token_id),
        "summary": summary,
        "train": rrc_build / "rrc_train",
    }


def train_cfg(s: dict[str, Any], out: pathlib.Path, **over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(
        model_dir=str(s["model"]),
        data_path=str(s["data"]),
        output_dir=str(out),
        backend="cpu",
        dtype="bf16",
        seq_len=48,
        micro_batch_size=2,
        grad_accum_steps=2,
        max_steps=4,
        learning_rate=5e-3,
        lr_scheduler_type="linear",
        warmup_steps=0,
        pad_token_id=s["pad"],
        seed=11,
        lora=dict(r=4, alpha=8, target_modules="all-linear"),
        save_steps=2,
    )
    cfg.update(over)
    return cfg


def run(exe: pathlib.Path, cfg: dict[str, Any], path: pathlib.Path) -> subprocess.CompletedProcess:
    path.write_text(json.dumps(cfg, indent=1))
    return subprocess.run([str(exe), "--config", str(path)], capture_output=True, text=True, timeout=1800)


def read_metrics(out: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


def test_e2e_cpu_smoke_prepare_sft_train_and_peft_load(pipeline, tmp_path):
    """CPU smoke (not a GPU validation): prepare_sft -> rrc_train (cpu) -> metrics -> PeftModel."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    s = pipeline
    assert s["summary"]  # prepare_sft printed its JSON summary
    out = tmp_path / "run"
    p = run(s["train"], train_cfg(s, out), tmp_path / "train.json")
    assert p.returncode == 0, f"rrc_train failed ({p.returncode}): {p.stderr}"

    rows = read_metrics(out)
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    for r in rows:
        assert math.isfinite(r["loss"]) and r["loss"] > 0, r
        assert math.isfinite(r["grad_norm"]) and r["grad_norm"] >= 0, r
        assert math.isfinite(r["lr"]) and r["tokens"] > 0 and r["input_tokens"] >= r["tokens"], r
        assert r["step_time_s"] >= 0 and r["elapsed_s"] >= 0, r
    assert rows[-1]["tokens_seen"] == sum(r["tokens"] for r in rows)
    assert all(r["lr"] > 0 for r in rows)  # linear decay without warmup: lr * (4 - (s - 1)) / 4

    for step in (2, 4):
        ck = out / f"checkpoint-{step}"
        for f in ("adapter_model.safetensors", "adapter_config.json", "rrc_state.bin"):
            assert (ck / f).is_file(), (ck, f)

    ck = out / "checkpoint-4"
    saved = load_file(str(ck / "adapter_model.safetensors"))
    assert saved and all(k.startswith("base_model.model.") for k in saved)
    assert any(k.endswith("lora_B.weight") and torch.count_nonzero(v) > 0 for k, v in saved.items())

    base = AutoModelForCausalLM.from_pretrained(str(s["model"]), dtype=torch.float32)
    ids = torch.tensor([[1, 9, 10, 11, 12, 4, 13]]) % MODEL_VOCAB
    with torch.no_grad():
        base_logits = base(input_ids=ids).logits
    pm = PeftModel.from_pretrained(base, str(ck)).eval()
    # from_pretrained only warns about unmatched keys: check that every saved tensor was loaded
    state = pm.state_dict()
    for k, v in saved.items():
        loaded_key = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        assert loaded_key in state, k
        assert torch.equal(state[loaded_key].to(v.dtype), v), k
    n_lora = sum(1 for k in state if ".lora_" in k)
    assert n_lora == len(saved)
    with torch.no_grad():
        logits = pm(input_ids=ids).logits
    assert torch.isfinite(logits).all()
    assert not torch.equal(logits, base_logits)  # the trained adapter changes the output


def test_e2e_cpu_build_refuses_cuda_backend(pipeline, tmp_path):
    s = pipeline
    out = tmp_path / "never_created"
    cfg = train_cfg(s, out, backend="cuda")
    p = run(s["train"], cfg, tmp_path / "train.json")
    assert p.returncode == 2, (p.returncode, p.stderr)
    assert '"backend": "cuda" needs the CUDA build of rrc_train' in p.stderr, p.stderr
    assert "rrc_train_cuda" in p.stderr, p.stderr
    assert not out.exists()  # refused before any output is created
    # parsing accepts the value: --print-schedule needs neither the model nor the backend
    (tmp_path / "train.json").write_text(json.dumps(cfg))
    q = subprocess.run(
        [str(s["train"]), "--config", str(tmp_path / "train.json"), "--print-schedule"],
        capture_output=True,
        text=True,
    )
    assert q.returncode == 0, q.stderr
    assert len(json.loads(q.stdout)["lr"]) == cfg["max_steps"]
    # an unknown backend is still a config error
    bad = train_cfg(s, out, backend="gpu")
    r = run(s["train"], bad, tmp_path / "bad.json")
    assert r.returncode == 2 and '"backend" = "gpu"' in r.stderr, r.stderr


def _cuda_home() -> pathlib.Path | None:
    home = os.environ.get("CUDA_HOME", "")
    if home and (pathlib.Path(home) / "include/cublasLt.h").is_file():
        return pathlib.Path(home)
    return None


@pytest.mark.skipif(
    _cuda_home() is None or os.environ.get("RRC_TEST_HOST_SHIM_TRAIN") != "1",
    reason="HOST-ONLY, opt-in: set CUDA_HOME (with include/cublasLt.h) and RRC_TEST_HOST_SHIM_TRAIN=1",
)
def test_host_only_cuda_backend_trains_on_fake_runtime(pipeline, tmp_path):
    """HOST-ONLY, not a GPU run: "backend": "cuda" through build/rrc_train_shim (real cuda/*.cu on a
    fake CUDA runtime and host threads, a cuBLASLt stand-in) agrees with the CPU backend."""
    s = pipeline
    subprocess.run(
        ["make", "-s", "-C", str(RRC_DIR), "cuda-host-train", f"CUDA_HOME={_cuda_home()}"],
        check=True,
    )
    shim = RRC_DIR / "build/rrc_train_shim"
    assert shim.is_file()
    over = dict(dtype="f32", max_steps=3, save_steps=0, seq_len=32)
    runs = {}
    for backend, exe in (("cpu", s["train"]), ("cuda", shim)):
        out = tmp_path / backend
        p = run(exe, train_cfg(s, out, backend=backend, **over), tmp_path / f"{backend}.json")
        assert p.returncode == 0, f"{backend}: rrc_train failed ({p.returncode}): {p.stderr}"
        runs[backend] = (read_metrics(out), load_file(str(out / "checkpoint-3/adapter_model.safetensors")))
    (m_cpu, a_cpu), (m_cuda, a_cuda) = runs["cpu"], runs["cuda"]
    assert [r["step"] for r in m_cuda] == [1, 2, 3]
    for rc, rg in zip(m_cpu, m_cuda):
        assert rc["tokens"] == rg["tokens"] and rc["input_tokens"] == rg["input_tokens"]
        assert rg["loss"] == pytest.approx(rc["loss"], rel=1e-4), (rc, rg)
        assert rg["grad_norm"] == pytest.approx(rc["grad_norm"], rel=1e-3, abs=1e-6), (rc, rg)
    assert a_cpu.keys() == a_cuda.keys()
    for k in a_cpu:
        err = (a_cpu[k] - a_cuda[k]).abs().max().item()
        scale = a_cpu[k].abs().max().item()
        assert err <= 1e-3 * max(scale, 1e-3), (k, err, scale)
