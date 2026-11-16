"""PyTorch + PEFT reference runs for the rrc training tests (rrc/tests/test_train_vs_torch.py).

* ``pack_reference``: an independent Python implementation of rrc's shuffling and packing (the spec in
  rrc/include/rrc/data.h), used to check the batches rrc_train trained on.
* ``train_reference``: the optimizer loop of rrc_train written with torch / transformers / PEFT:
  ``torch.optim.AdamW(foreach=False)``, ``transformers.get_scheduler``,
  ``torch.nn.utils.clip_grad_norm_``, gradient accumulation with the loss of the whole window
  (sum of per-token cross entropy / labelled tokens of the window). Each packed document is run
  through HF separately (position ids 0.., no cross-document attention), which is what rrc's packing
  must be equivalent to.

Nothing is downloaded; models and adapters are local directories.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MASK64 = (1 << 64) - 1


def splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def permutation(seed: int, epoch: int, n: int) -> list[int]:
    _, x = splitmix64_next(seed & MASK64)
    state = x ^ ((0xD1B54A32D192ED03 * (epoch + 1)) & MASK64)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        state, r = splitmix64_next(state)
        j = r % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def pack_reference(
    docs: list[tuple[list[int], list[int]]],
    seq_len: int,
    rows: int,
    count: int,
    seed: int = 0,
    shuffle: bool = True,
    truncation: str = "truncate",
    pad_id: int = 0,
    start: tuple[int, int] = (0, 0),
) -> tuple[dict[str, list[list[list[int]]]], list[tuple[int, int]]]:
    """The first `count` micro-batches from cursor `start` = (epoch, index). Returns
    ({"tokens", "labels", "position_ids", "doc_ids"} -> [count][rows][seq_len], cursors after each)."""
    usable = [i for i, (t, _) in enumerate(docs) if truncation == "truncate" or len(t) <= seq_len]
    n = len(usable)
    epoch, index = start
    while index >= n:
        index -= n
        epoch += 1
    cache: dict[int, list[int]] = {}

    def order(e: int) -> list[int]:
        if e not in cache:
            cache.clear()
            cache[e] = permutation(seed, e, n) if shuffle else list(range(n))
        return cache[e]

    out: dict[str, list[list[list[int]]]] = {k: [] for k in ("tokens", "labels", "position_ids", "doc_ids")}
    cursors = []
    for _ in range(count):
        mb: dict[str, list[list[int]]] = {k: [] for k in out}
        for _r in range(rows):
            tok: list[int] = []
            lab: list[int] = []
            pos: list[int] = []
            did: list[int] = []
            k = 0
            while len(tok) < seq_len:
                d = usable[order(epoch)[index]]
                t, m = docs[d]
                ln = min(len(t), seq_len)
                if ln > seq_len - len(tok):
                    break
                tok += t[:ln]
                lab += [t[i] if m[i] else -100 for i in range(ln)]
                pos += list(range(ln))
                did += [k] * ln
                k += 1
                index += 1
                if index == n:
                    index, epoch = 0, epoch + 1
            pad = seq_len - len(tok)
            tok += [pad_id] * pad
            lab += [-100] * pad
            pos += list(range(pad))
            did += [k] * pad
            for key, v in zip(("tokens", "labels", "position_ids", "doc_ids"), (tok, lab, pos, did)):
                mb[key].append(v)
        for key in out:
            out[key].append(mb[key])
        cursors.append((epoch, index))
    return out, cursors


@dataclasses.dataclass
class RefConfig:
    learning_rate: float
    lr_scheduler_type: str
    warmup_steps: int
    max_steps: int
    max_grad_norm: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    loss_reduction: str = "window_tokens"


def split_documents(tokens: Any, labels: Any, doc_ids: Any) -> list[tuple[Any, Any]]:
    """Per row, the contiguous runs of equal doc id -> [(token ids, labels)] (torch long tensors)."""
    out = []
    for r in range(tokens.shape[0]):
        d = doc_ids[r]
        start = 0
        T = tokens.shape[1]
        for t in range(1, T + 1):
            if t == T or d[t] != d[start]:
                out.append((tokens[r, start:t].long(), labels[r, start:t].long()))
                start = t
    return out


def train_reference(
    model_dir: str,
    adapter_dir: str,
    windows: list[dict[str, Any]],
    cfg: RefConfig,
    dtype: str = "f32",
) -> dict[str, Any]:
    """windows[s] = {"tokens", "labels", "doc_ids"} [M, B, T] tensors of optimizer step s + 1.
    Returns per-step loss / lr / grad_norm and the final LoRA parameters (PEFT saved-key names)."""
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, get_scheduler

    torch_dtype = torch.float32 if dtype == "f32" else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype, attn_implementation="eager")
    pm = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    pm.train()  # dropout is 0 in every tiny config; train() only matters for PEFT's bookkeeping
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        assert p.dtype == torch.float32
    opt = torch.optim.AdamW(
        params, lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay, foreach=False
    )
    sched = get_scheduler(
        cfg.lr_scheduler_type, opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps
    )
    losses, lrs, norms, tokens = [], [], [], []
    for w in windows:
        M = w["tokens"].shape[0]
        mb_docs = [split_documents(w["tokens"][m], w["labels"][m], w["doc_ids"][m]) for m in range(M)]
        counts = [sum(int((lab[1:] != -100).sum()) for _, lab in docs) for docs in mb_docs]
        total = sum(counts)
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        for m, docs in enumerate(mb_docs):
            for ids, lab in docs:
                tgt = lab[1:]
                n = int((tgt != -100).sum())
                if n == 0:
                    continue
                logits = pm(input_ids=ids[None]).logits[0]
                s = F.cross_entropy(logits[:-1].float(), tgt, ignore_index=-100, reduction="sum")
                scale = 1.0 / total if cfg.loss_reduction == "window_tokens" else 1.0 / (counts[m] * M)
                (s * scale).backward()
                loss_val += float(s.detach()) * scale
        if cfg.max_grad_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        else:
            norm = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
        losses.append(loss_val)
        norms.append(float(norm))
        tokens.append(total)
    final = {
        n.replace(".default", ""): p.detach().clone() for n, p in pm.named_parameters() if p.requires_grad
    }
    return {"loss": losses, "lr": lrs, "grad_norm": norms, "tokens": tokens, "params": final}
