"""Tokenize SFT data offline into rrc's .rrcbin format (rrc/include/rrc/data.h, rrc/docs/README.md).

    python rrc/tools/prepare_sft.py --tokenizer /path/to/local/tokenizer --input data.jsonl \
        --output data.rrcbin [--chat-template template.jinja] [--no-eos]

The tokenizer is loaded from a LOCAL directory only (``local_files_only=True``; nothing is downloaded).
Each JSONL line is one example, either

* ``{"messages": [{"role": "...", "content": "..."}, ...]}`` -- rendered with the tokenizer's chat
  template (or ``--chat-template``); only the tokens of assistant turns are training targets, or
* ``{"prompt": "...", "completion": "..."}`` -- ``tokenize(prompt)`` (with the tokenizer's special
  tokens, e.g. BOS) followed by ``tokenize(completion)`` (without) and EOS (unless ``--no-eos``);
  the completion and EOS tokens are the targets.

Assistant spans of a conversation are found on the rendered text: for assistant message i, the span
is ``[len(render(messages[:i], add_generation_prompt=True)), len(render(messages[:i+1])))``; the
template must be prefix-stable (each rendering a prefix of the next and of the full conversation),
otherwise the example is refused. The full text is tokenized once (``add_special_tokens=False``,
since the template already contains them) with offsets, and a token is a target iff its first
character lies in an assistant span (so the end-of-turn marker the template emits inside the span is
a target too).

.rrcbin layout (little-endian): ``b"RRCBIN01"``, u64 num_docs, u64 vocab_size, u64 total_tokens,
then per document u32 len, u32 tokens[len], u8 label_mask[len]. ``vocab_size`` is ``len(tokenizer)``
(or ``--vocab-size``). The file is written to ``OUTPUT.tmp`` and renamed. Examples that tokenize to
zero tokens are skipped and counted. A JSON summary is printed on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections.abc import Iterable, Sequence
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MAGIC = b"RRCBIN01"
HEADER = struct.Struct("<8sQQQ")


def write_rrcbin(path: str, docs: Sequence[tuple[Sequence[int], Sequence[int]]], vocab_size: int) -> None:
    """Write documents [(tokens, label_mask)] atomically (path.tmp + rename)."""
    if not docs:
        raise ValueError("no documents to write")
    if not 1 <= vocab_size < 2**31:
        raise ValueError(f"vocab_size {vocab_size} out of range")
    total = 0
    for i, (toks, mask) in enumerate(docs):
        if len(toks) == 0 or len(toks) != len(mask):
            raise ValueError(f"document {i}: empty or tokens/mask length mismatch")
        if min(toks) < 0 or max(toks) >= vocab_size:
            raise ValueError(f"document {i}: token id outside [0, {vocab_size})")
        if any(m not in (0, 1) for m in mask):
            raise ValueError(f"document {i}: label_mask values must be 0 or 1")
        total += len(toks)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(MAGIC, len(docs), vocab_size, total))
        for toks, mask in docs:
            f.write(struct.pack("<I", len(toks)))
            f.write(struct.pack(f"<{len(toks)}I", *toks))
            f.write(bytes(mask))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_rrcbin(path: str) -> tuple[int, list[tuple[list[int], list[int]]]]:
    """Read a .rrcbin file (minimal checks; the C reader is the validating one)."""
    with open(path, "rb") as f:
        data = f.read()
    magic, n, vocab, total = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("bad magic")
    pos = HEADER.size
    docs = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", data, pos)
        toks = list(struct.unpack_from(f"<{ln}I", data, pos + 4))
        mask = list(data[pos + 4 + 4 * ln : pos + 4 + 5 * ln])
        docs.append((toks, mask))
        pos += 4 + 5 * ln
    if pos != len(data) or sum(len(t) for t, _ in docs) != total:
        raise ValueError("inconsistent file")
    return vocab, docs


def _render(tok: Any, messages: list[dict[str, Any]], template: str | None, gen_prompt: bool) -> str:
    if not messages:
        return ""
    out = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=gen_prompt, chat_template=template
    )
    if not isinstance(out, str):
        raise TypeError("apply_chat_template(tokenize=False) did not return a string")
    return out


def tokenize_messages(
    tok: Any, messages: list[dict[str, Any]], template: str | None = None
) -> tuple[list[int], list[int]]:
    if template is None and not getattr(tok, "chat_template", None):
        raise ValueError("'messages' example but the tokenizer has no chat template (pass --chat-template)")
    full = _render(tok, messages, template, False)
    spans = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        prefix = _render(tok, messages[:i], template, True)
        upto = _render(tok, messages[: i + 1], template, False)
        if not upto.startswith(prefix) or not full.startswith(upto):
            raise ValueError(
                "the chat template is not prefix-stable for this conversation (rendering of the first i "
                "messages is not a prefix of the next rendering); cannot locate the assistant tokens"
            )
        spans.append((len(prefix), len(upto)))
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = enc["offset_mapping"]
    mask = [1 if any(s <= a < e for s, e in spans) else 0 for a, _ in offs]
    return ids, mask


def tokenize_prompt_completion(
    tok: Any, prompt: str, completion: str, add_eos: bool = True
) -> tuple[list[int], list[int]]:
    p = list(tok(prompt, add_special_tokens=True)["input_ids"])
    c = list(tok(completion, add_special_tokens=False)["input_ids"])
    if add_eos:
        if tok.eos_token_id is None:
            raise ValueError("the tokenizer has no eos_token_id (pass --no-eos)")
        c.append(int(tok.eos_token_id))
    return p + c, [0] * len(p) + [1] * len(c)


def tokenize_example(
    tok: Any, ex: dict[str, Any], template: str | None = None, add_eos: bool = True
) -> tuple[list[int], list[int]]:
    if "messages" in ex:
        if set(ex) - {"messages"}:
            raise ValueError(f"unexpected keys next to 'messages': {sorted(set(ex) - {'messages'})}")
        return tokenize_messages(tok, ex["messages"], template)
    if "prompt" in ex and "completion" in ex:
        if set(ex) - {"prompt", "completion"}:
            raise ValueError(f"unexpected keys: {sorted(set(ex) - {'prompt', 'completion'})}")
        if not isinstance(ex["prompt"], str) or not isinstance(ex["completion"], str):
            raise ValueError("'prompt' and 'completion' must be strings")
        return tokenize_prompt_completion(tok, ex["prompt"], ex["completion"], add_eos)
    raise ValueError("an example needs 'messages', or 'prompt' and 'completion'")


def convert(
    tok: Any,
    examples: Iterable[dict[str, Any]],
    output: str,
    template: str | None = None,
    add_eos: bool = True,
    vocab_size: int | None = None,
) -> dict[str, int]:
    vocab = int(vocab_size if vocab_size is not None else len(tok))
    docs = []
    skipped = 0
    for i, ex in enumerate(examples):
        try:
            ids, mask = tokenize_example(tok, ex, template, add_eos)
        except Exception as e:
            raise ValueError(f"example {i}: {e}") from e
        if not ids:
            skipped += 1
            continue
        docs.append((ids, mask))
    write_rrcbin(output, docs, vocab)
    return {
        "num_docs": len(docs),
        "vocab_size": vocab,
        "total_tokens": sum(len(t) for t, _ in docs),
        "label_tokens": sum(sum(m) for _, m in docs),
        "skipped_empty": skipped,
    }


def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            ex = json.loads(line)
            if not isinstance(ex, dict):
                raise ValueError(f"{path}:{n}: not a JSON object")
            yield ex


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="local tokenizer directory (never downloaded)")
    ap.add_argument("--input", required=True, help="JSONL file")
    ap.add_argument("--output", required=True, help=".rrcbin file to write")
    ap.add_argument("--chat-template", help="file with a Jinja chat template overriding the tokenizer's")
    ap.add_argument("--no-eos", action="store_true", help="do not append EOS to prompt/completion examples")
    ap.add_argument("--vocab-size", type=int, help="header vocab_size (default len(tokenizer))")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    if not os.path.isdir(args.tokenizer):
        ap.error(f"--tokenizer {args.tokenizer} is not a local directory")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    template = None
    if args.chat_template:
        with open(args.chat_template, encoding="utf-8") as f:
            template = f.read()
    stats = convert(tok, read_jsonl(args.input), args.output, template, not args.no_eos, args.vocab_size)
    json.dump(stats, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
