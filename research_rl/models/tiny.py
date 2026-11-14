"""Tiny, fully offline models and tokenizers for tests, CI and smoke runs.

Nothing here downloads anything. The tokenizer is a character-level ``PreTrainedTokenizerFast``
built from an in-memory vocabulary, and the model is a randomly initialised Llama (SwiGLU FFN,
GQA attention) with a few thousand parameters. These objects exercise the same code paths as
real checkpoints in TRL, but **results obtained with them say nothing about real LLM quality**.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>")
DEFAULT_CHARS = (
    "0123456789+-*/=()^ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!:;'\"\n<>{}[]\\_$#"
)

# A minimal chat template so conversational datasets also work.
CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def build_char_tokenizer(chars: str = DEFAULT_CHARS) -> PreTrainedTokenizerFast:
    uniq = list(dict.fromkeys(chars))
    vocab = {t: i for i, t in enumerate([*SPECIAL_TOKENS, *uniq])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.padding_side = "left"
    return tok


def tiny_llama_config(
    vocab_size: int,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    max_position_embeddings: int = 256,
) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )


def build_tiny_llama(vocab_size: int, seed: int = 0, **config_overrides: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    model = LlamaForCausalLM(tiny_llama_config(vocab_size, **config_overrides))
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = 2
    model.generation_config.bos_token_id = 1
    return model


def save_tiny_model(path: str | Path, seed: int = 0, **config_overrides: int) -> Path:
    """Write a tiny Llama + char tokenizer to ``path`` (a normal HF model directory)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tok = build_char_tokenizer()
    model = build_tiny_llama(len(tok), seed=seed, **config_overrides)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path
