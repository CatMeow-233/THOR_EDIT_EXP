"""Shared fixtures: tiny offline tokenizer/model. Nothing is downloaded."""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(min(4, os.cpu_count() or 1))

from research_rl.models.tiny import build_char_tokenizer, build_tiny_llama, save_tiny_model  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer():
    return build_char_tokenizer()


@pytest.fixture
def tiny_model(tokenizer):
    return build_tiny_llama(len(tokenizer), seed=0)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A tiny HF model directory (weights + tokenizer) usable as `model="<path>"` in TRL."""
    return str(save_tiny_model(tmp_path_factory.mktemp("tiny_llama"), seed=0))
