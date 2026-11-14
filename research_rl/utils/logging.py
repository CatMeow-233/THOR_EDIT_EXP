"""Consistent loggers for ResearchRL code, with an optional rank-0-only mode for multi-process runs."""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | rank %(rank)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_TAG = "_research_rl_handler"


def get_rank() -> int:
    """Global process rank: ``torch.distributed`` if initialised, else ``$RANK``, else 0."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # pragma: no cover - torch always importable here
        pass
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


class _RankFilter(logging.Filter):
    """Injects ``record.rank``; with ``rank_zero_only`` drops records below ERROR on ranks != 0."""

    def __init__(self, rank_zero_only: bool) -> None:
        super().__init__()
        self.rank_zero_only = rank_zero_only

    def filter(self, record: logging.LogRecord) -> bool:
        rank = get_rank()
        record.rank = rank
        return not (self.rank_zero_only and rank != 0 and record.levelno < logging.ERROR)


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    *,
    rank_zero_only: bool = False,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Return ``logging.getLogger(name)`` with exactly one ResearchRL handler attached.

    Repeated calls reconfigure (level, rank filter, stream) instead of stacking handlers. With
    ``rank_zero_only`` non-zero ranks only emit ERROR and CRITICAL records, so failures on any
    rank stay visible. The logger does not propagate to the root logger (no duplicate lines).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        if getattr(h, _HANDLER_TAG, False):
            logger.removeHandler(h)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    setattr(handler, _HANDLER_TAG, True)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.addFilter(_RankFilter(rank_zero_only))
    logger.addHandler(handler)
    return logger
