"""Read-only access to gated-loop decision logs under an allowed runs directory.

A run named ``<name>`` is either ``<runs_dir>/<name>.jsonl`` or ``<runs_dir>/<name>/decisions.jsonl``
(the JSONL log :class:`~research_rl.capability.loop.GatedLoop` writes: one
``DecisionRecord.to_dict()`` per line). Names must match :data:`RUN_NAME_RE` (no path separators,
no leading dot), the resolved file must lie inside the resolved runs directory (so a symlink
pointing outside is refused) and must be a regular file no larger than ``max_bytes``.

The records are untrusted data: only known fields of the expected types are copied into the
summaries; everything else is ignored. Nothing here touches a model.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

__all__ = ["RUN_NAME_RE", "RunAccessError", "list_runs", "read_decisions", "resolve_run"]

RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
MAX_LOG_BYTES = 32 * 1024 * 1024


class RunAccessError(PermissionError):
    """The run name is invalid or its log resolves outside the runs directory."""


def _contained(root: Path, candidate: Path) -> Path | None:
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(root):
        raise RunAccessError("the run log resolves outside the runs directory")
    return resolved if resolved.is_file() else None


def resolve_run(runs_dir: Path, name: str) -> Path:
    """Path of the decision log of run ``name`` (``RunAccessError`` / ``FileNotFoundError``)."""
    if not isinstance(name, str) or not RUN_NAME_RE.fullmatch(name):
        raise RunAccessError(f"invalid run name {name!r}")
    root = runs_dir.resolve(strict=True)
    for candidate in (root / f"{name}.jsonl", root / name / "decisions.jsonl"):
        found = _contained(root, candidate)
        if found is not None:
            return found
    raise FileNotFoundError(f"no decision log for run {name!r}")


def list_runs(runs_dir: Path | None) -> list[dict[str, Any]]:
    """Runs with a decision log directly under ``runs_dir`` (sorted by name)."""
    if runs_dir is None:
        return []
    root = runs_dir.resolve(strict=True)
    out: dict[str, dict[str, Any]] = {}
    for entry in sorted(root.iterdir()):
        if entry.suffix == ".jsonl" and entry.is_file():
            name = entry.stem
        elif entry.is_dir() and (entry / "decisions.jsonl").is_file():
            name = entry.name
        else:
            continue
        if not RUN_NAME_RE.fullmatch(name) or name in out:
            continue
        try:
            path = resolve_run(runs_dir, name)
        except (RunAccessError, FileNotFoundError):
            continue
        out[name] = {"name": name, "bytes": path.stat().st_size}
    return [out[k] for k in sorted(out)]


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str(value: Any, limit: int = 500) -> str | None:
    return value[:limit] if isinstance(value, str) else None


def _test(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "slice": _str(raw.get("slice"), 200),
        "n": _int(raw.get("n")),
        "point": _num(raw.get("point")),
        "bound": _num(raw.get("bound")),
        "passed": raw.get("passed") if isinstance(raw.get("passed"), bool) else None,
        "method": _str(raw.get("method"), 100),
        "alpha": _num(raw.get("alpha")),
        "p_value": _num(raw.get("p_value")),
        "margin": _num(raw.get("margin")),
    }


def _verdict(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    retention = raw.get("retention")
    return {
        "decision_index": _int(raw.get("decision_index")),
        "transfer": _test(raw.get("transfer")),
        "retention": [t for t in (_test(x) for x in retention) if t is not None]
        if isinstance(retention, list)
        else [],
        "alarm": raw.get("alarm") if isinstance(raw.get("alarm"), bool) else None,
        "transfer_ok": raw.get("transfer_ok") if isinstance(raw.get("transfer_ok"), bool) else None,
        "retention_ok": raw.get("retention_ok") if isinstance(raw.get("retention_ok"), bool) else None,
    }


def _record(raw: dict[str, Any]) -> dict[str, Any]:
    verdicts = raw.get("verdicts")
    parsed = (
        [v for v in (_verdict(x) for x in verdicts) if v is not None] if isinstance(verdicts, list) else []
    )
    groups = raw.get("groups_reverted")
    return {
        "decision_index": _int(raw.get("decision_index")),
        "update_index": _int(raw.get("update_index")),
        "decision": _str(raw.get("decision"), 50),
        "reason": _str(raw.get("reason")),
        "shrink_lambda": _num(raw.get("shrink_lambda")),
        "groups_reverted": [g[:200] for g in groups if isinstance(g, str)]
        if isinstance(groups, list)
        else [],
        "reward_v1": _num(raw.get("reward_v1")),
        "reward_v2": _num(raw.get("reward_v2")),
        "alarm": raw.get("alarm") if isinstance(raw.get("alarm"), bool) else None,
        "override": _str(raw.get("override")),
        "verdict_count": len(parsed),
        "first_verdict": parsed[0] if parsed else None,
        "last_verdict": parsed[-1] if parsed else None,
    }


def read_decisions(
    runs_dir: Path, name: str, *, limit: int = 1000, max_bytes: int = MAX_LOG_BYTES
) -> dict[str, Any]:
    """Parse the decision log of run ``name``: the last ``limit`` records, summarised.

    A line that is not JSON is skipped (a torn last line is normal while a run writes; other bad
    lines are counted in ``skipped_lines``). ``reward_change_v1`` is ``reward_v1 -`` the first
    record's ``reward_v1``; ``accepted_transfer_gain`` is the running sum of the first verdict's
    transfer point estimate over ACCEPT decisions (a rough capability trace from the gate's own
    evidence, not an independent measurement).
    """
    path = resolve_run(runs_dir, name)
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"decision log of {name!r} is {size} bytes; the limit is {max_bytes}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[dict[str, Any]] = []
    skipped = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except (ValueError, RecursionError):
            # ValueError: not JSON (JSONDecodeError) or an int over Python's int-string limit;
            # RecursionError: nested too deeply to parse
            if i != len(lines) - 1:
                skipped += 1
            continue
        if not isinstance(raw, dict):
            skipped += 1
            continue
        records.append(_record(raw))
    first_reward = next((r["reward_v1"] for r in records if r["reward_v1"] is not None), None)
    cumulative = 0.0
    for r in records:
        r["reward_change_v1"] = (
            None if first_reward is None or r["reward_v1"] is None else r["reward_v1"] - first_reward
        )
        fv = r["first_verdict"]
        point = None if fv is None or fv["transfer"] is None else fv["transfer"]["point"]
        if r["decision"] == "accept" and point is not None:
            cumulative += point
        r["accepted_transfer_gain"] = cumulative
    total = len(records)
    return {
        "name": name,
        "total_records": total,
        "skipped_lines": skipped,
        "records": records[-limit:] if limit > 0 else [],
        "note": (
            "reward_change_v1 is the training reward relative to the first decision; accepted_transfer_gain "
            "sums the gate's transfer point estimates over accepted blocks (a rough trace, not an "
            "independent capability measurement)"
        ),
    }
