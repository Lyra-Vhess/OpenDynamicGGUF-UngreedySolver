"""Pinned eval suites: dev (fast) vs release (full test splits)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def suite_names() -> tuple[str, ...]:
    return tuple(load_config()["suites"])


def suite_limits(name: str) -> dict[str, int | None]:
    """Per-task caps. ``None`` means the full official test split."""
    cfg = load_config()
    suites = cfg["suites"]
    if name not in suites:
        known = ", ".join(suites)
        raise ValueError(f"Unknown suite {name!r}. Choose one of: {known}")
    raw = suites[name]["limits"]
    return {task: (None if n is None else int(n)) for task, n in raw.items()}


def resolve_limits(
    suite: str,
    *,
    limit: int | None = None,
    tasks: list[str] | None = None,
) -> dict[str, int | None]:
    """Suite caps, optionally overridden by a single global ``--limit``.

    ``--limit 0`` forces the full split on every selected task.
    """
    cfg = load_config()
    selected = list(tasks or cfg["tasks"])
    if limit == 0:
        return {t: None for t in selected}
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be >= 0")
        return {t: int(limit) for t in selected}
    pinned = suite_limits(suite)
    return {t: pinned.get(t) for t in selected}
