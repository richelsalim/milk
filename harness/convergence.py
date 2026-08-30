"""Convergence rule (FROZEN). The starter kit ships the rule as parameters only
(baseline_scores.json: convergence_rule = {epsilon: 0.002, N: 3}), not as code —
this implements exactly:

  converged when the best validation primary over the last N scored iterations does
  not exceed the best before that window by more than eps.

Also stops at max-iters (scored + abandoned) and at the wall-clock ceiling measured
from started_at, whichever comes first. Abandoned iterations never enter the window
(they produced no score).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def shipped_defaults() -> tuple[float, int]:
    rule = json.loads((REPO / "starter_kit" / "baseline_scores.json").read_text(encoding="utf-8"))["convergence_rule"]
    return float(rule["epsilon"]), int(rule["N"])


def converged(scored_primaries: list[float], eps: float, n: int) -> bool:
    if len(scored_primaries) <= n:
        return False
    window = scored_primaries[-n:]
    before = scored_primaries[:-n]
    return max(window) <= max(before) + eps


def window_deltas(scored_primaries: list[float], n: int) -> list[float]:
    """Per-window-entry gain over the pre-window best, for the summary block."""
    if len(scored_primaries) <= n:
        return []
    before = max(scored_primaries[:-n])
    return [p - before for p in scored_primaries[-n:]]


def stop_reason(scored_primaries, iterations_used, elapsed_sec, *,
                eps, n, max_iters, wall_clock_hours) -> str | None:
    if converged(scored_primaries, eps, n):
        return "converged"
    if iterations_used >= max_iters:
        return "cap"
    if elapsed_sec >= wall_clock_hours * 3600:
        return "ceiling"
    return None
