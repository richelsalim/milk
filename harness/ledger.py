"""Iteration ledger (FROZEN): results.tsv, metrics.json, events.jsonl.

results.tsv commit column: a row cannot contain the hash of the commit that includes
it, so rows are written with an empty commit field and BACKFILLED from the git log
(message prefix `iter <n>:`) at the next ledger write; `finish` seals the last row.
All sealed rows carry real short hashes. Never edited by hand.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from harness import git_ops

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results.tsv"
COLUMNS = ["iter", "commit", "status", "primary", "gauc", "ndcg5",
           "delta_vs_baseline", "train_sec", "seed", "description"]


def init_results() -> None:
    RESULTS.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")


def append_result(iter_n: int, status: str, metrics: dict | None, seeds: list[int],
                  description: str, delta_vs_baseline: float | None,
                  train_sec: float | None) -> None:
    backfill_commits()
    f = lambda v, spec="%.4f": (spec % v) if v is not None else ""  # noqa: E731
    row = [str(iter_n), "", status,
           f((metrics or {}).get("primary")), f((metrics or {}).get("gauc")),
           f((metrics or {}).get("ndcg5")), f(delta_vs_baseline, "%+.4f"),
           f(train_sec, "%.1f"), ",".join(map(str, seeds)),
           description.replace("\t", " ").replace("\n", " ")]
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write("\t".join(row) + "\n")


def backfill_commits() -> None:
    if not RESULTS.exists():
        return
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    changed = False
    for k, line in enumerate(lines[1:], start=1):
        parts = line.split("\t")
        if len(parts) == len(COLUMNS) and parts[1] == "":
            h = git_ops.find_iteration_commit(int(parts[0]))
            if h:
                parts[1] = h
                lines[k] = "\t".join(parts)
                changed = True
    if changed:
        RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    return [dict(zip(COLUMNS, line.split("\t"))) for line in lines[1:] if line.strip()]


def write_metrics(it_dir: Path, payload: dict) -> None:
    it_dir.mkdir(parents=True, exist_ok=True)
    (it_dir / "metrics.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def append_event(it_dir: Path, iter_n: int, attempt: int, etype: str, detail: str,
                 action: str) -> None:
    it_dir.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "iter": iter_n, "attempt": attempt,
           "type": etype, "detail": detail[:500], "action": action}
    with open(it_dir / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
