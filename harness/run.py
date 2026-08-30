"""Run state (runs/<tag>/run.json) and paths (FROZEN)."""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def now() -> float:
    """Wall clock; tests monkeypatch harness.run.now for the ceiling cases."""
    return time.time()


class Run:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dir = REPO / "runs" / run_id
        self.path = self.dir / "run.json"
        self.state: dict = {}

    @classmethod
    def load(cls, run_id: str) -> "Run":
        r = cls(run_id)
        if not r.path.exists():
            raise FileNotFoundError(f"{r.path} missing — run: python -m harness start --run-id {run_id}")
        r.state = json.loads(r.path.read_text(encoding="utf-8"))
        return r

    @classmethod
    def latest(cls) -> "Run":
        runs = sorted((REPO / "runs").glob("*/run.json"), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise FileNotFoundError("no runs/<tag>/run.json found — start a run first")
        return cls.load(runs[-1].parent.name)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def it_dir(self, i: int) -> Path:
        return self.dir / "iterations" / str(i)

    # convenience -----------------------------------------------------------
    @property
    def status(self) -> str:
        return self.state["status"]

    @property
    def branch(self) -> str:
        return f"autoresearch/{self.run_id}"

    def elapsed(self) -> float:
        return now() - self.state["started_at"]

    def iterations_used(self) -> int:
        return self.state["counts"]["scored"] + self.state["counts"]["abandoned"]
