"""Subprocess watchdog for train.py runs (FROZEN): stdout+stderr to a log file,
hard timeout at 2 x budget + grace (default 120 s, env KUAIRAND_WATCHDOG_GRACE_SEC
for the test suite), RSS sampling with a configurable cap."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]


def _kill_tree(proc: psutil.Process) -> None:
    try:
        for child in proc.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        proc.kill()
    except psutil.NoSuchProcess:
        pass


def run_train(out_dir: Path, seed: int, time_budget: int, rss_cap_gb: float,
              stdout_path: Path, model: str | None = None, features: str | None = None,
              extra_args: list[str] | None = None) -> dict:
    """Returns {status: ok|timeout|oom|error, returncode, elapsed, peak_rss_mb}."""
    grace = float(os.environ.get("KUAIRAND_WATCHDOG_GRACE_SEC", "120"))
    hard = 2 * time_budget + grace
    cmd = [sys.executable, "train.py", "--out", str(out_dir), "--seed", str(seed),
           "--time-budget", str(time_budget)]
    if model:
        cmd += ["--model", model]
    if features:
        cmd += ["--features", features]
    cmd += extra_args or []
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(stdout_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n=== seed {seed} @ {time.strftime('%F %T')}: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                env={**os.environ, "PYTHONUTF8": "1"})
        ps = psutil.Process(proc.pid)
        peak = 0
        status = "ok"
        while proc.poll() is None:
            try:
                rss = ps.memory_info().rss
                rss += sum(c.memory_info().rss for c in ps.children(recursive=True))
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                break
            if peak > rss_cap_gb * 1e9:
                status = "oom"
                _kill_tree(ps)
                break
            if time.time() - t0 > hard:
                status = "timeout"
                _kill_tree(ps)
                break
            time.sleep(0.2)
        proc.wait(timeout=30)
    elapsed = time.time() - t0
    rc = proc.returncode
    if status == "ok" and rc != 0:
        status = "error"
    return {"status": status, "returncode": rc, "elapsed": round(elapsed, 1),
            "peak_rss_mb": round(peak / 1e6, 1)}
