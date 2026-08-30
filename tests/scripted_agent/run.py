"""Scripted research-agent driver: plays the agent against the real harness in a
temporary clone, so the real repo is never dirtied.

small mode: the QUALITY-knob train stub on the 20k fixture, budget 5 s, the exact
9-step sequence from IMPLEMENTATION.md phase 5, full expected-vs-observed table.
full mode: the real train.py on the real data, budget 300 s, behaviour asserts only.
Exit code 0 iff every assertion matched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUB = Path(__file__).resolve().parent / "train_stub.py"


class Driver:
    def __init__(self, mode: str):
        self.mode = mode
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.ws = Path(tempfile.gettempdir()) / f"scripted-{ts}"
        self.checks: list[tuple[str, str, str, bool]] = []

    # -- infrastructure ------------------------------------------------------
    def clone(self):
        subprocess.run(["git", "clone", "--quiet", str(REPO), str(self.ws)], check=True)
        root = (REPO / "data") if self.mode == "full" else (REPO / "data" / "cache" / "fixture_small")
        self.env = {**os.environ, "KUAIRAND_DATA_ROOT": str(root),
                    "KUAIRAND_WATCHDOG_GRACE_SEC": "3" if self.mode == "small" else "120",
                    "GIT_AUTHOR_NAME": "scripted", "GIT_AUTHOR_EMAIL": "s@s",
                    "GIT_COMMITTER_NAME": "scripted", "GIT_COMMITTER_EMAIL": "s@s"}
        if (self.ws / "runs" / "scripted").exists():
            # a previous scripted ledger may be committed in the repo (phase 6 copies it
            # under runs/); reset it in the clone or events.jsonl counts double up
            self.git("rm", "-r", "-q", "runs/scripted")
            self.git("commit", "-qm", "scripted: reset prior run ledger")

    def git(self, *args) -> str:
        return subprocess.run(["git", "-C", str(self.ws), *args], capture_output=True,
                              text=True, encoding="utf-8").stdout.strip()

    def hz(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "harness", *args], cwd=self.ws,
                              capture_output=True, text=True, encoding="utf-8",
                              env=self.env, timeout=1800)

    def expect(self, name: str, expected, observed):
        ok = expected == observed
        self.checks.append((name, str(expected), str(observed), ok))
        return ok

    def hyp(self, i: int, text: str):
        d = self.ws / "runs" / "scripted" / "iterations" / str(i)
        d.mkdir(parents=True, exist_ok=True)
        (d / "hypothesis.md").write_text(text, encoding="utf-8")

    # -- stub manipulation ---------------------------------------------------
    def set_q(self, q: float, nan: bool = False, loop: bool = False):
        src = (self.ws / "tests" / "scripted_agent" / "train_stub.py").read_text(encoding="utf-8")
        src = src.replace("QUALITY = 0.20", f"QUALITY = {q}")
        if loop:
            src = src.replace("# FAULT", "import time as _t\n_t.sleep(120)")
        if nan:
            src = src.replace("np.save(out / \"val_scores.npy\", val)",
                              "val[:] = float(\"nan\")\nnp.save(out / \"val_scores.npy\", val)")
        (self.ws / "train.py").write_text(src, encoding="utf-8")

    def break_syntax(self):
        (self.ws / "train.py").write_text("def broken(:\n", encoding="utf-8")

    # -- assertions shared ---------------------------------------------------
    def results(self):
        lines = (self.ws / "results.tsv").read_text(encoding="utf-8").splitlines()
        return [line.split("\t") for line in lines[1:]]

    def all_events(self):
        events = []
        for p in sorted((self.ws / "runs" / "scripted" / "iterations").glob("*/events.jsonl")):
            events += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        return events

    def check_submission(self) -> bool:
        raw = Path(self.env["KUAIRAND_DATA_ROOT"]) / "raw"
        proc = subprocess.run(
            [sys.executable, "submit.py", "--check", "--split", "test", "--data_dir",
             str(raw), str(self.ws / "submissions" / "scripted" / "final.csv")],
            cwd=self.ws / "starter_kit", capture_output=True, text=True, encoding="utf-8",
            env={**self.env, "PYTHONUTF8": "1"})
        return proc.returncode == 0

    def artifact_set(self, i: int, scored: bool) -> bool:
        d = self.ws / "runs" / "scripted" / "iterations" / str(i)
        need = ["hypothesis.md", "diff.patch", "stdout.log"]
        need.append("metrics.json" if scored else "events.jsonl")
        return all((d / f).exists() for f in need)

    def one_commit_each(self, n: int) -> bool:
        log = self.git("log", "--format=%s").splitlines()
        return all(sum(s.startswith(f"iter {i}:") for s in log) == 1 for i in range(1, n + 1))

    # -- sequences -----------------------------------------------------------
    def run_small(self):
        self.clone()
        stub = (self.ws / "tests" / "scripted_agent" / "train_stub.py").read_text(encoding="utf-8")
        (self.ws / "train.py").write_text(stub, encoding="utf-8")
        self.git("add", "train.py")
        self.git("commit", "-qm", "scripted: stub train.py")
        p = self.hz("start", "--run-id", "scripted", "--time-budget", "5")
        assert p.returncode == 0, p.stdout + p.stderr

        def it(desc):
            return self.hz("iterate", "--desc", desc)

        self.set_q(0.20)
        self.hyp(1, "baseline reproduction")
        self.expect("01 keep", True, "DECISION: KEEP" in it("q=0.20 baseline").stdout)

        self.set_q(0.30)
        self.hyp(2, "raise quality to 0.30")
        self.expect("02 keep", True, "DECISION: KEEP" in it("q=0.30").stdout)

        self.hyp(3, "broken change, then fixed at 0.10")
        self.break_syntax()
        p = it("syntax error attempt")
        self.expect("03 error event", True, "ERROR error" in p.stdout)
        self.set_q(0.10)
        self.expect("03 revert", True, "DECISION: REVERT" in it("fixed, q=0.10").stdout)

        self.set_q(0.90)
        self.hyp(4, "raise quality to 0.90")
        self.expect("04 keep", True, "DECISION: KEEP" in it("q=0.90").stdout)

        self.hyp(5, "infinite loop idea")
        self.set_q(0.95, loop=True)
        for k in (1, 2, 3):
            p = it("infinite loop")
        self.expect("05 abandoned", True, "ABANDONED" in p.stdout)

        self.hyp(6, "nan bug, then fixed at 0.50")
        self.set_q(0.50, nan=True)
        p = it("nan attempt")
        self.expect("06 nan event", True, "ERROR nan" in p.stdout)
        self.set_q(0.50)
        self.expect("06 revert", True, "DECISION: REVERT" in it("fixed, q=0.50").stdout)

        self.set_q(0.90)
        self.hyp(7, "retry 0.90 (equal, not better)")
        self.expect("07 revert (equal)", True, "DECISION: REVERT" in it("q=0.90 again").stdout)

        self.set_q(0.85)
        self.hyp(8, "slightly below best")
        p = it("q=0.85")
        self.expect("08 revert", True, "DECISION: REVERT" in p.stdout)
        self.expect("08 STOP converged", True, "STOP <converged>" in p.stdout)

        p = it("after stop")
        self.expect("09 refused with STOP", True, p.returncode == 2 and "STOP" in p.stdout)

        rows = self.results()
        self.expect("results rows", 8, len(rows))
        self.expect("statuses", ["keep", "keep", "revert", "keep", "abandoned",
                                 "revert", "revert", "revert"], [r[2] for r in rows])
        kept = [int(r[0]) for r in rows if r[2] == "keep"]
        self.expect("kept iterations", [1, 2, 4], kept)
        final = self.ws / "submissions" / "scripted" / "final.csv"
        it4 = self.ws / "submissions" / "scripted" / "iter_4.csv"
        self.expect("final.csv is the 04 submission", True,
                    final.read_bytes() == it4.read_bytes())
        self.expect("final.csv passes --check", True, self.check_submission())
        ev = self.all_events()
        counts = {t: sum(e["type"] == t for e in ev) for t in
                  ("error", "timeout", "nan", "stop")}
        self.expect("one error event", 1, counts["error"])
        self.expect("three timeout events", 3, counts["timeout"])
        self.expect("one nan event", 1, counts["nan"])
        self.expect("one stop event", 1, counts["stop"])
        state = json.loads((self.ws / "runs" / "scripted" / "run.json").read_text(encoding="utf-8"))
        self.expect("run.json converged", "converged", state["status"])
        self.expect("converged at iteration 8", 8, state["counts"]["scored"] + state["counts"]["abandoned"])
        c4 = [line.split()[0] for line in self.git("log", "--format=%h %s").splitlines()
              if "iter 4:" in line][0]
        self.expect("mutable surface == iter 4 commit", "",
                    self.git("diff", c4, "--", "train.py", "recsys"))
        for i in range(1, 9):
            self.expect(f"artifacts iter {i}", True, self.artifact_set(i, scored=i != 5))
        self.expect("one commit per iteration", True, self.one_commit_each(8))
        self.expect("tree clean", "", self.git("status", "--porcelain"))

    def run_full(self):
        self.clone()
        p = self.hz("start", "--run-id", "scripted", "--time-budget", "300")
        assert p.returncode == 0, p.stdout + p.stderr

        self.hyp(1, "baseline reproduction")
        p = self.hz("iterate", "--desc", "baseline reproduction")
        self.expect("01 keep", True, "DECISION: KEEP" in p.stdout)

        self.hyp(2, "unchanged rerun (deterministic, equal, must revert)")
        p = self.hz("iterate", "--desc", "unchanged rerun")
        self.expect("02 revert (equal)", True, "DECISION: REVERT" in p.stdout)

        self.hyp(3, "broken edit, then fixed")
        train = (self.ws / "train.py").read_text(encoding="utf-8")
        self.break_syntax()
        p = self.hz("iterate", "--desc", "syntax error attempt")
        self.expect("03 error event", True, "ERROR error" in p.stdout)
        (self.ws / "train.py").write_text(train, encoding="utf-8")
        p = self.hz("iterate", "--desc", "fixed rerun")
        self.expect("03 scored", True, "DECISION:" in p.stdout)

        p = self.hz("finish")
        self.expect("finish ok", 0, p.returncode)
        self.expect("final.csv passes --check", True, self.check_submission())
        for i in range(1, 4):
            self.expect(f"artifacts iter {i}", True, self.artifact_set(i, scored=True))
        self.expect("one commit per iteration", True, self.one_commit_each(3))
        self.expect("tree clean", "", self.git("status", "--porcelain"))

    # -- report --------------------------------------------------------------
    def report(self) -> int:
        w = max(len(c[0]) for c in self.checks) + 2
        print(f"\n{'check'.ljust(w)}{'expected'.ljust(28)}observed")
        bad = 0
        for name, exp, obs, ok in self.checks:
            mark = "ok " if ok else "FAIL"
            bad += not ok
            print(f"{mark} {name.ljust(w)}{exp[:26].ljust(28)}{obs[:40]}")
        print(f"\n{len(self.checks) - bad}/{len(self.checks)} checks passed "
              f"(workspace: {self.ws})")
        return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.scripted_agent.run")
    ap.add_argument("--fixture", choices=["small", "full"], default="small")
    a = ap.parse_args()
    d = Driver(a.fixture)
    try:
        d.run_small() if a.fixture == "small" else d.run_full()
    except AssertionError as e:
        print(f"FATAL: {e}")
        return 1
    return d.report()


if __name__ == "__main__":
    sys.exit(main())
