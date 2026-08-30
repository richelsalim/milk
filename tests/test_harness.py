"""Harness tests (FROZEN): run the real CLI in a disposable git workspace on the
fixture data root, with planted train.py stubs. Target: whole file under 3 minutes."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import prepare
from harness.convergence import converged

REPO = Path(prepare.__file__).resolve().parent
FIXTURE = str(REPO / "data" / "cache" / "fixture_small")


# ---------------------------------------------------------------- convergence unit

def test_convergence_sequences():
    eps, n = 0.002, 3
    assert not converged([0.60], eps, n)
    # a window iteration still gaining more than eps over the pre-window best -> continue
    assert not converged([0.60, 0.61, 0.6, 0.6], eps, n)
    # must converge: three scored iterations, none above best-before + eps
    assert converged([0.62, 0.620, 0.621, 0.6215], eps, n)
    # must not converge: a window iteration gains more than eps over the pre-window best
    assert not converged([0.62, 0.61, 0.61, 0.623], eps, n)
    # gain of exactly eps is NOT an improvement
    assert converged([0.62, 0.622, 0.62, 0.62], eps, n)
    # fewer than n+1 scored iterations never converge
    assert not converged([0.62, 0.62, 0.62], eps, n)
    # abandoned iterations never enter the list, so a long run of them changes nothing
    seq = [0.62, 0.63, 0.6301, 0.6302, 0.6303]
    assert converged(seq, eps, n)


# ---------------------------------------------------------------- workspace

STUB_OK = '''\
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--time-budget", type=int, default=300)
ap.add_argument("--model", default="fm")
ap.add_argument("--features", default=None)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
n_val = prepare.load("val").height
n_test = prepare.load("test").height
q = QUALITY
rng = np.random.default_rng(a.seed)
lv = prepare.load("val")["long_view"].to_numpy().astype(np.float64)
val = (q * lv + (1 - q) * rng.random(n_val)).astype(np.float32)
test = rng.random(n_test).astype(np.float32)
EXTRA
np.save(out / "val_scores.npy", val)
np.save(out / "test_scores.npy", test)
(out / "config.json").write_text(json.dumps({"model": "stub", "q": q, "seed": a.seed}), encoding="utf-8")
print("primary: 0.99")  # lie — the harness must recompute and ignore this
'''


def make_ws(tmp_path_factory) -> Path:
    ws = Path(tmp_path_factory.mktemp("harness_ws"))
    for f in ("prepare.py", "train.py", ".gitignore", ".gitattributes"):
        shutil.copyfile(REPO / f, ws / f)
    for d in ("starter_kit", "harness"):
        shutil.copytree(REPO / d, ws / d)
    shutil.copytree(REPO / "recsys", ws / "recsys",
                    ignore=shutil.ignore_patterns("__pycache__"))
    g = ["git", "-C", str(ws)]
    for cmd in (["init", "-q", "-b", "main"], ["add", "-A"],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(g + cmd, check=True, capture_output=True)
    return ws


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    return make_ws(tmp_path_factory)


def env_for(ws):
    return {**os.environ, "KUAIRAND_DATA_ROOT": FIXTURE,
            "KUAIRAND_WATCHDOG_GRACE_SEC": "3",
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def hz(ws, *args, check=False):
    proc = subprocess.run([sys.executable, "-m", "harness", *args], cwd=ws,
                          capture_output=True, text=True, encoding="utf-8",
                          env=env_for(ws), timeout=300)
    if check and proc.returncode != 0:
        raise AssertionError(f"harness {args} rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return proc


def plant(ws, q=0.5, extra=""):
    (ws / "train.py").write_text(STUB_OK.replace("QUALITY", str(q)).replace("EXTRA", extra), encoding="utf-8")


def hyp(ws, tag, i, text="hypothesis: test"):
    d = ws / "runs" / tag / "iterations" / str(i)
    d.mkdir(parents=True, exist_ok=True)
    (d / "hypothesis.md").write_text(text, encoding="utf-8")


def start(ws, tag, **kw):
    args = ["start", "--run-id", tag, "--time-budget", "3", "--rss-cap-gb",
            str(kw.pop("rss_cap_gb", 16))]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return hz(ws, *args, check=True)


def git_out(ws, *args):
    return subprocess.run(["git", "-C", str(ws), *args], capture_output=True,
                          text=True).stdout.strip()


def read_results(ws):
    lines = (ws / "results.tsv").read_text(encoding="utf-8").splitlines()
    return [line.split("\t") for line in lines[1:]]


# ---------------------------------------------------------------- scenarios

def test_refusals_keep_revert_and_ledger(ws):
    tag = "t1"
    start(ws, tag)
    # refuses without hypothesis.md
    p = hz(ws, "iterate", "--desc", "no hypothesis")
    assert p.returncode == 2 and "REFUSED" in p.stdout

    # iteration 1: keep (q=0.6); train.py prints a fake primary the ledger must ignore
    plant(ws, q=0.6)
    hyp(ws, tag, 1)
    p = hz(ws, "iterate", "--desc", "baseline stub", check=True)
    assert "DECISION: KEEP" in p.stdout
    rows = read_results(ws)
    assert rows[0][2] == "keep"
    assert float(rows[0][3]) < 0.95, "harness must recompute metrics, not trust train.py"
    assert (ws / "submissions" / tag / "final.csv").exists()
    n_test = 2528
    assert len((ws / "submissions" / tag / "final.csv").read_text(encoding="utf-8").splitlines()) == n_test + 1
    assert git_out(ws, "status", "--porcelain") == ""

    kept_train = (ws / "train.py").read_bytes()

    # iteration 2: worse (q=0.1) -> revert restores train.py byte for byte
    plant(ws, q=0.1, extra="MARKER = 1\n")
    hyp(ws, tag, 2)
    p = hz(ws, "iterate", "--desc", "worse stub", check=True)
    assert "DECISION: REVERT" in p.stdout
    assert (ws / "train.py").read_bytes() == kept_train
    assert git_out(ws, "status", "--porcelain") == ""

    # exactly one commit per iteration: init + start + iter1 + iter2
    log = git_out(ws, "log", "--format=%s").splitlines()
    assert sum(s.startswith("iter 1:") for s in log) == 1
    assert sum(s.startswith("iter 2:") for s in log) == 1

    # results.tsv commit backfill: row 1 sealed with a real hash by iteration 2's write
    rows = read_results(ws)
    assert rows[0][1] != "" and rows[0][1] in git_out(ws, "log", "--format=%h")


def test_failure_matrix(ws):
    tag = "t3"
    start(ws, tag)
    plant(ws, q=0.5)
    hyp(ws, tag, 1)
    hz(ws, "iterate", "--desc", "baseline", check=True)

    cases = [
        ("error", None),                                # syntax error
        ("timeout", "import time\ntime.sleep(60)\n"),
        ("nan", "val[:] = float('nan')\n"),
        ("shape", "val = val[:-3]\n"),
        ("missing", "raise SystemExit(0)\n"),
    ]
    i = 2
    for etype, payload in cases:
        if payload is None:
            (ws / "train.py").write_text("def broken(:\n", encoding="utf-8")
        else:
            plant(ws, q=0.5, extra=payload)
        hyp(ws, tag, i, f"hypothesis: {etype}")
        for _ in range(3):
            p = hz(ws, "iterate", "--desc", f"fail {etype}")
            assert "Traceback (most recent call last)" not in p.stdout + p.stderr, etype
        events = [json.loads(line) for line in
                  (ws / "runs" / tag / "iterations" / str(i) / "events.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        assert any(e["type"] == etype for e in events), (etype, events)
        assert "ABANDONED" in p.stdout
        assert git_out(ws, "status", "--porcelain") == ""
        i += 1

    # oom: allocate far past a tiny cap; verify a fresh run with rss cap 0.3 GB
    tag2 = "t4"
    start(ws, tag2, rss_cap_gb=0.3)
    plant(ws, q=0.5, extra="import numpy as _n, time\nbig = _n.ones(200_000_000, dtype=_n.float64)\ntime.sleep(10)\n")
    hyp(ws, tag2, 1)
    for _ in range(3):
        p = hz(ws, "iterate", "--desc", "oom")
        assert "Traceback (most recent call last)" not in p.stdout + p.stderr
    events = [json.loads(line) for line in
              (ws / "runs" / tag2 / "iterations" / "1" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["type"] == "oom" for e in events), events
    assert "ABANDONED" in p.stdout

    # abandoned counts toward the cap but not the convergence window
    state = json.loads((ws / "runs" / tag2 / "run.json").read_text(encoding="utf-8"))
    assert state["counts"]["abandoned"] == 1 and state["scored_primaries"] == []


def test_stop_conditions(ws):
    # convergence STOP then refusal
    tag = "t5"
    start(ws, tag, max_iters=50)
    qs = [0.9, 0.15, 0.14, 0.13]  # keep, then three no-gain scored iterations
    for i, q in enumerate(qs, start=1):
        plant(ws, q=q)
        hyp(ws, tag, i)
        p = hz(ws, "iterate", "--desc", f"q={q}", check=True)
    assert "STOP <converged>" in p.stdout
    p = hz(ws, "iterate", "--desc", "after stop")
    assert p.returncode == 2 and "STOP" in p.stdout

    # wall-clock ceiling via a faked started_at
    tag = "t6"
    start(ws, tag, wall_clock_hours=6)
    rj = ws / "runs" / tag / "run.json"
    state = json.loads(rj.read_text(encoding="utf-8"))
    state["started_at"] -= 7 * 3600
    rj.write_text(json.dumps(state), encoding="utf-8")
    hyp(ws, tag, 1)
    plant(ws, q=0.5)
    p = hz(ws, "iterate", "--desc", "past ceiling")
    assert p.returncode == 2 and "STOP <ceiling>" in p.stdout

    # iteration cap
    tag = "t7"
    start(ws, tag, max_iters=1)
    hyp(ws, tag, 1)
    plant(ws, q=0.5)
    p = hz(ws, "iterate", "--desc", "only one", check=True)
    assert "STOP <cap>" in p.stdout


def test_finish_writes_resources_and_bundle(ws):
    tag = "t8"
    start(ws, tag)
    plant(ws, q=0.7)
    hyp(ws, tag, 1)
    hz(ws, "iterate", "--desc", "keep", check=True)
    p = hz(ws, "finish", check=True)
    assert "final.csv re-validated" in p.stdout
    res = json.loads((ws / "runs" / tag / "resources.json").read_text(encoding="utf-8"))
    assert res["scored"] == 1 and res["kept"] == 1 and "iteration_interpretation" in res
    assert (ws / "runs" / tag / "bundle.tar.gz").exists()
    assert git_out(ws, "status", "--porcelain") == ""
    # last results row sealed with a real commit hash
    rows = read_results(ws)
    assert rows[-1][1] != ""
