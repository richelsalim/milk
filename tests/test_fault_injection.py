"""Fault injection (FROZEN): beyond the phase-4 failure matrix — OOM, corrupted
checkpoint at refit, git dirty at start, disk-full on the submission write,
train.py deleting its own output. No traceback escapes the harness and the run
can continue or finish cleanly afterwards."""

import json

import pytest

from tests.test_harness import git_out, hyp, hz, make_ws, plant, read_results, start


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    return make_ws(tmp_path_factory)


def _no_tb(p):
    assert "Traceback (most recent call last)" not in p.stdout + p.stderr


def test_git_dirty_at_start(ws):
    (ws / "junk.py").write_text("x = 1\n", encoding="utf-8")
    p = hz(ws, "start", "--run-id", "fdirty")
    _no_tb(p)
    assert p.returncode == 2 and "dirty" in p.stdout
    (ws / "junk.py").unlink()


def test_oom_allocation(ws):
    tag = "foom"
    start(ws, tag, rss_cap_gb=1)
    plant(ws, q=0.5, extra=(
        "import numpy as _n, time as _t\n"
        "chunks = []\n"
        "for _ in range(256):  # toward 64 GB, killed by the RSS cap long before\n"
        "    chunks.append(_n.ones(31_250_000))\n"
        "    _t.sleep(0.05)\n"))
    hyp(ws, tag, 1)
    for _ in range(3):
        p = hz(ws, "iterate", "--desc", "oom")
        _no_tb(p)
    events = [json.loads(line) for line in
              (ws / "runs" / tag / "iterations" / "1" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert any(e["type"] == "oom" for e in events), events
    assert "ABANDONED" in p.stdout
    # run continues
    plant(ws, q=0.5)
    hyp(ws, tag, 2)
    p = hz(ws, "iterate", "--desc", "recovery", check=True)
    assert "DECISION: KEEP" in p.stdout
    assert git_out(ws, "status", "--porcelain") == ""


def test_train_deletes_own_output(ws):
    tag = "fdel"
    start(ws, tag)
    plant(ws, q=0.5)
    hyp(ws, tag, 1)
    hz(ws, "iterate", "--desc", "baseline", check=True)
    plant(ws, q=0.6, extra=(
        "import atexit, os\n"
        "atexit.register(lambda: [os.remove(out / f) for f in "
        "('val_scores.npy', 'test_scores.npy') if (out / f).exists()])\n"))
    hyp(ws, tag, 2)
    for _ in range(3):
        p = hz(ws, "iterate", "--desc", "self-deleting")
        _no_tb(p)
    events = [json.loads(line) for line in
              (ws / "runs" / tag / "iterations" / "2" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert any(e["type"] == "missing" for e in events), events
    assert "ABANDONED" in p.stdout


def test_disk_full_on_submission_write(ws):
    tag = "fdisk"
    start(ws, tag)
    # a plain file where the run's submission directory must go -> mkdir/write fails
    (ws / "submissions").mkdir(exist_ok=True)
    blocker = ws / "submissions" / tag
    blocker.write_bytes(b"disk full simulation")
    plant(ws, q=0.7)
    hyp(ws, tag, 1)
    p = hz(ws, "iterate", "--desc", "keep blocked by disk")
    _no_tb(p)
    assert "ERROR error" in p.stdout
    blocker.unlink()
    p = hz(ws, "iterate", "--desc", "retry after space freed", check=True)
    assert "DECISION: KEEP" in p.stdout
    assert git_out(ws, "status", "--porcelain") == ""


def test_corrupted_checkpoint_on_refit(ws):
    tag = "fcorrupt"
    start(ws, tag)
    plant(ws, q=0.7)
    hyp(ws, tag, 1)
    hz(ws, "iterate", "--desc", "keep", check=True)
    cfg = ws / "checkpoints" / tag / "best" / "config.json"
    cfg.write_bytes(b"\x00corrupt\xff")
    p = hz(ws, "finish", "--also-refit", "--run-id", tag)
    _no_tb(p)
    assert p.returncode == 0 and "refit failed" in p.stdout
    assert (ws / "runs" / tag / "resources.json").exists()
    rows = read_results(ws)
    assert rows[-1][1] != ""  # ledger sealed
    assert git_out(ws, "status", "--porcelain") == ""
