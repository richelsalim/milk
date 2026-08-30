"""train.py contract on the fixture: writes the three files with the right lengths."""

import json
import os
import subprocess
import sys

import numpy as np

import prepare

FIXTURE = str(prepare.REPO / "data" / "cache" / "fixture_small")


def test_train_contract(tmp_path):
    out = tmp_path / "t"
    env = {**os.environ, "KUAIRAND_DATA_ROOT": FIXTURE}
    proc = subprocess.run(
        [sys.executable, "train.py", "--out", str(out), "--time-budget", "20"],
        capture_output=True, text=True, env=env, cwd=prepare.REPO, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    os.environ["KUAIRAND_DATA_ROOT"] = FIXTURE
    try:
        n_val = prepare.load("val").height
        n_test = prepare.load("test").height
    finally:
        del os.environ["KUAIRAND_DATA_ROOT"]

    val = np.load(out / "val_scores.npy")
    test = np.load(out / "test_scores.npy")
    assert val.dtype == np.float32 and len(val) == n_val and np.isfinite(val).all()
    assert test.dtype == np.float32 and len(test) == n_test and np.isfinite(test).all()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["model"] == "fm" and cfg["features"] == "fm5" and cfg["seed"] == 0
    assert cfg["info"].get("rounds_used") is not None
