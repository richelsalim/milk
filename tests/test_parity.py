"""Parity of the frozen layer against the published starter-kit numbers (FROZEN).

Needs the real cache: python prepare.py --build
"""

import re
import subprocess
import sys

import numpy as np
import pytest

import prepare


def test_fm_baseline_parity():
    """Official baseline via subprocess reproduces the published validation numbers."""
    proc = subprocess.run(
        [sys.executable, "baseline.py", "--model", "fm",
         "--data_dir", str(prepare.raw_dir().resolve())],
        cwd=prepare.STARTER_KIT, capture_output=True, text=True, encoding="utf-8",
        env=prepare._sk_env(), timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    m = re.search(r"valid\s+GAUC ([\d.]+) \| nDCG@5 ([\d.]+) \| primary ([\d.]+)", proc.stdout)
    assert m, proc.stdout[-2000:]
    gauc, ndcg5, primary = (float(m.group(i)) for i in (1, 2, 3))
    assert abs(primary - 0.6016) <= 0.003, f"primary {primary}"
    assert abs(gauc - 0.6674) <= 0.003, f"gauc {gauc}"
    assert abs(ndcg5 - 0.5357) <= 0.003, f"ndcg5 {ndcg5}"


def test_random_rung():
    n = prepare.load("val").height
    scores = np.random.default_rng(0).random(n)
    primary = prepare.evaluate("val", scores)["primary"]
    assert 0.46 <= primary <= 0.49, primary


def test_popularity_rung():
    """Kit popularity baseline (smoothed training-window long_view rate; published 0.5807).

    The literal 'count, 0 for unseen' variant scores 0.5435 on the real validation split,
    outside the pinned [0.55, 0.60] — starter-kit code takes precedence (decisions.md).
    """
    scores = prepare.popularity_scores(prepare.load("train"), prepare.load("val"))
    primary = prepare.evaluate("val", scores)["primary"]
    assert 0.55 <= primary <= 0.60, primary


def test_toy_example():
    """3 users exercising the pinned evaluate conventions.

    user A: labels by arrival [1, 0], scores [9, 1]      -> AUC 1, nDCG 1, GAUC weight 1
    user B: labels [0, 0, 0]                             -> zero positives: nDCG 0, counted
                                                            in the mean, excluded from GAUC
    user D: arrival [(5, y=1), (5, y=0), (3, y=1)]       -> tie at 5 (midrank AUC), weight 2
      AUC: pos@5 vs neg@5 tie = 0.5, pos@3 vs neg@5 = 0  -> (0.5 + 0) / 2 = 0.25
      nDCG@5: stable desc order labels [1, 0, 1]
        DCG  = (2^1-1)/log2(2) + 0 + (2^1-1)/log2(4) = 1 + 0.5 = 1.5
        IDCG = 1 + 1/log2(3) = 1.6309297535714574     -> nDCG = 0.9197207891481876
    GAUC = (1*1.0 + 2*0.25) / (1+2)                     = 0.5
    nDCG = (1 + 0 + 0.9197207891481876) / 3             = 0.6399069297160625
    primary                                             = 0.5699534648580313
    """
    ev = prepare._sk_module("evaluate").evaluate
    users = ["A", "A", "B", "B", "B", "D", "D", "D"]
    labels = [1, 0, 0, 0, 0, 1, 0, 1]
    scores = [9, 1, 3, 2, 1, 5, 5, 3]
    res = ev(users, labels, scores)
    assert res["GAUC"] == pytest.approx(0.5, abs=1e-12)
    assert res["nDCG@5"] == pytest.approx(0.6399069297160625, abs=1e-12)
    assert res["primary"] == pytest.approx(0.5699534648580313, abs=1e-12)
