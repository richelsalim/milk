"""Feature-layer tests (FROZEN): time safety, brute-force parity, guards, speed, determinism."""

import hashlib
import os
import re
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import prepare
from recsys.features import blocks as B
from recsys.features.spec import SPECS, _assemble, build

FULL = SPECS["full"]


# ---------------------------------------------------------------- synthetic 3-user log

def _toy_log(labels: list[int]) -> pl.DataFrame:
    """9 rows, 3 users, strictly increasing time_ms; labels[i] is row i's long_view."""
    n = len(labels)
    users = [1, 1, 1, 2, 2, 2, 3, 3, 3][:n]
    videos = [10, 11, 10, 10, 12, 11, 12, 12, 10][:n]
    return pl.DataFrame({
        "user_id": users, "video_id": videos,
        "date": [20220408 + i // 3 for i in range(n)],
        "hourmin": [900 + 10 * i for i in range(n)],
        "time_ms": [B.T0_MS + 3_600_000 * (i + 1) for i in range(n)],
        "is_click": [x ^ 1 for x in labels], "is_like": [0] * n, "is_follow": [0] * n,
        "is_comment": [0] * n, "is_forward": [0] * n, "is_hate": [0] * n,
        "long_view": labels, "play_time_ms": [1000.0 * (1 + 9 * x) for x in labels],
        "duration_ms": [10_000.0 + 500 * i for i in range(n)],
        "profile_stay_time": [0.0] * n, "comment_stay_time": [0.0] * n,
        "is_profile_enter": [0] * n, "is_rand": [0] * n, "tab": [1] * n,
    })


def _toy_tables() -> dict:
    videos = [10, 11, 12]
    video_basic = pl.DataFrame({
        "video_id": videos, "author_id": [100, 100, 200],
        "video_type": ["NORMAL"] * 3, "upload_dt": ["2022-04-01"] * 3,
        "upload_type": ["Web"] * 3, "music_id": [7, 7, 8],
        "music_type": [9.0, 9.0, 4.0], "tag": ["1,2", "2", "3"],
    })
    video_stat = pl.DataFrame({"video_id": videos,
                               **{c: [1.0, 2.0, 3.0] for c in B.STAT_COLS}})
    user = pl.DataFrame({
        "user_id": [1, 2, 3], "user_active_degree": ["high_active", "full_active", "day_new"],
        "is_lowactive_period": [0, 0, 0], "is_live_streamer": [0, 1, 0],
        "is_video_author": [0, 0, 1], "follow_user_num": [5, 10, 0],
        "fans_user_num": [1, 2, 3], "friend_user_num": [0, 1, 0],
        "register_days": [100, 200, 300],
        **{f"onehot_feat{i}": [1, 2, 3] for i in range(18)},
    })
    return {"user": user, "video_basic": video_basic, "video_stat": video_stat}


def _toy_X(labels):
    rows = _toy_log(labels)
    X, meta, _ = _assemble(FULL, rows, rows, _toy_tables(), "train")
    return X, meta


def test_time_safety_later_labels_do_not_matter():
    base = [1, 0, 1, 0, 1, 0, 1, 1, 0]
    X0, meta = _toy_X(base)
    # shuffle the labels of every row strictly later than row 4 (preserves the multiset)
    shuffled = base[:5] + [base[i] for i in (7, 8, 5, 6)]
    assert shuffled != base
    X1, _ = _toy_X(shuffled)
    np.testing.assert_array_equal(X0[:5], X1[:5],
                                  err_msg="features of earlier rows changed when later labels shuffled")


def test_time_safety_earlier_label_flip_is_seen():
    base = [1, 0, 1, 0, 1, 0, 1, 1, 0]
    X0, meta = _toy_X(base)
    flipped = list(base)
    flipped[0] ^= 1  # user 1's first row; user 1's later rows must see it
    X1, _ = _toy_X(flipped)
    col = meta["columns"].index("hu_lv_rate")
    assert X0[1, col] != X1[1, col]
    assert X0[2, col] != X1[2, col]


def test_val_rows_match_brute_force():
    """Real data: recompute three feature families for a handful of val rows by brute force."""
    X, meta, group = build("full", "val")
    train = prepare.load("train")
    val = prepare.load("val")
    gmean = train["long_view"].mean()
    cols = {c: i for i, c in enumerate(meta["columns"])}
    vb = prepare.tables()["video_basic"].select("video_id", "author_id")
    tr_author = train.join(vb, on="video_id", how="left")
    val_author = val.join(vb, on="video_id", how="left")

    rng_rows = [0, 1000, 50_000, 100_000, 124_908]
    for r in rng_rows:
        u = val["user_id"][r]
        v = val["video_id"][r]
        tu = train.filter(pl.col("user_id") == u)
        exp_user = (tu["long_view"].sum() + 20 * gmean) / (tu.height + 20)
        assert X[r, cols["hu_lv_rate"]] == pytest.approx(exp_user, rel=1e-4)

        tv = train.filter(pl.col("video_id") == v)
        exp_vid = (tv["long_view"].sum() + 20 * gmean) / (tv.height + 20)
        assert X[r, cols["hi_vid_lv_rate"]] == pytest.approx(exp_vid, rel=1e-4)

        a = val_author["author_id"][r]
        ta = tr_author.filter((pl.col("user_id") == u)
                              & (pl.col("author_id").fill_null(-1) == (a if a is not None else -1)))
        exp_ua = (ta["long_view"].sum() + 20 * gmean) / (ta.height + 20)
        assert X[r, cols["x_ua_lv_rate"]] == pytest.approx(exp_ua, rel=1e-4)


def test_no_label_columns_and_group_order():
    for split in ("train", "val", "test"):
        X, meta, group = build("full", split)
        assert not set(meta["columns"]) & set(prepare.FEEDBACK_COLS)
        expect = prepare.load(split)["user_id"].to_numpy()
        np.testing.assert_array_equal(group, expect)
        assert X.dtype == np.float32 and np.isfinite(X).all()


def test_full_build_under_60s_single_core():
    fdir = prepare.cache_dir() / "features"
    for p in fdir.glob("full-train-*"):
        p.unlink()
    env = {**os.environ, "POLARS_MAX_THREADS": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "recsys.features", "build", "--spec", "full", "--split", "train"],
        capture_output=True, text=True, env=env, cwd=prepare.REPO, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    sec = float(re.search(r"in ([\d.]+)s", proc.stdout).group(1))
    print(f"\nfull/train single-core build: {sec:.1f}s")
    assert sec < 60, f"full spec build took {sec:.1f}s"


def test_two_builds_identical_hashes():
    def h(spec, split):
        X, meta, group = build(spec, split, use_cache=False)
        return hashlib.sha256(X.tobytes() + group.tobytes()).hexdigest(), meta

    h1, m1 = h("full", "val")
    h2, m2 = h("full", "val")
    assert h1 == h2 and m1 == m2


def test_val_history_end_leak_guard():
    with pytest.raises(ValueError):
        build("full", "val", history_end=20220422)
