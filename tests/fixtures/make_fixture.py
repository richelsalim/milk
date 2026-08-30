"""Deterministic ~20k-row fixture at data/cache/fixture_small/ (FROZEN).

Samples users present in all three splits (seeded shuffle, greedy fill to the row
target), keeps their raw log rows as text (byte-faithful), copies the video tables,
filters the user table, then builds a normal parquet cache with prepare.build under
KUAIRAND_DATA_ROOT — so the fixture is a full parallel data root: every prepare.*
call, starter_kit data.load and submit.py --check work on it unchanged.
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import prepare  # noqa: E402

FIXTURE_ROOT = REPO / "data" / "cache" / "fixture_small"
TARGET_ROWS = 20_000
SEED = 0


def main() -> None:
    raw = REPO / "data" / "raw"
    log1, log2 = prepare.LOG_FILES["pure"]
    f1 = pl.read_csv(raw / log1, infer_schema=False)
    f2 = pl.read_csv(raw / log2, infer_schema=False)
    d2 = f2.with_columns(pl.col("date").cast(pl.Int64).alias("_d"))
    val_users = set(d2.filter(pl.col("_d") <= prepare.SPLITS["val"][1])["user_id"])
    test_users = set(d2.filter(pl.col("_d") >= prepare.SPLITS["test"][0])["user_id"])
    eligible = sorted(set(f1["user_id"]) & val_users & test_users, key=int)

    counts = dict(
        pl.concat([f1["user_id"], f2["user_id"]]).value_counts().iter_rows()
    )
    order = np.random.default_rng(SEED).permutation(len(eligible))
    picked, total = [], 0
    for i in order:
        u = eligible[i]
        picked.append(u)
        total += counts[u]
        if total >= TARGET_ROWS:
            break
    keep = set(picked)

    fraw = FIXTURE_ROOT / "raw"
    fraw.mkdir(parents=True, exist_ok=True)
    f1.filter(pl.col("user_id").is_in(keep)).write_csv(fraw / log1)
    f2.filter(pl.col("user_id").is_in(keep)).write_csv(fraw / log2)
    user_f, video_b, video_s = prepare.STATIC_FILES["pure"]
    pl.read_csv(raw / user_f, infer_schema=False).filter(
        pl.col("user_id").is_in(keep)
    ).write_csv(fraw / user_f)
    shutil.copyfile(raw / video_b, fraw / video_b)
    shutil.copyfile(raw / video_s, fraw / video_s)

    os.environ["KUAIRAND_DATA_ROOT"] = str(FIXTURE_ROOT)
    prepare.build("pure")
    print(f"fixture: {len(picked)} users, {total} log rows -> {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
