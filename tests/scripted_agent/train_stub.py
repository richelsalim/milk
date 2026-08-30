"""Scripted-agent train.py stub: deterministic scores from a QUALITY knob.

val/test scores = QUALITY x time-safe popularity feature + (1 - QUALITY) x seeded
noise, so validation primary rises monotonically with QUALITY and every KEEP/REVERT
in the scripted sequence is predictable. The driver rewrites the QUALITY line (and
sometimes injects faults) between iterations, playing the research agent.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402

QUALITY = 0.20
# FAULT

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--time-budget", type=int, default=300)
ap.add_argument("--model", default="stub")
ap.add_argument("--features", default=None)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

train = prepare.load("train")
pop = (train.group_by("video_id")
       .agg(pl.len().alias("n"), pl.col("long_view").sum().alias("p")))
gmean = train["long_view"].mean()


def scores(split, rng):
    frame = prepare.load(split)
    joined = frame.select("video_id").join(pop, on="video_id", how="left")
    rate = ((joined["p"].fill_null(0) + 20 * gmean) / (joined["n"].fill_null(0) + 20)).to_numpy()
    noise = rng.random(frame.height)
    return (QUALITY * rate / rate.max() + (1 - QUALITY) * noise).astype(np.float32)


rng = np.random.default_rng(a.seed)
val = scores("val", rng)
test = scores("test", rng)
np.save(out / "val_scores.npy", val)
np.save(out / "test_scores.npy", test)
(out / "config.json").write_text(json.dumps(
    {"model": "stub", "quality": QUALITY, "seed": a.seed}), encoding="utf-8")
print(f"stub done q={QUALITY}")
