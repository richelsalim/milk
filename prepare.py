"""prepare.py — FROZEN: data and evaluation layer (see CLAUDE.md).

Loads splits in exactly starter_kit/data.py's row order, builds/verifies the parquet
cache, exposes the organizer metric and the submission writer. The research agent
never edits this file and never reads data/raw/ directly.

Layout under the active data root (env KUAIRAND_DATA_ROOT, default ./data):
  raw/    the KuaiRand CSVs (never read by the mutable surface)
  cache/  {train,val,test}.parquet, static tables, sizes.json, MANIFEST.sha256

The 20k test fixture is a full parallel root (data/cache/fixture_small/) with the
same layout, so submit.py --check and every prepare.* call work on it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent
STARTER_KIT = REPO / "starter_kit"

SPLITS = {"train": (20220408, 20220421), "val": (20220422, 20220428), "test": (20220429, 20220508)}
HISTORY_END = 20220421

# Post-impression columns: never present in the test frame (data policy 1).
FEEDBACK_COLS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter",
]

SPLIT_SIZES = {"pure": {"train": 1_141_112, "val": 124_909, "test": 170_588}}

LOG_FILES = {
    "pure": ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"),
    "1k": ("log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv"),
    "27k": ("log_standard_4_08_to_4_21_27k_part1.csv", "log_standard_4_22_to_5_08_27k.csv"),
}
STATIC_FILES = {
    "pure": ("user_features_pure.csv", "video_features_basic_pure.csv",
             "video_features_statistic_pure.csv"),
    "1k": ("user_features_1k.csv", "video_features_basic_1k.csv",
           "video_features_statistic_1k.csv"),
    "27k": ("user_features_27k.csv", "video_features_basic_27k.csv",
            "video_features_statistic_27k.csv"),
}

LOG_SCHEMA = {
    "user_id": pl.Int64, "video_id": pl.Int64, "date": pl.Int64, "hourmin": pl.Int64,
    "time_ms": pl.Int64, "is_click": pl.Int64, "is_like": pl.Int64, "is_follow": pl.Int64,
    "is_comment": pl.Int64, "is_forward": pl.Int64, "is_hate": pl.Int64, "long_view": pl.Int64,
    "play_time_ms": pl.Float64, "duration_ms": pl.Float64, "profile_stay_time": pl.Float64,
    "comment_stay_time": pl.Float64, "is_profile_enter": pl.Int64, "is_rand": pl.Int64,
    "tab": pl.Int64,
}

FM_PARITY = {"primary": 0.6016, "gauc": 0.6674, "ndcg5": 0.5357, "tol": 0.003}
RANDOM_RANGE = (0.46, 0.49)
POPULARITY_RANGE = (0.55, 0.60)


def data_root() -> Path:
    return Path(os.environ.get("KUAIRAND_DATA_ROOT", str(REPO / "data")))


def _is_real_pure_root(dataset: str) -> bool:
    return dataset == "pure" and data_root().resolve() == (REPO / "data").resolve()


def raw_dir() -> Path:
    return data_root() / "raw"


def cache_dir(dataset: str = "pure") -> Path:
    base = data_root() / "cache"
    return base if dataset == "pure" else base / dataset


def _norm_split(split: str) -> str:
    split = {"valid": "val", "validation": "val"}.get(split, split)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return split


def _sk_module(name: str):
    """Import a starter_kit module byte-for-byte, without touching sys.path."""
    spec = importlib.util.spec_from_file_location(f"sk_{name}", STARTER_KIT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sk_env() -> dict:
    # starter kit prints unicode; Windows pipes default to cp1252 without this
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


# ---------------------------------------------------------------- load / tables

def load(split: str, dataset: str = "pure", history_end: int | None = None) -> pl.DataFrame:
    """Split frame in the exact row order of starter_kit data.load().

    "test" carries no feedback columns. history_end (train only) widens or narrows
    the training window; anything >= 20220429 would touch test labels and raises.
    """
    split = _norm_split(split)
    cd = cache_dir(dataset)
    path = cd / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: python prepare.py --build --dataset {dataset}")
    frame = pl.read_parquet(path)
    if split == "test":
        return frame.drop([c for c in FEEDBACK_COLS if c in frame.columns])
    if split == "train":
        he = HISTORY_END if history_end is None else int(history_end)
        if he >= SPLITS["test"][0]:
            raise ValueError(f"history_end {he} reaches into the test window")
        if he < HISTORY_END:
            frame = frame.filter(pl.col("date") <= he)
        elif he > HISTORY_END:
            extra = pl.read_parquet(cd / "val.parquet").filter(pl.col("date") <= he)
            frame = pl.concat([frame, extra])
    return frame


def tables(dataset: str = "pure") -> dict[str, pl.DataFrame]:
    """The static KuaiRand tables: user, video_basic, video_stat."""
    cd = cache_dir(dataset)
    out = {}
    for key in ("user", "video_basic", "video_stat"):
        path = cd / f"{key}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run: python prepare.py --build --dataset {dataset}")
        out[key] = pl.read_parquet(path)
    return out


# ---------------------------------------------------------------- evaluate

def evaluate(split: str, scores, dataset: str = "pure") -> dict:
    """Score via starter_kit/evaluate.py. Raises on split == 'test' (no local labels)."""
    split = _norm_split(split)
    if split == "test":
        raise ValueError("test labels are not available locally; submit final.csv instead")
    frame = load(split, dataset=dataset)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != frame.height:
        raise ValueError(f"{len(scores)} scores for {frame.height} rows in {split}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or Inf")
    res = _sk_module("evaluate").evaluate(
        frame["user_id"].to_list(), frame["long_view"].to_list(), scores.tolist()
    )
    return {"gauc": res["GAUC"], "ndcg5": res["nDCG@5"], "primary": res["primary"]}


# ---------------------------------------------------------------- submission

def write_submission(split: str, scores, path: str | Path, dataset: str = "pure") -> Path:
    """Write the pinned CSV schema, then validate with starter_kit submit.py --check."""
    split = _norm_split(split)
    if split == "train":
        raise ValueError("submissions are for val or test")
    frame = load(split, dataset=dataset)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != frame.height:
        raise ValueError(f"{len(scores)} scores for {frame.height} rows in {split}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or Inf")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uids, vids = frame["user_id"].to_list(), frame["video_id"].to_list()
    with open(path, "w", newline="") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        fh.writelines(
            f"{i},{u},{v},{s:.9g}\n" for i, (u, v, s) in enumerate(zip(uids, vids, scores))
        )
    kit_split = "valid" if split == "val" else "test"
    proc = subprocess.run(
        [sys.executable, "submit.py", "--check", "--split", kit_split,
         "--data_dir", str(raw_dir().resolve()), str(path.resolve())],
        cwd=STARTER_KIT, capture_output=True, text=True, encoding="utf-8", env=_sk_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"submit.py --check failed:\n{proc.stdout}\n{proc.stderr}")
    return path


# ---------------------------------------------------------------- build

def _manifest(paths: list[Path], out: Path) -> None:
    lines = []
    for p in sorted(paths):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h} *{p.name}")
    out.write_text("\n".join(lines) + "\n")


def _check_manifest(manifest: Path, base: Path) -> list[str]:
    errors = []
    if not manifest.exists():
        return [f"missing {manifest}"]
    for line in manifest.read_text().splitlines():
        h, name = line.split(" *")
        p = base / name
        if not p.exists():
            errors.append(f"missing {p}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != h:
            errors.append(f"hash mismatch {p}")
    return errors


def build(dataset: str = "pure") -> None:
    rd, cd = raw_dir(), cache_dir(dataset)
    cd.mkdir(parents=True, exist_ok=True)
    log1, log2 = LOG_FILES[dataset]
    read = {"schema_overrides": LOG_SCHEMA}
    logs = pl.concat([pl.read_csv(rd / log1, **read), pl.read_csv(rd / log2, **read)])
    sizes = {}
    for split, (lo, hi) in SPLITS.items():
        frame = logs.filter(pl.col("date").is_between(lo, hi))
        if split == "test":
            frame = frame.drop(FEEDBACK_COLS)
        frame.write_parquet(cd / f"{split}.parquet")
        sizes[split] = frame.height
    user_f, video_b, video_s = STATIC_FILES[dataset]
    for key, fname in (("user", user_f), ("video_basic", video_b), ("video_stat", video_s)):
        pl.read_csv(rd / fname, infer_schema_length=None).write_parquet(cd / f"{key}.parquet")
    (cd / "sizes.json").write_text(json.dumps(sizes))
    _manifest(
        [cd / f"{s}.parquet" for s in SPLITS] + [cd / f"{k}.parquet" for k in ("user", "video_basic", "video_stat")],
        cd / "MANIFEST.sha256",
    )
    print(f"built cache at {cd}: sizes {sizes}")


# ---------------------------------------------------------------- verify

def popularity_scores(train: pl.DataFrame, eval_frame: pl.DataFrame, prior: float = 20.0) -> np.ndarray:
    """The kit's popularity baseline (baseline.py run_pop): smoothed training-window
    long_view rate per video, global mean for unseen. The literal 'count, 0 for unseen'
    variant in IMPLEMENTATION.md scores 0.5435, outside its own pinned range; the kit
    formula (ground-truth precedence) scores 0.5807. See reports/decisions.md."""
    stats = train.group_by("video_id").agg(pl.len().alias("imp"), pl.col("long_view").sum().alias("pos"))
    gmean = train["long_view"].sum() / train.height
    joined = eval_frame.select("video_id").join(stats, on="video_id", how="left")
    imp = joined["imp"].fill_null(0).to_numpy().astype(np.float64)
    pos = joined["pos"].fill_null(0).to_numpy().astype(np.float64)
    out = np.full(len(imp), gmean)
    seen = imp > 0
    out[seen] = (pos[seen] + prior * gmean) / (imp[seen] + prior)
    return out


def _parity_key() -> str:
    a = (STARTER_KIT / "MANIFEST.sha256").read_bytes()
    b = (cache_dir("pure") / "MANIFEST.sha256").read_bytes()
    return hashlib.sha256(a + b + b"popularity=kit-smoothed-rate").hexdigest()


def _run_fm_baseline() -> dict:
    proc = subprocess.run(
        [sys.executable, "baseline.py", "--model", "fm", "--data_dir", str(raw_dir().resolve())],
        cwd=STARTER_KIT, capture_output=True, text=True, encoding="utf-8", env=_sk_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"baseline.py failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    m = re.search(r"valid\s+GAUC ([\d.]+) \| nDCG@5 ([\d.]+) \| primary ([\d.]+)", proc.stdout)
    if not m:
        raise RuntimeError(f"could not parse baseline output:\n{proc.stdout[-2000:]}")
    return {"gauc": float(m.group(1)), "ndcg5": float(m.group(2)), "primary": float(m.group(3))}


def _parity_rungs(errors: list[str]) -> dict:
    """FM subprocess + random + popularity on validation, cached on the manifests."""
    pj = cache_dir("pure") / "parity.json"
    key = _parity_key()
    if pj.exists():
        cached = json.loads(pj.read_text())
        if cached.get("key") == key:
            return cached
    val = load("val")
    n = val.height

    rng_scores = np.random.default_rng(0).random(n)
    rand = evaluate("val", rng_scores)

    popv = evaluate("val", popularity_scores(load("train"), val))

    fm = _run_fm_baseline()
    result = {"key": key, "fm": fm, "random": rand, "popularity": popv}
    pj.write_text(json.dumps(result, indent=1))
    return result


def verify(dataset: str = "pure") -> int:
    errors: list[str] = []
    errors += _check_manifest(STARTER_KIT / "MANIFEST.sha256", STARTER_KIT)
    cd = cache_dir(dataset)
    errors += _check_manifest(cd / "MANIFEST.sha256", cd)

    if not errors:
        expected = (SPLIT_SIZES[dataset] if _is_real_pure_root(dataset) and dataset in SPLIT_SIZES
                    else json.loads((cd / "sizes.json").read_text()))
        for split in SPLITS:
            h = pl.read_parquet(cd / f"{split}.parquet").height
            if h != expected[split]:
                errors.append(f"{split}: {h} rows, expected {expected[split]}")
        test = load("test", dataset=dataset)
        leaked = set(test.columns) & set(FEEDBACK_COLS)
        if leaked:
            errors.append(f"test frame carries feedback columns: {sorted(leaked)}")

    if not errors and _is_real_pure_root(dataset):
        rungs = _parity_rungs(errors)
        fm = rungs["fm"]
        for k in ("primary", "gauc", "ndcg5"):
            if abs(fm[k] - FM_PARITY[k]) > FM_PARITY["tol"]:
                errors.append(f"fm parity: {k} {fm[k]:.4f} vs {FM_PARITY[k]} (tol {FM_PARITY['tol']})")
        if not RANDOM_RANGE[0] <= rungs["random"]["primary"] <= RANDOM_RANGE[1]:
            errors.append(f"random rung primary {rungs['random']['primary']:.4f} outside {RANDOM_RANGE}")
        if not POPULARITY_RANGE[0] <= rungs["popularity"]["primary"] <= POPULARITY_RANGE[1]:
            errors.append(f"popularity rung primary {rungs['popularity']['primary']:.4f} outside {POPULARITY_RANGE}")
        if not errors:
            print(f"parity: fm {fm['primary']:.4f} | random {rungs['random']['primary']:.4f} "
                  f"| popularity {rungs['popularity']['primary']:.4f}")

    if errors:
        for e in errors:
            print(f"VERIFY FAIL: {e}")
        return 1
    print(f"verify OK ({dataset} @ {data_root()})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dataset", default="pure", choices=list(LOG_FILES))
    a = ap.parse_args()
    if a.build:
        build(a.dataset)
    if a.verify:
        sys.exit(verify(a.dataset))
    if not (a.build or a.verify):
        print(__doc__)
