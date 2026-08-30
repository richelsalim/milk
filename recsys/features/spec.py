"""FeatureSpec registry, assembly and cache.

build(spec, split, history_end, dataset) -> (X float32, meta, group) with X row-aligned
to prepare.load(split) order and group = user_id per row. Cached under
<data root>/cache/features/<spec>-<split>-he<he>-<key>.parquet (+ .meta.json), keyed on
spec name+version, split, history_end, dataset and the input cache manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prepare  # noqa: E402
from recsys.features import blocks as B  # noqa: E402


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: int
    blocks: tuple[str, ...]


SPECS = {
    "fm5": FeatureSpec("fm5", 1, ("fm5_fields",)),
    "full": FeatureSpec("full", 1, ("ctx", "item_static", "item_stats", "user_static",
                                   "hist_user", "hist_item", "cross", "target_enc", "ids")),
    "full_nostats": FeatureSpec("full_nostats", 1, ("ctx", "item_static", "user_static",
                                                    "hist_user", "hist_item", "cross",
                                                    "target_enc", "ids")),
    # seq lives in its own spec: 100 padded id/value columns are for attention models,
    # not for trees — reports/decisions.md.
    "full_seq": FeatureSpec("full_seq", 1, ("ctx", "item_static", "item_stats", "user_static",
                                            "hist_user", "hist_item", "cross", "target_enc",
                                            "ids", "seq")),
}

CATEGORICAL = {"ctx_tab", "ctx_hour", "ctx_dow", "ctx_is_rand", "it_dur_bucket",
               "it_video_type", "it_upload_type", "it_music_type", "us_active_degree",
               "us_lowactive", "us_live_streamer", "us_video_author"}
CATEGORICAL_PREFIXES = ("id_", "fm5_", "us_oh")


def fm5_fields(rows, hist, tables, info):
    """The official baseline's 5 fields (user, video, author, tab, dur_bucket) as
    train-vocabulary codes, UNK = last index. Leakage: none — identity/context only."""
    out = rows.select("_idx")
    for col, name in (("user_id", "fm5_user"), ("video_id", "fm5_video"),
                      ("_author", "fm5_author"), ("tab", "fm5_tab"),
                      ("_dur_bucket", "fm5_dur_bucket")):
        vocab = info["vocabs"][col]
        out = out.with_columns(rows[col].replace_strict(vocab, default=len(vocab)).alias(name))
    return out


B.BLOCKS["fm5_fields"] = fm5_fields


def _vocab(series: pl.Series) -> dict:
    return {v: i for i, v in enumerate(sorted(series.unique().to_list()))}


def _assemble(spec: FeatureSpec, rows_raw: pl.DataFrame, hist_raw: pl.DataFrame,
              tables: dict, split: str):
    """Pure assembly from already-loaded frames (tests inject synthetic ones here)."""
    edges = np.quantile(hist_raw["duration_ms"].to_numpy(), np.linspace(0, 1, 11)[1:-1])
    hist = B.enrich(hist_raw, tables, edges)
    rows = hist if split == "train" else B.enrich(rows_raw, tables, edges)
    info = {
        "split": split,
        "split_start": prepare.SPLITS[split][0],
        "gmean": float(hist["long_view"].mean()),
        "click_mean": float(hist["is_click"].mean()),
        "half_life_days": 7.0,
        "vocabs": {c: _vocab(hist[c]) for c in
                   ("user_id", "video_id", "_author", "_tag1", "_music", "tab", "_dur_bucket")},
    }
    out = rows.select("_idx")
    for name in spec.blocks:
        out = out.join(B.BLOCKS[name](rows, hist, tables, info), on="_idx")
    out = out.sort("_idx").drop("_idx")

    columns = out.columns
    X = out.cast(pl.Float32).to_numpy()
    group = rows_raw["user_id"].to_numpy().astype(np.int64)
    cat_idx = [i for i, c in enumerate(columns)
               if c in CATEGORICAL or c.startswith(CATEGORICAL_PREFIXES)]
    field_dims = {}
    for c, col in (("id_user", "user_id"), ("id_video", "video_id"), ("id_author", "_author"),
                   ("id_tag", "_tag1"), ("id_music", "_music"),
                   ("fm5_user", "user_id"), ("fm5_video", "video_id"),
                   ("fm5_author", "_author"), ("fm5_tab", "tab"),
                   ("fm5_dur_bucket", "_dur_bucket")):
        if c in columns:
            field_dims[c] = len(info["vocabs"][col]) + 1
    meta = {"spec": spec.name, "version": spec.version, "split": split,
            "columns": columns, "categorical_idx": cat_idx, "field_dims": field_dims}
    return X, meta, group


def _cache_key(spec: FeatureSpec, split: str, history_end: int, dataset: str) -> str:
    manifest = (prepare.cache_dir(dataset) / "MANIFEST.sha256").read_bytes()
    raw = f"{spec.name}|{spec.version}|{split}|{history_end}|{dataset}|".encode() + manifest
    return hashlib.sha256(raw).hexdigest()[:10]


def build(spec_name: str, split: str, history_end: int = prepare.HISTORY_END,
          dataset: str = "pure", use_cache: bool = True):
    spec = SPECS[spec_name]
    split = {"valid": "val"}.get(split, split)
    if split == "val" and history_end > prepare.SPLITS["train"][1]:
        raise ValueError("building val features with history_end inside the val window leaks")

    fdir = prepare.cache_dir(dataset) / "features"
    key = _cache_key(spec, split, history_end, dataset)
    fpath = fdir / f"{spec.name}-{split}-he{history_end}-{key}.parquet"
    mpath = fpath.with_suffix(".meta.json")
    if use_cache and fpath.exists() and mpath.exists():
        frame = pl.read_parquet(fpath)
        meta = json.loads(mpath.read_text())
        group = frame["__group"].to_numpy().astype(np.int64)
        X = frame.drop("__group").to_numpy().astype(np.float32)
        return X, meta, group

    rows_raw = prepare.load(split, dataset=dataset,
                            history_end=history_end if split == "train" else None)
    hist_raw = rows_raw if split == "train" else prepare.load("train", dataset=dataset,
                                                              history_end=history_end)
    tables = prepare.tables(dataset)
    X, meta, group = _assemble(spec, rows_raw, hist_raw, tables, split)
    meta["history_end"] = history_end

    if use_cache:
        fdir.mkdir(parents=True, exist_ok=True)
        cache_frame = pl.DataFrame(X, schema=meta["columns"]).with_columns(
            pl.Series("__group", group)
        )
        cache_frame.write_parquet(fpath)
        mpath.write_text(json.dumps(meta))
    return X, meta, group


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m recsys.features")
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--spec", required=True, choices=list(SPECS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--history-end", type=int, default=prepare.HISTORY_END)
    ap.add_argument("--dataset", default="pure")
    a = ap.parse_args(argv)
    t0 = time.time()
    X, meta, group = build(a.spec, a.split, a.history_end, a.dataset)
    print(f"{a.spec}/{a.split}: X {X.shape} float32, {len(meta['categorical_idx'])} categorical, "
          f"group {len(group)} rows in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
