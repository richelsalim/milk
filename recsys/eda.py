"""EDA for KuaiRand-Pure -> reports/eda.md. Reads only through prepare (data policy)."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare  # noqa: E402


def _rate_by(frame: pl.DataFrame, expr: pl.Expr, name: str) -> list[str]:
    g = (frame.group_by(expr.alias(name)).agg(pl.len().alias("n"), pl.col("long_view").mean().alias("rate"))
         .sort(name))
    lines = [f"| {name} | rows | long_view rate |", "|---|---|---|"]
    lines += [f"| {k} | {n:,} | {r:.4f} |" for k, n, r in g.iter_rows()]
    return lines


def main() -> None:
    train = prepare.load("train")
    val = prepare.load("val")
    out = ["# EDA: KuaiRand-Pure", ""]

    out += [f"Overall long_view rate: train {train['long_view'].mean():.4f}, "
            f"val {val['long_view'].mean():.4f}", ""]

    edges = np.quantile(train["duration_ms"].to_numpy(), np.linspace(0, 1, 11)[1:-1])
    out += ["## Label rate by duration decile (train; edges from train quantiles)", ""]
    out += _rate_by(train, pl.col("duration_ms").map_batches(
        lambda s: pl.Series(np.searchsorted(edges, s.to_numpy()))), "dur_decile")
    out += ["", "## Label rate by tab (train)", ""]
    out += _rate_by(train, pl.col("tab"), "tab")
    out += ["", "## Label rate by hour of day (train)", ""]
    out += _rate_by(train, pl.col("hourmin") // 100, "hour")
    out += ["", "## Label rate by is_rand (train)", ""]
    out += _rate_by(train, pl.col("is_rand"), "is_rand")

    per_user = val.group_by("user_id").agg(pl.len().alias("n"), pl.col("long_view").sum().alias("pos"))
    n = per_user["n"].to_numpy()
    out += ["", "## Validation split shape", ""]
    out += [f"- impressions per user: median {np.median(n):.0f}, p90 {np.quantile(n, 0.9):.0f} "
            f"(users: {per_user.height:,})"]
    zero = (per_user["pos"] == 0).mean()
    allpos = (per_user["pos"] == per_user["n"]).mean()
    out += [f"- users with zero positives: {zero:.1%} (nDCG pinned at 0, excluded from GAUC)",
            f"- users with all positives: {allpos:.1%} (nDCG pinned at 1, excluded from GAUC)",
            f"- discriminative users: {1 - zero - allpos:.1%}"]

    pairs = val.group_by("user_id", "video_id").len()
    dup = pairs.filter(pl.col("len") > 1)
    dup_rows = int(dup["len"].sum()) if dup.height else 0
    out += [f"- repeated (user, video) pairs: {dup.height:,} pairs covering {dup_rows:,} rows "
            f"({dup_rows / val.height:.2%} of val), max multiplicity "
            f"{int(dup['len'].max()) if dup.height else 1}"]

    tu, vu = set(train["user_id"].to_list()), set(val["user_id"].to_list())
    ti, vi = set(train["video_id"].to_list()), set(val["video_id"].to_list())
    out += ["", "## Train/validation overlap", ""]
    out += [f"- users: {len(tu & vu):,} of {len(vu):,} val users seen in train ({len(tu & vu) / len(vu):.1%})",
            f"- items: {len(ti & vi):,} of {len(vi):,} val videos seen in train ({len(ti & vi) / len(vi):.1%})"]

    out += ["", "## play_time_ms vs duration_ms by long_view (train)", ""]
    g = (train.with_columns((pl.col("play_time_ms") / pl.col("duration_ms").clip(1)).alias("watch_ratio"))
         .group_by("long_view")
         .agg(pl.len().alias("n"), pl.col("play_time_ms").mean().alias("mean_play"),
              pl.col("play_time_ms").median().alias("med_play"),
              pl.col("duration_ms").mean().alias("mean_dur"),
              pl.col("watch_ratio").mean().alias("mean_wr"),
              pl.col("watch_ratio").median().alias("med_wr"))
         .sort("long_view"))
    out += ["| long_view | rows | mean play_ms | median play_ms | mean dur_ms | mean watch_ratio | median watch_ratio |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {lv} | {nn:,} | {mp:,.0f} | {dp:,.0f} | {md:,.0f} | {mw:.3f} | {dw:.3f} |"
            for lv, nn, mp, dp, md, mw, dw in g.iter_rows()]
    out += ["",
            "long_view is a deterministic function of play_time and duration (KuaiRand defines it as "
            "play_time >= duration for short videos, >= 18s for longer ones), and duration_ms is known "
            "at impression time — watch-time/watch-ratio modelling is a direct auxiliary signal.",
            "",
            "Evaluation ranks within a user (~5 impressions each), so user-constant features only "
            "help through interactions or a tree model; item, context and user x item signal is what "
            "moves GAUC / nDCG@5."]

    path = Path(prepare.REPO) / "reports" / "eda.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
