"""Feature blocks. Each block is a pure function (rows, hist, tables, info) -> pl.DataFrame
of new feature columns aligned to `rows` (carrying `_idx` for order safety).

Time-safety contract shared by every hist-derived block:
- split == "train": per-key aggregates use ONLY rows strictly earlier than the row's
  time_ms (exclusive cumulative sums on a time-sorted frame, ties excluded via a
  min-over-(keys, time) pass — sums are of non-negative values so the first row of a
  tie group carries the pre-tie cumulative).
- split == "val"/"test": per-key aggregates over the full training window (hist).
Scalars (duration bucket edges, the global label mean used as smoothing prior) are
train-window constants; per-key label aggregates are never contaminated by the row
itself or its future.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

T0_MS = 1_649_347_200_000  # 2022-04-08 00:00 UTC, only used to keep exp() arguments small
MS_PER_DAY = 86_400_000.0
PRIOR = 20.0
DECAY_PRIOR = 5.0
SEQ_LEN = 20


def _days(col: str = "time_ms") -> pl.Expr:
    return (pl.col(col) - T0_MS) / MS_PER_DAY


def _smoothed(pos: str, cnt: str, gmean: float, prior: float = PRIOR) -> pl.Expr:
    return (pl.col(pos) + prior * gmean) / (pl.col(cnt) + prior)


def strict_past_sums(frame: pl.DataFrame, keys: list[str], values: dict[str, pl.Expr]) -> pl.DataFrame:
    """Exclusive per-key cumulative sums, strict in time_ms (ties excluded).

    Returns frame[_idx] + one column per entry in `values`, in the original row order.
    Every value expression must be non-negative (the tie fix relies on monotone cumsums).
    """
    work = frame.select("_idx", *keys, "time_ms", **values)
    names = list(values)
    work = work.sort([*keys, "time_ms", "_idx"])
    work = work.with_columns(
        (pl.col(n).cum_sum().over(keys) - pl.col(n)).alias(n) for n in names
    )
    work = work.with_columns(
        pl.col(n).min().over([*keys, "time_ms"]).alias(n) for n in names
    )
    return work.sort("_idx").select("_idx", *names)


def full_window_sums(hist: pl.DataFrame, keys: list[str], values: dict[str, pl.Expr]) -> pl.DataFrame:
    return hist.select(*keys, **values).group_by(keys).agg(pl.all().sum())


def _key_sums(rows: pl.DataFrame, hist: pl.DataFrame, split: str,
              keys: list[str], values: dict[str, pl.Expr]) -> pl.DataFrame:
    """Per-row past sums: strict-past (train) or full-window join (val/test).

    Missing keys in hist yield nulls — callers fill_null(0) so unseen == empty history.
    """
    if split == "train":
        return strict_past_sums(rows, keys, values)
    agg = full_window_sums(hist, keys, values)
    out = rows.select("_idx", *keys).join(agg, on=keys, how="left")
    return out.sort("_idx").select("_idx", *values)


# ---------------------------------------------------------------- plain blocks

def ctx(rows, hist, tables, info):
    """Row-own context: tab, hour, day-of-week, is_rand, days since split start.
    Leakage: none — every input is known at impression time from the row itself."""
    date = pl.col("date").cast(pl.String).str.to_date("%Y%m%d")
    return rows.select(
        "_idx",
        pl.col("tab").alias("ctx_tab"),
        (pl.col("hourmin") // 100).alias("ctx_hour"),
        date.dt.weekday().alias("ctx_dow"),
        pl.col("is_rand").alias("ctx_is_rand"),
        (date - pl.lit(info["split_start"]).cast(pl.String).str.to_date("%Y%m%d"))
        .dt.total_days().alias("ctx_days_since_start"),
    )


def item_static(rows, hist, tables, info):
    """Static video properties from video_features_basic_pure (+ row's duration_ms).
    Leakage: none — static table plus impression-time row columns."""
    return rows.select(
        "_idx",
        pl.col("duration_ms").alias("it_duration_ms"),
        pl.col("_dur_bucket").alias("it_dur_bucket"),
        pl.col("_video_type_code").alias("it_video_type"),
        pl.col("_upload_type_code").alias("it_upload_type"),
        pl.col("_music_type").fill_null(-1).alias("it_music_type"),
        pl.col("_tag_count").alias("it_tag_count"),
        (pl.col("date").cast(pl.String).str.to_date("%Y%m%d")
         - pl.col("_upload_dt").str.to_date("%Y-%m-%d"))
        .dt.total_days().fill_null(-1).alias("it_upload_age_days"),
    )


STAT_COLS = ["show_cnt", "play_cnt", "play_user_num", "complete_play_cnt", "valid_play_cnt",
             "long_time_play_cnt", "short_time_play_cnt", "like_cnt", "comment_cnt",
             "share_cnt", "play_duration", "play_progress"]


def item_stats(rows, hist, tables, info):
    """Aggregate video counters from video_features_statistic_pure (log1p of counts).
    Leakage: the table's aggregation window is NOT pinned by KuaiRand — it may include
    dates past HISTORY_END; switchable off via the full_nostats spec for that reason."""
    stat = tables["video_stat"].select(
        "video_id",
        *[pl.col(c).cast(pl.Float64).fill_null(0).log1p().alias(f"is_{c}") for c in STAT_COLS[:-1]],
        pl.col("play_progress").cast(pl.Float64).fill_null(0).alias("is_play_progress"),
    )
    return (rows.select("_idx", "video_id").join(stat, on="video_id", how="left")
            .sort("_idx").drop("video_id").fill_null(0))


def user_static(rows, hist, tables, info):
    """Static user profile from user_features_pure (categoricals as codes).
    Leakage: none pinned — static profile table shipped with KuaiRand."""
    u = tables["user"]
    uad = {v: i for i, v in enumerate(sorted(u["user_active_degree"].drop_nulls().unique()))}
    cols = [
        pl.col("user_active_degree").replace_strict(uad, default=-1).alias("us_active_degree"),
        pl.col("is_lowactive_period").fill_null(-1).alias("us_lowactive"),
        pl.col("is_live_streamer").fill_null(-1).alias("us_live_streamer"),
        pl.col("is_video_author").fill_null(-1).alias("us_video_author"),
        pl.col("follow_user_num").fill_null(-1).alias("us_follow_num"),
        pl.col("fans_user_num").fill_null(-1).alias("us_fans_num"),
        pl.col("friend_user_num").fill_null(-1).alias("us_friend_num"),
        pl.col("register_days").fill_null(-1).alias("us_register_days"),
    ] + [pl.col(f"onehot_feat{i}").cast(pl.Float64).fill_null(-1).alias(f"us_oh{i}")
         for i in range(18)]
    return (rows.select("_idx", "user_id").join(u.select("user_id", *cols), on="user_id", how="left")
            .sort("_idx").drop("user_id").fill_null(-1))


# ---------------------------------------------------------------- history blocks

def hist_user(rows, hist, tables, info):
    """Training-window per-user aggregates (impressions, long_view/click rate, watch
    ratio, duration, distinct authors). Leakage: strictly-past rows for train, full
    training window for val/test; the row's own feedback never enters."""
    split, g = info["split"], info["gmean"]
    values = {
        "hu_n": pl.lit(1.0),
        "hu_pos": pl.col("long_view").cast(pl.Float64),
        "hu_click": pl.col("is_click").cast(pl.Float64),
        "hu_wr": pl.col("_wr"),
        "hu_dur": pl.col("duration_ms"),
        "hu_new_author": pl.col("_first_author").cast(pl.Float64),
    }
    s = _key_sums(rows, hist, split, ["user_id"], values).fill_null(0)
    return s.select(
        "_idx",
        pl.col("hu_n").alias("hu_impressions"),
        _smoothed("hu_pos", "hu_n", g).alias("hu_lv_rate"),
        _smoothed("hu_click", "hu_n", info["click_mean"]).alias("hu_click_rate"),
        (pl.col("hu_wr") / (pl.col("hu_n") + 1)).alias("hu_mean_wr"),
        (pl.col("hu_dur") / (pl.col("hu_n") + 1)).alias("hu_mean_dur"),
        pl.col("hu_new_author").alias("hu_distinct_authors"),
    )


def _entity_rates(rows, hist, tables, info, key: str, prefix: str) -> pl.DataFrame:
    g = info["gmean"]
    values = {f"{prefix}_n": pl.lit(1.0), f"{prefix}_pos": pl.col("long_view").cast(pl.Float64)}
    s = _key_sums(rows, hist, info["split"], [key], values).fill_null(0)
    return s.select(
        "_idx",
        pl.col(f"{prefix}_n").alias(f"{prefix}_impressions"),
        _smoothed(f"{prefix}_pos", f"{prefix}_n", g).alias(f"{prefix}_lv_rate"),
    )


def hist_item(rows, hist, tables, info):
    """Training-window aggregates per video, author, tag and music, plus an
    exponentially decayed per-video/per-author variant (half-life in days).
    Leakage: strictly-past for train, full window for val/test (see module docstring)."""
    out = _entity_rates(rows, hist, tables, info, "video_id", "hi_vid")
    for key, prefix in (("_author", "hi_auth"), ("_tag1", "hi_tag"), ("_music", "hi_music")):
        out = out.join(_entity_rates(rows, hist, tables, info, key, prefix), on="_idx")
    lam = math.log(2) / info["half_life_days"]
    g = info["gmean"]
    for key, prefix in (("video_id", "hi_vid_d"), ("_author", "hi_auth_d")):
        values = {
            f"{prefix}_w": (lam * pl.col("_t_days")).exp(),
            f"{prefix}_wpos": (lam * pl.col("_t_days")).exp() * pl.col("long_view"),
        }
        s = _key_sums(rows, hist, info["split"], [key], values).fill_null(0)
        s = s.join(rows.select("_idx", "_t_days"), on="_idx")
        s = s.with_columns(
            (pl.col(f"{prefix}_w") * (-lam * pl.col("_t_days")).exp()).alias("dw"),
            (pl.col(f"{prefix}_wpos") * (-lam * pl.col("_t_days")).exp()).alias("dwp"),
        )
        out = out.join(
            s.select(
                "_idx",
                pl.col("dw").alias(f"{prefix}_imp"),
                ((pl.col("dwp") + DECAY_PRIOR * g) / (pl.col("dw") + DECAY_PRIOR)).alias(f"{prefix}_lv_rate"),
            ),
            on="_idx",
        )
    return out


def cross(rows, hist, tables, info):
    """User x item interaction statistics that survive within-user ranking.
    Leakage: strictly-past for train, full training window for val/test."""
    split, g = info["split"], info["gmean"]
    out = None
    for keys, prefix in ((["user_id", "_author"], "x_ua"), (["user_id", "_tag1"], "x_ut"),
                         (["user_id", "_dur_bucket"], "x_ud")):
        values = {f"{prefix}_n": pl.lit(1.0), f"{prefix}_pos": pl.col("long_view").cast(pl.Float64)}
        if prefix == "x_ud":
            values[f"{prefix}_wr"] = pl.col("_wr")
        s = _key_sums(rows, hist, split, keys, values).fill_null(0)
        sel = [
            pl.col(f"{prefix}_n").alias(f"{prefix}_impressions"),
            _smoothed(f"{prefix}_pos", f"{prefix}_n", g).alias(f"{prefix}_lv_rate"),
        ]
        if prefix == "x_ud":
            sel.append((pl.col(f"{prefix}_wr") / (pl.col(f"{prefix}_n") + 1)).alias(f"{prefix}_mean_wr"))
        s = s.select("_idx", *sel)
        out = s if out is None else out.join(s, on="_idx")
    values = {"x_vd_n": pl.lit(1.0), "x_vd_pos": pl.col("long_view").cast(pl.Float64)}
    s = _key_sums(rows, hist, split, ["video_id", "_uad"], values).fill_null(0)
    out = out.join(
        s.select("_idx", _smoothed("x_vd_pos", "x_vd_n", g).alias("x_vd_lv_rate")), on="_idx"
    )
    return out


def target_enc(rows, hist, tables, info):
    """Heavily smoothed target encoding of high-cardinality ids (prior=100).
    Leakage: time-safe by construction — strictly-past cumulative encoding for train
    (the per-row limit of time-ordered folds), full training window for val/test."""
    g = info["gmean"]
    out = None
    for key, name in (("video_id", "te_video"), ("_author", "te_author"),
                      ("_music", "te_music"), ("_tag1", "te_tag")):
        values = {"n": pl.lit(1.0), "pos": pl.col("long_view").cast(pl.Float64)}
        s = _key_sums(rows, hist, info["split"], [key], values).fill_null(0)
        s = s.select("_idx", _smoothed("pos", "n", g, prior=100.0).alias(name))
        out = s if out is None else out.join(s, on="_idx")
    return out


def ids(rows, hist, tables, info):
    """Train-vocabulary integer codes for embedding models (user, video, author, tag,
    music), UNK = last index per field. Leakage: none — identity columns only."""
    out = rows.select("_idx")
    for col, name in (("user_id", "id_user"), ("video_id", "id_video"),
                      ("_author", "id_author"), ("_tag1", "id_tag"), ("_music", "id_music")):
        vocab = info["vocabs"][col]
        out = out.with_columns(
            rows[col].replace_strict(vocab, default=len(vocab)).alias(name)
        )
    return out


def seq(rows, hist, tables, info):
    """Last SEQ_LEN impressions of the user before this one (video/author/tag ids,
    duration, watch ratio), padded with -1/0. Leakage: shifted strictly-past rows for
    train; the user's last SEQ_LEN training-window rows for val/test (never rows of
    the evaluation split — no cross-row reads inside a split)."""
    feats = {"video_id": ("sq_v", -1), "_author": ("sq_a", -1), "_tag1": ("sq_t", -1),
             "duration_ms": ("sq_d", 0.0), "_wr": ("sq_w", 0.0)}
    if info["split"] == "train":
        work = rows.select("_idx", "user_id", "time_ms", *feats).sort(["user_id", "time_ms", "_idx"])
        cols = []
        for col, (pfx, fill) in feats.items():
            cols += [pl.col(col).shift(k).over("user_id").fill_null(fill).alias(f"{pfx}{k}")
                     for k in range(1, SEQ_LEN + 1)]
        return work.with_columns(cols).sort("_idx").select("_idx", *[c.meta.output_name() for c in cols])
    tail = (hist.sort(["user_id", "time_ms"]).group_by("user_id", maintain_order=True)
            .agg([pl.col(c).tail(SEQ_LEN).alias(c) for c in feats]))
    cols = []
    for col, (pfx, fill) in feats.items():
        cols += [pl.col(col).list.get(-k, null_on_oob=True).fill_null(fill).alias(f"{pfx}{k}")
                 for k in range(1, SEQ_LEN + 1)]
    tail = tail.select("user_id", *cols)
    return (rows.select("_idx", "user_id").join(tail, on="user_id", how="left")
            .sort("_idx").drop("user_id").fill_null(0))


def seq_pos(rows, hist, tables, info):
    """Last SEQ_LEN POSITIVE (long_view == 1) impressions of the user before this row
    (video/author/tag ids, duration, watch ratio), padded with -1/0. Leakage contract
    identical to `seq`: strictly-past rows for train (same (user_id, time_ms, _idx)
    tie order), the user's last SEQ_LEN positive training-window rows for val/test —
    never rows of the evaluation split itself."""
    feats = {"video_id": ("sp_v", -1), "_author": ("sp_a", -1), "_tag1": ("sp_t", -1),
             "duration_ms": ("sp_d", 0.0), "_wr": ("sp_w", 0.0)}
    names = [f"{pfx}{k}" for _c, (pfx, _f) in feats.items() for k in range(1, SEQ_LEN + 1)]
    if info["split"] == "train":
        work = rows.select("_idx", "user_id", "time_ms", "long_view", *feats).sort(
            ["user_id", "time_ms", "_idx"])
        flag = (pl.col("long_view") == 1).cast(pl.Int64)
        work = work.with_columns((flag.cum_sum().over("user_id") - flag).alias("_ppos"))
        lists = (work.filter(pl.col("long_view") == 1)
                 .group_by("user_id", maintain_order=True)
                 .agg([pl.col(c).alias(f"_L{c}") for c in feats]))
        work = work.join(lists, on="user_id", how="left")
        cols = [pl.when(pl.col("_ppos") >= k)
                .then(pl.col(f"_L{c}").list.get(pl.col("_ppos") - k, null_on_oob=True))
                .otherwise(pl.lit(fill)).alias(f"{pfx}{k}")
                for c, (pfx, fill) in feats.items() for k in range(1, SEQ_LEN + 1)]
        return work.with_columns(cols).sort("_idx").select("_idx", *names)
    tail = (hist.filter(pl.col("long_view") == 1).sort(["user_id", "time_ms"])
            .group_by("user_id", maintain_order=True)
            .agg([pl.col(c).tail(SEQ_LEN).alias(c) for c in feats]))
    cols = []
    for c, (pfx, fill) in feats.items():
        cols += [pl.col(c).list.get(-k, null_on_oob=True).fill_null(fill).alias(f"{pfx}{k}")
                 for k in range(1, SEQ_LEN + 1)]
    tail = tail.select("user_id", *cols)
    out = rows.select("_idx", "user_id").join(tail, on="user_id", how="left")
    fills = {f"{pfx}{k}": fill for _c, (pfx, fill) in feats.items()
             for k in range(1, SEQ_LEN + 1)}
    return (out.sort("_idx").drop("user_id")
            .with_columns([pl.col(n).fill_null(v) for n, v in fills.items()]))


BLOCKS = {
    "ctx": ctx, "item_static": item_static, "item_stats": item_stats,
    "user_static": user_static, "hist_user": hist_user, "hist_item": hist_item,
    "cross": cross, "target_enc": target_enc, "ids": ids, "seq": seq,
    "seq_pos": seq_pos,
}


def enrich(frame: pl.DataFrame, tables: dict, edges: np.ndarray) -> pl.DataFrame:
    """Attach helper columns every block shares: _idx, _author, _tag1, _music, _uad,
    _dur_bucket, _t_days, _wr (train only), _first_author, static codes."""
    vb = tables["video_basic"]
    vtypes = {v: i for i, v in enumerate(sorted(vb["video_type"].drop_nulls().unique()))}
    utypes = {v: i for i, v in enumerate(sorted(vb["upload_type"].drop_nulls().unique()))}
    vjoin = vb.select(
        "video_id",
        pl.col("author_id").alias("_author"),
        pl.col("tag").str.split(",").list.get(0, null_on_oob=True).cast(pl.Int64, strict=False)
        .fill_null(-1).alias("_tag1"),
        pl.col("tag").str.split(",").list.len().fill_null(0).alias("_tag_count"),
        pl.col("music_id").fill_null(-1).alias("_music"),
        pl.col("music_type").alias("_music_type"),
        pl.col("upload_dt").alias("_upload_dt"),
        pl.col("video_type").replace_strict(vtypes, default=-1).alias("_video_type_code"),
        pl.col("upload_type").replace_strict(utypes, default=-1).alias("_upload_type_code"),
    )
    u = tables["user"]
    uad = {v: i for i, v in enumerate(sorted(u["user_active_degree"].drop_nulls().unique()))}
    ujoin = u.select(
        "user_id", pl.col("user_active_degree").replace_strict(uad, default=-1).alias("_uad")
    )
    out = (frame.with_row_index("_idx")
           .join(vjoin, on="video_id", how="left")
           .join(ujoin, on="user_id", how="left")
           .sort("_idx")
           .with_columns(
               pl.col("_author").fill_null(-1), pl.col("_tag1").fill_null(-1),
               pl.col("_music").fill_null(-1), pl.col("_uad").fill_null(-1),
               pl.Series("_dur_bucket", np.searchsorted(edges, frame["duration_ms"].to_numpy())),
               ((pl.col("time_ms") - T0_MS) / MS_PER_DAY).alias("_t_days"),
           ))
    if "long_view" in frame.columns:
        out = out.with_columns(
            (pl.col("play_time_ms") / pl.col("duration_ms").clip(1)).clip(0, 5).alias("_wr"),
        )
        out = out.sort(["user_id", "_author", "time_ms", "_idx"]).with_columns(
            pl.col("_author").is_first_distinct().over("user_id").cast(pl.Int64).alias("_first_author")
        ).sort("_idx")
    return out
