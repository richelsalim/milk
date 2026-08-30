# recsys/features

**Read this first:** evaluation ranks within a user (~5 impressions each), so any feature
that is constant within a user contributes nothing on its own — user-side features only
help through interactions (FM/torch crosses) or a tree model's splits combining them with
item/context columns. Spend iteration budget on item, context and user x item signal.

`build(spec, split, history_end=20220421, dataset="pure") -> (X float32, meta, group)`
- X rows are in exact `prepare.load(split)` order; `group` = user_id per row.
- `meta`: `columns`, `categorical_idx`, `field_dims` (vocab sizes for id/fm5 columns).
- Cached at `<root>/cache/features/<spec>-<split>-he<he>-<key>.parquet` keyed on spec
  name+version, split, history_end, dataset and the data-cache manifest. Bump a spec's
  `version` whenever a block's semantics change.

## Time-safety contract (all history blocks)

- **train rows**: per-key aggregates use only rows **strictly earlier** than the row's
  `time_ms` (exclusive cumulative sums on a time-sorted frame; ties at the same
  timestamp are excluded).
- **val/test rows**: per-key aggregates over the **full training window** (`hist`).
- Scalars (duration-decile edges, global label mean used as smoothing prior) are
  train-window constants. The row's own feedback never enters its features.

## Blocks

| block | inputs | leakage argument (one line) |
|---|---|---|
| ctx | row columns | tab/hour/dow/is_rand/days-since-split-start are known at impression time |
| item_static | video_features_basic + row duration | static table + impression-time row columns only |
| item_stats | video_features_statistic | KuaiRand does **not** pin this table's aggregation window — may overlap eval dates; switchable off via `full_nostats` |
| user_static | user_features | static profile table shipped with the dataset |
| hist_user | training logs | strict-past (train) / train-window (val, test) per-user label aggregates |
| hist_item | training logs | same contract per video / author / tag / music, + exponentially decayed variant (half-life 7 d, exact per-row via exp-cumsums) |
| cross | training logs + static joins | same contract on (user x author), (user x tag), (user x dur_bucket), (video x user_active_degree) |
| target_enc | training logs | strictly-past cumulative smoothed encoding (per-row limit of time-ordered folds); prior=100 |
| ids | identity | train-vocab codes + UNK for embedding models; no labels |
| seq | training logs | last 20 strictly-earlier impressions (train) / last 20 train-window impressions (val, test) — never rows of the evaluation split |

## Specs

| spec | blocks | X cols |
|---|---|---|
| fm5 (v1) | the official baseline's 5 fields as train-vocab codes | 5 |
| full (v1) | ctx, item_static, item_stats, user_static, hist_user, hist_item, cross, target_enc, ids | 85 |
| full_nostats (v1) | full minus item_stats (unpinned aggregation window) | 73 |
| full_seq (v1) | full + seq (padded id lists — for attention models, useless for trees) | 185 |

CLI: `python -m recsys.features build --spec full --split val [--history-end N] [--dataset pure]`
