# Phase 2 report: time-safe feature layer

Specs registered: fm5 (5 cols), full (85 cols, 35 categorical), full_nostats (73),
full_seq (185; seq split into its own spec — padded id lists are for attention models,
not trees; reports/decisions.md). Strict-past semantics for train rows via exclusive
cumulative sums with tie exclusion; full-training-window joins for val/test.

## Gate commands and real output

```
$ uv run pytest -q tests/test_features.py
.......                                                                  [100%]
7 passed in 19.66s
  (includes: full/train single-core rebuild timing test — 15.9s < 60s)

$ uv run python -m recsys.features build --spec full --split val
full/val: X (124909, 85) float32, 35 categorical, group 124909 rows in 0.1s   (cached)
first build: full/val in 2.4s, full/train in 15.9s (POLARS_MAX_THREADS=1)
```

Tests cover: later-label shuffle invariance + earlier-label flip sensitivity on a
synthetic 3-user log; brute-force recomputation of hu_lv_rate / hi_vid_lv_rate /
x_ua_lv_rate for 5 real validation rows; no feedback column in any split's X; group ==
prepare.load user order; two uncached builds hash-identical; val history_end leak guard.
