# Phase 6 report: deliverables pack and bonus datasets

## Gate commands and real output

```
$ uv run python -m harness report --run-id scripted
report written to D:\milk\reports\scripted (with trajectory.png)
  -> results_table.md, resources.md, interventions.md, iteration_log.md, trajectory.png
     (the scripted phase-5 dry run, ledger copied under runs/scripted)

$ uv run python prepare.py --dataset 1k --build
built cache at D:/milk/data/cache/1k: sizes {'train': 5055984, 'val': 2524980, 'test': 4132081}

$ uv run python prepare.py --dataset 1k --verify
verify OK (1k @ D:\milk\data)
EXIT=0

$ uv run python -m recsys.zoo bench --dataset 1k --models lgbm_lambdarank --budget 900
lgbm_lambdarank: primary 0.6573 gauc 0.6888 ndcg5 0.6257 (356s, 9739 MB)
K1_BENCH_EXIT=0

(two fixes surfaced by 1k and folded into the mutable zone: LightGBM's 10k-row query cap
-> oversized user groups are chunked; model early-stop evaluation now carries the dataset
through feature meta instead of assuming pure)
```

## Bonus datasets

- **1k** (11.7M rows): built via order-preserving polars streaming sinks
  (train 5,055,984 / val 2,524,980 / test 4,132,081); verify checks the cache manifest,
  sizes.json, and the no-feedback test frame (the pinned parity rungs are pure-only).
- **27k** (322M rows): loader implemented (multi-part scan + deterministic user sampling,
  `prepare.SAMPLE_27K_MOD = 24`, ~4% of users; policy documented in prepare.py and
  recorded via run.json's dataset field). Not run end to end — the raw 27k files are
  not downloaded; 1k is green so the precondition is met if ever needed.

## Deliverables

- `harness report` renders the judge pack for any run id (results table with absolute
  deltas vs the published baseline, resources, interventions, per-iteration log with
  hypothesis/diff/metrics/events, score trajectory).
- README.md: overview, setup, reproduce (start -> iterate -> finish -> report, plus
  re-scoring final.csv), limitations with TODO markers, solo team contributions.
- docs/devpost.md skeleton: problem fit, tools, libraries, datasets and assets.
