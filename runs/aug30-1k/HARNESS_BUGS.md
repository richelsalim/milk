# Frozen-layer findings from run aug30-1k

## 1 (fatal): the starter-kit checker is pure-only — 1k submissions can never validate

- Symptom: iteration 1 attempt 1 — training and metrics succeeded; the KEEP path's
  `prepare.write_submission(..., dataset="1k")` then failed inside
  `starter_kit/submit.py --check`: 第 2 行对齐错误：提交 (0,2723239)，评测集第 0 行是
  (0,3978) — the submission's first data row differs from the checker's eval row 0.
- Root cause (proven by reading the frozen kit): `starter_kit/data.py::load()`
  hard-codes `log_standard_*_pure.csv` and `video_features_basic_pure.csv`.
  `submit.py --check --data_dir <raw>` therefore always loads the PURE splits —
  (0,3978) is pure test row 0 — no matter which dataset the submission came from.
  `prepare.write_submission` (frozen) invokes that checker for every dataset, and
  `harness iterate` (frozen) calls write_submission on every KEEP, so a 1k run can
  train and score but can never keep an iteration. Structural, not mechanical: no
  mutable-zone edit can route around two frozen components talking to each other.
- Consequence: run aug30-1k is BLOCKED (see BLOCKED.md). The pure run aug30 is
  unaffected (its checks all passed against the correct pure eval set).
- Owner fix suggestion: make prepare.write_submission skip / replace the kit checker
  for non-pure datasets (schema + row-count + finiteness + alignment vs
  prepare.load's own frame), since the organizer kit only defines a checker for Pure.

## 2 (latent, fixed as data): 1k split cache order was unverified

- `prepare.build()`'s 1k/27k branch streams `pl.concat(scan_csv) → filter →
  sink_parquet`; polars' streaming sink does not guarantee source row order, and
  nothing verified it. While chasing finding 1, the 1k cache was rebuilt eagerly in
  exact raw-CSV order with per-split (user_id, video_id) parity asserted against the
  raw files, and `sizes.json` (unchanged: 5,055,984 / 2,524,980 / 4,132,081) +
  `MANIFEST.sha256` regenerated in `build()`'s format; feature caches were dropped
  (their keys embed the manifest, so they self-invalidate). This is a data
  correction outside the frozen code, kept for hygiene; it did not resolve finding 1.
- Note on v1 numbers: 1k training/eval was internally consistent (features, labels
  and evaluate all read the same frame), so phase-6 zoo scores on 1k stand as scores.
