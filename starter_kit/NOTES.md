# starter_kit/NOTES.md

Read of every vendored file (2026-08-30). Vendored byte for byte from the extracted kit the
organizers shipped (`kuairand-starter-kit/kuairand-starter-kit/`, committed 5fcf12b); the zip
itself was already removed from the repo before this build started. `MANIFEST.sha256` covers the
7 organizer files; `.gitattributes` marks `starter_kit/** -text` so git never rewrites their bytes.

## data.load(data_dir) — exact return, per split

Returns `dict[split] -> list[tuple]`, one tuple per log row:

```
(date:int, user_id:str, video_id:str, author_id:str, tab:str, duration_ms:float, label:int)
 x[0]      x[1]         x[2]          x[3]           x[4]     x[5]              x[6]
```

- `author_id` is joined from `video_features_basic_pure.csv`; missing video_id -> `'UNK'`.
- `label = 1 if row['long_view'] != '0' else 0`.
- The SAME tuple shape for every split — the kit's own "test" rows still carry the label
  (hidden-test stripping is the organizer's job server-side; locally `prepare.py` does it).
- Splits: train `20220408–20220421`, valid `20220422–20220428`, test `20220429–20220508`
  (inclusive bounds). The kit's name for validation is `valid` (prepare.py accepts `val`).

## Row-ordering rule

Read `log_standard_4_08_to_4_21_pure.csv` fully, then `log_standard_4_22_to_5_08_pure.csv`,
in `csv.DictReader` (file) order; filter each split by `lo <= date <= hi` keeping that order.
Measured on the real files: file 1 is exactly the train window (1,141,112 rows), file 2 is
exactly valid+test (124,909 + 170,588 = 295,497 rows), so each split is a contiguous
original-file-order slice. `row_id` in a submission is the 0-based index into this order.

## The 5 categorical FM fields (data.FIELDS)

`user_id`, `video_id`, `author_id`, `tab`, `dur_bucket` where `dur_bucket` is
`str(int(np.searchsorted(edges, duration_ms)))` with `edges` = the 9 inner decile quantiles of
**training** `duration_ms` (`np.quantile(train_durations, linspace(0,1,11)[1:-1])`).
Vocabulary is built on train only; unseen values map to a per-field UNK slot appended at the end.
Encoded ids are offset so all fields share one embedding table (`field_dims` cumsum offsets).

## Does the convergence rule ship as code?

No. It ships as **parameters only**: `baseline_scores.json -> convergence_rule = {epsilon: 0.002, N: 3}`
plus prose in README.md ("3 consecutive validation iterations improving by <= 0.002 => converged").
There is no reference implementation. `harness/convergence.py` implements the rule and reads
eps/N defaults from `baseline_scores.json`.

## submit.py CLI

```
python submit.py <path> [--data_dir DIR] [--split {valid,test}] (--make | --check | --score)
```
- `path` is positional; `--data_dir` defaults to `./KuaiRand-Pure/data` (we pass `data/raw`).
- `--check`: header must be exactly `row_id,user_id,video_id,score`; row count must equal the
  split; `row_id` must be 0-based consecutive; `user_id`/`video_id` must match the split rows
  positionally; score must parse as float and not be NaN/Inf.
- `--score` additionally runs evaluate (valid only in practice; test labels are local-only here).
- `--make` trains the official FM and writes an example submission.
- It imports `data`/`evaluate`/`baseline` as top-level modules -> run it with cwd=starter_kit/
  (or starter_kit on sys.path). It is `python3` in the docs; on this Windows host `python3`
  does not exist, so prepare.py invokes it via `sys.executable`.

## evaluate.py conventions (ground truth, matches Appendix A)

- nDCG gain `2^rel - 1`; per-user list sorted by score descending (Python sort is stable:
  ties keep within-user arrival order); users with zero positives score nDCG 0 and are included
  in the mean; nDCG@5 with log2 discounts.
- GAUC: only users with `0 < positives < impressions`, weighted by positive count; AUC is
  Mann-Whitney U with midrank tie correction; degenerate fallbacks return 0.5.
- `primary = (GAUC + nDCG@5) / 2`.

## Published numbers (baseline_scores.json)

| rung | valid primary | test primary |
|---|---|---|
| random (seeds 0-4) | 0.4834 | 0.4753 |
| item popularity (smoothed rate, prior=20) | 0.5807 | 0.5715 |
| FM official (k=16, lr=0.001, batch 8192, <=40 epochs, patience 4) | **0.6016** (GAUC 0.6674, nDCG@5 0.5357) | 0.5946 (std 0.0008 over 5 seeds) |
| oracle ceiling | 0.8484 | 0.8645 |

## Post-impression columns (dropped from prepare.load("test"))

From the log header: `is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view,
play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter` — everything observed
after the impression. Kept for test: `user_id, video_id, date, hourmin, time_ms, duration_ms,
is_rand, tab` (all known at impression time). No additional post-impression fields exist in
the kit beyond Appendix A's list.

## Contradictions with Appendix A

None. Splits, sizes, label, metrics, conventions, submission schema, baseline numbers and the
convergence parameters all agree with Appendix A. The problem statement's "NDCG@10 / Recall@50 /
click" row is superseded by evaluate.py exactly as Appendix A says. One naming nit only:
the kit says `valid` where Appendix A says validation/`val`; prepare.py maps the alias.
