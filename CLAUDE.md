# CLAUDE.md

I am competing in the "Autonomous ML Research Agent for Recommender Systems" track (ByteDance, KuaiRand). This repo is an autoresearch-style harness (karpathy/autoresearch) for the KuaiRand-Pure ranking benchmark. Read this file fully before touching anything.

## The task in one line
Rank each user's logged impressions in the evaluation split by P(long_view). Score = mean(GAUC, nDCG@5), computed only by the organizer's evaluate.py. Beat the official Factorization Machine baseline on the hidden test set, autonomously, with auditable per-iteration logs.

## Pinned numbers
- Splits, date-based, from log_standard_4_08_to_4_21_pure.csv and log_standard_4_22_to_5_08_pure.csv: train 20220408 to 20220421 (1,141,112 rows), validation 20220422 to 20220428 (124,909 rows), test 20220429 to 20220508 (170,588 rows).
- Official baseline: Factorization Machine, k=16, lr=0.001, 5 categorical fields, numpy only, about 40 s on one CPU core. Validation GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016. Hidden test GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946 (5-seed std 0.0008).
- Reference rungs on the hidden test: random 0.4753, item popularity 0.5715, perfect ranking 0.8645 (GAUC 1.0000, nDCG@5 0.7289, because 27.1% of test users have no positive and 9.2% are all-positive). Judge progress against 0.8645, not 1.0.
- Convergence rule: a run is converged when the validation primary has not improved by more than eps = 0.002 over the last N = 3 consecutive scored iterations. Hard caps: 50 iterations per run and a 6 h wall-clock ceiling. Whichever fires first stops the run.
- Submission: CSV with header row_id,user_id,video_id,score, one line per evaluation-split row. row_id is the 0-based index into the split in data.load() order. (user_id, video_id) is not unique (3.06% of test rows are repeated pairs, up to 12 times), so row_id is the only key. NaN and Inf are rejected. `python3 submit.py --check` is the validator; `--make` produces an example.
- evaluate.py conventions: nDCG gain is 2^rel - 1; users with zero positives score nDCG 0 and are included in the mean; GAUC uses only users with 0 < positives < impressions and weights each user by positive count.
- Each user has about 5 logged impressions in the evaluation split. Recall@50 is meaningless here and is not scored.

## Three zones
FROZEN for the research agent (built under my instructions; `python prepare.py --verify` checks their hashes):
- starter_kit/**: organizer files, vendored byte for byte.
- prepare.py: loads splits through starter_kit data.load(), builds caches, exposes evaluate and submission helpers, strips test labels.
- harness/**: the loop runner.
- tests/**: parity, guards, features, models, harness and fault-injection tests.
MUTABLE (the research agent's whole surface):
- train.py: composes features + model + loss + training. Must satisfy the contract below.
- recsys/**: the environment library (features, models, losses, blending, eda).
HUMAN-OWNED (edited by me between runs, never by the research agent):
- program.md, CLAUDE.md, IMPLEMENTATION.md, README.md, docs/**, reports/**.

## Data policy (non-negotiable)
1. Test labels never enter memory. prepare.load("test") returns features only; every feedback column is dropped before return. A test asserts this.
2. No label from any row dated after HISTORY_END may be used to fit anything. Default HISTORY_END = 20220421 (train end). This excludes log_random_*.csv entirely by default, because that file lives in the validation and test date range.
3. No cross-row aggregation inside an evaluation split (no test-time popularity, no counting duplicates across rows). Features for a row may use: the training-window logs, the static user and video feature files shipped with KuaiRand, and the row's own columns.
4. Only KuaiRand files feed KuaiRand models. No external data, no pretrained weights that saw these benchmarks' test labels.
5. Never reimplement the metric. Call starter_kit.evaluate through prepare.evaluate. The harness recomputes metrics from the saved score arrays and ignores anything train.py prints.

## train.py contract
`python train.py --out <dir> [--seed S] [--time-budget SEC] [--model NAME] [--features NAME]`
Writes to <dir>: val_scores.npy (float32, length 124,909, data.load() order), test_scores.npy (float32, length 170,588, same rule), config.json (everything needed to re-run: model, feature spec, hyperparameters, number of rounds or epochs actually used, seed, subsampling), and exits 0. Respects --time-budget as training wall-clock (default 300 s); the harness hard-kills at 2x budget + 120 s. Deterministic for a fixed seed on CPU.

## What an iteration is
An iteration is one scored experiment: one hypothesis, one change, one train.py run, one validation score. A failed attempt (crash, timeout, bad output) does not consume an iteration number; the same iteration may be retried up to 3 attempts, after which it is abandoned. Abandoned iterations count toward the 50 cap but not toward the convergence window, because they produced no score. This interpretation is written into reports/ so the judges can see it.

## Iteration ledger
- results.tsv: tab-separated, tracked by git, one row per iteration: iter, commit, status (keep | revert | abandoned), primary, gauc, ndcg5, delta_vs_baseline, train_sec, seed, description. Never edited by hand.
- runs/<run_id>/iterations/<iter>/: hypothesis.md (written by the agent before running), diff.patch, metrics.json, events.jsonl, config.json, stdout.log. The harness writes everything except hypothesis.md.
- runs/<run_id>/notes.md: the agent's reflection after each iteration. runs/<run_id>/interventions.jsonl: every human touch, logged by me.

## Coding conventions
Python 3.11, uv, ruff. CPU-first; torch models auto-detect CUDA or MPS but never require it. Redirect long outputs to files; never let training logs flood the context. One idea per commit. Commit message format during research runs: `iter <n>: <short description> (primary=<x>)`.

## Ground truth precedence
starter_kit/evaluate.py > starter_kit/data.py > this file > program.md > the problem statement. If two of these disagree, stop and report the discrepancy to me instead of guessing.
