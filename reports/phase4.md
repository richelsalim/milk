# Phase 4 report: the autoresearch loop

Harness CLI: start / iterate / abandon / revert / status / intervene / finish (+ report
stub for phase 6). Shape preserved from docs/autoresearch: one mutable surface, one
metric, one branch per run, keep-or-revert, results.tsv.

Design notes (full detail in code docstrings):
- Metrics are always recomputed by the harness from val_scores.npy through
  prepare.evaluate; train.py's own printout is ignored (tested with a stub that lies).
- One commit per iteration; commit hashes are derived from the git log by message
  prefix (`iter <n>:`) because a committed file cannot carry its own commit hash —
  results.tsv commit cells are backfilled at the next ledger write and sealed by finish.
- Failure attempts (error/timeout/oom/nan/shape/missing) never consume an iteration
  number; the third failure abandons the iteration (counts toward the 50 cap, not the
  convergence window). A broad exception guard means no traceback escapes iterate.
- Convergence rule implemented from the starter kit's shipped parameters
  (baseline_scores.json: eps=0.002, N=3) — it ships no code; also the 50-iteration cap
  and the 6 h wall-clock ceiling, whichever fires first, plus a divergence warning after
  5 consecutive revert/abandoned iterations (never stops the run).
- Watchdog: hard kill at 2 x budget + 120 s (grace overridable via env for tests only),
  0.2 s RSS sampling with a configurable cap (default 16 GB), child processes included.

## Gate commands and real output

```
$ uv run pytest -q tests/test_harness.py
.....                                                                    [100%]
5 passed in 79.83s (0:01:19)
```

Smoke run (real data, default train.py = the official FM):

```
$ uv run python -m harness start --run-id smoke --max-iters 3
started autoresearch/smoke @ 097c744 (baseline primary 0.6016)

$ uv run python -m harness iterate --desc "baseline reproduction"
=== ITERATION 1 (attempt 1) ===
DECISION: KEEP
primary: 0.6016  gauc: 0.6674  ndcg5: 0.5358
delta_vs_baseline: +0.0000  delta_vs_best: +0.0000
best_so_far: iter 1 @ 05e0070 (0.6016)

$ uv run python -m harness finish
final.csv re-validated (submit.py --check passed)
bundled bundle.tar.gz (3.6 MB, untracked)

results.tsv: 1  05e0070  keep  0.6016  0.6674  0.5358  +0.0000  40.1  0  baseline reproduction
$ PYTHONUTF8=1 uv run python starter_kit/submit.py --check --split test --data_dir data/raw submissions/smoke/final.csv
[check passed: 170,588 rows, split=test]
$ git checkout main && git branch -D autoresearch/smoke   # tree clean afterwards
```

The smoke iteration scored primary 0.6016 (delta 0.0000) vs the fm rung 0.6016 (within 0.003),
submissions/smoke/final.csv passed submit.py --check, and the smoke branch was deleted.
