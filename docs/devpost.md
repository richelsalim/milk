# Devpost skeleton: Autonomous ML Research Agent for Recommender Systems (KuaiRand)

## How the solution addresses the problem statement

- One mutable surface (`train.py` + `recsys/`) that an LLM research agent edits; everything
  that defines the benchmark (data splits, metric, submission format) is frozen and
  hash-verified (`prepare.py --verify`), so the agent can experiment aggressively without
  being able to cheat.
- The harness (`python -m harness`) runs the autoresearch loop: one hypothesis -> one
  experiment -> one commit; metrics recomputed by the harness through the organizer's
  evaluate.py (never trusting the training script's printouts); KEEP/REVERT decided on
  validation primary; convergence rule eps=0.002 over N=3 scored iterations; 50-iteration
  and 6-hour caps; watchdog with hard timeout and RSS cap; every failure becomes a typed
  event in the ledger and is retried up to 3 attempts before the iteration is abandoned.
- Auditability: per-iteration hypothesis.md, diff.patch, metrics.json, events.jsonl,
  results.tsv (git-tracked), one commit per iteration, a validated submission for every
  kept iteration, and `harness report` renders judge-facing tables, an iteration log and
  the score trajectory.
- The environment library (`recsys/`) gives the agent composable parts: a time-safe
  feature layer (strict-past aggregates for training rows), a 12-rung model zoo
  (FM parity port, LightGBM pointwise/lambdarank, DeepFM, DCNv2, MMoE, PLE, censored
  watch-time CWM, DIN-lite sequence attention, blending) and within-user ranking losses.

## Development tools

Claude (agent), uv, ruff, pytest, git; Windows 11 / CPU-only (12 logical cores).

## APIs, libraries and frameworks

Python 3.11, numpy, polars, pyarrow, LightGBM, PyTorch (CPU wheels), scikit-learn
(utilities), matplotlib, psutil. No external services; everything runs locally.

## Datasets and assets

- KuaiRand-Pure (organizer starter kit + https://zenodo.org/records/10439422, MIT-licensed
  starter kit files vendored byte-for-byte under `starter_kit/`).
- KuaiRand-1K wired as a bonus dataset; KuaiRand-27K loader with a documented deterministic
  user-sampling policy.
- karpathy/autoresearch README + program.md vendored under `docs/autoresearch/` (MIT).
- No external data, no pretrained weights (data policy in CLAUDE.md).
