# Deliverables — Autonomous ML Research Agent for Recommender Systems (KuaiRand-Pure)

> ## +0.0055 validation primary over the official baseline — before the first scored autonomous run
>
> | | GAUC | nDCG@5 | **primary** |
> |---|---|---|---|
> | official FM baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
> | **this repo, validation-best (blend)** | **0.6743** | **0.5398** | **0.6071** |
> | **absolute delta** (the judging formula: mean of per-metric deltas) | **+0.0069** | **+0.0041** | **+0.0055** |
>
> That is ~7× the baseline's 5-seed noise (σ = 0.0008) and 2.75× the convergence
> threshold (ε = 0.002), reached inside one 300 s CPU-only training budget. The ladder:
> random 0.4827 → item popularity 0.5807 → official FM 0.6016 → **ours 0.6071**.
> Bonus benchmark readiness: the same pipeline runs KuaiRand-1k end to end
> (lgbm_lambdarank validation primary 0.6573; no official 1k baseline is published).

This folder organizes the repository's existing artifacts against the four required
deliverables and the judging criteria. Nothing here is new evidence — every number
links back to a tracked report or ledger produced by the build's gated phases.

## The four required deliverables

| # | Requirement | Where |
|---|---|---|
| 1 | Written project description (Devpost) | [devpost_description.md](devpost_description.md) — paste-ready |
| 2 | Public repository: structured code + README (overview, setup, reproduce, limitations, contributions) | Repo root [README.md](../README.md); constitution in [CLAUDE.md](../CLAUDE.md); build audit trail in [IMPLEMENTATION.md](../IMPLEMENTATION.md) (fully ticked master checklist) + `reports/phase*.md` |
| 3 | Run & iteration logs (hypothesis, diff, metrics, error/recovery per iteration) + manual-intervention summary | [iteration_logs.md](iteration_logs.md) — ledger anatomy + the demonstration run (`runs/scripted/`, rendered at `reports/scripted/`); **0 manual interventions** |
| 4 | Final submission (starter-kit schema) + results table + resource usage | [results_summary.md](results_summary.md) + [submission/](submission/) (validated `final.csv` from the validation-best configuration) |

## How this scores against the judging criteria

- **Technical Execution (35%)** — Primary metric: validation-best +0.0055 over baseline
  (table above); the hidden test is scored once on `final.csv`. Robustness: a watchdog
  with hard timeout and RSS caps, typed failure events (error/timeout/oom/nan/shape/missing),
  3-attempt retry then abandon, divergence warnings — proven by a 9-step scripted
  adversarial rehearsal (34/34 checks: crash, infinite loop, NaN, equal-score,
  convergence-stop) and a fault-injection suite (OOM, disk-full, corrupted checkpoint,
  dirty git, self-deleting outputs) in `tests/`. Full go/no-go outputs:
  [reports/final.md](../reports/final.md).
- **Innovation & Problem Insight (20%)** — what was targeted and why is written down
  before every change: [reports/eda.md](../reports/eda.md) (where the within-user signal
  lives), [reports/phase3.md](../reports/phase3.md) (the diagnosis that user-constant
  features cancel in within-user ranking, the full-data fix worth +0.004 on every torch
  model, and the budget-feasible diverse blend), and
  [reports/decisions.md](../reports/decisions.md) (every decision, with mechanisms).
- **Impact & Relevance / Autonomy (20%)** — the harness runs the whole loop itself:
  one hypothesis → one experiment → one commit, KEEP/REVERT decided on recomputed
  validation metrics, convergence/cap/ceiling stops, `interventions.jsonl` logs every
  human touch. Demonstration run: **0 interventions** (`runs/scripted/resources.json`).
- **Feasibility & Practicality (15%)** — CPU-only (no GPU anywhere), 300 s per
  experiment, official baseline reproduced in ~40 s; `harness finish` writes tokens,
  wall-clock, iteration count and GPU-hours into `resources.json` for every run.

## Status and what comes next

The environment is released at tag `v1.0`: every phase gate of IMPLEMENTATION.md is
green (baseline parity to ±0.0001, 40-test suite, scripted rehearsals, deterministic
zoo). The scored autonomous run — where the agent iterates from the reproduced baseline
under the convergence rule — is staged by [IMPLEMENTATION_V2.md](../IMPLEMENTATION_V2.md),
which also lists the concrete levers expected to push past 0.6071. After that run,
`results_summary.md` and `submission/` are refreshed from its converged ledger.
