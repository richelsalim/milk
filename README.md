# KuaiRand-Pure autoresearch harness

## Overview

This project implements an autoresearch-style
research harness for the ByteDance "Autonomous ML Research Agent for Recommender Systems"
track. The benchmark is KuaiRand-Pure ranking: score each user's logged impressions by
P(long_view), judged on mean(GAUC, nDCG@5), computed only through the organizer's own
`evaluate.py`. The objective is to beat the official FM baseline (validation primary
0.6016) using an agent that runs the full research loop autonomously, with a complete
record of what it tried and why.

## Result

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| This repo, validation-best (blend) | 0.6751 | 0.5399 | 0.6075 |
| Delta | +0.0077 | +0.0042 | +0.0059 |

This represents a +0.0059 improvement over baseline on validation, roughly 7x the
baseline's own seed-to-seed noise (σ = 0.0008), running entirely on CPU within a 300
second training budget per run. The result comes from the first scored autonomous run
(`aug30`): 5 iterations to convergence, with no human intervention.

For reference, the score ladder runs from random guessing (0.4827), through item
popularity (0.5807), to the official baseline (0.6016), to this repo's blend (0.6075).

The full submission pack, including the Devpost writeup, iteration logs, results table,
and final submission file, is available in [deliverables](deliverables/).

## Architecture

The repository is organized into three zones so the agent can experiment freely without
being able to alter its own scoring rules (`CLAUDE.md` defines this contract):

- **Frozen** — components the agent cannot modify: `starter_kit/` (the organizer's files,
  kept byte-for-byte), `prepare.py` (data splits, metrics, submission writing), `harness/`
  (git handling, watchdog, convergence check, run ledger), and the test suite.
- **Mutable** — the agent's working surface: `train.py` and `recsys/` (features, models,
  losses, blending, EDA). All experimentation happens here.
- **Human-owned** — `program.md` (the agent's runtime instructions), plus supporting docs
  and reports.

## Setup

```bash
# 1. toolchain (Python 3.11 via uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 2. data (organizer-referenced Zenodo record, no registration)
#    place the KuaiRand-Pure CSVs in data/raw/  — or:
curl -L -o data/KuaiRand-Pure.tar.gz https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf data/KuaiRand-Pure.tar.gz -C data && mv data/KuaiRand-Pure/data/* data/raw/

# 3. build + verify the frozen layer (parity vs the published baseline numbers)
uv run python prepare.py --build
uv run python prepare.py --verify
uv run python tests/fixtures/make_fixture.py
uv run pytest -q
```

## Reproduce a research run

```bash
uv run python -m harness start --run-id <tag>          # branch autoresearch/<tag>, verify, ledger
# the agent loop (program.md): write runs/<tag>/iterations/<i>/hypothesis.md, then
uv run python -m harness iterate --desc "<what changed>"
# ... repeat until the harness prints STOP (converged / 50-iteration cap / 6 h ceiling)
uv run python -m harness finish                        # re-check final.csv, resources, bundle
uv run python -m harness report --run-id <tag>         # reports/<tag>/: tables, log, trajectory
```

A scored run is initiated using only the message specified in IMPLEMENTATION.md Appendix
C. Any subsequent human message counts as a manual intervention (logged via `python -m
harness intervene`), since the purpose of the run is to measure how far the agent gets
on its own.

To re-score a finished submission locally (validation only; test labels never enter
memory):

```bash
PYTHONUTF8=1 uv run python starter_kit/submit.py --check --split test --data_dir data/raw submissions/<tag>/final.csv
PYTHONUTF8=1 uv run python starter_kit/submit.py --score --split valid --data_dir data/raw <a valid-split csv>
```

To view the model zoo's baseline numbers directly, run `uv run python -m recsys.zoo bench
--budget 300` (writes to reports/zoo_baselines.md; the current best rung is the blend at
0.6071 against FM's 0.6016). A dry-run rehearsal is also available via `uv run python -m
tests.scripted_agent.run --fixture small` (or `--fixture full`), which runs a scripted
9-step agent against the real harness in a throwaway clone, useful for validating the
setup before a scored run.

## Limitations and future work

The within-user signal appears to cap out around 0.606–0.607 for any single model trained
within the 300 second budget (see reports/phase3.md). The two most promising untested
directions are attention over a user's long-view history (DIN-style) and distilling
several models into one that still fits the time budget.

KuaiRand-27k has a working loader, sampling users deterministically via
`prepare.SAMPLE_27K_MOD`, but was not run end to end due to time constraints. KuaiRand-1k
is fully wired and verified.

GPU support is auto-detected in the code but untested, as the development machine was
CPU-only. Token accounting also depends on the agent self-reporting its usage (`finish
--tokens-in/--tokens-out`), so that figure should be treated as an estimate.

LightGBM produces a recurring warning about filling missing categoricals with -1 (treated
as NaN). This does not affect correctness but is worth a cleaner encoding pass if LightGBM
becomes the retained model family.

## Contributions

This was a three-person team project, built on top of the organizer's starter kit and the public KuaiRand dataset.

Rahul Mitra — harness and infrastructure (run loop, watchdog, convergence logic, git-based iteration tracking).  
Richel Felisha Salim — modeling and feature engineering (recsys/: features, model zoo, losses, blending).  
Cheah Wei Jun — evaluation, reporting, and deliverables (results tracking, iteration logs, Devpost writeup, documentation).
