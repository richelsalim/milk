# KuaiRand-Pure autoresearch harness

> **+0.0059 validation primary over the official baseline — 0.6075 vs 0.6016
> (+0.0077 GAUC, +0.0042 nDCG@5), ~7× the baseline's seed noise, CPU-only, inside one
> 300 s training budget — produced by the first scored autonomous run (`aug30`:
> 5 iterations to convergence, 0 manual interventions).** Ladder: random 0.4827 →
> popularity 0.5807 → official FM 0.6016 → **this repo's blend 0.6075**. Full
> submission pack: [deliverables/](deliverables/README.md).

An [autoresearch](https://github.com/karpathy/autoresearch)-style research harness for the
ByteDance "Autonomous ML Research Agent for Recommender Systems" track on the KuaiRand-Pure
ranking benchmark: rank each user's logged impressions by P(long_view); score =
mean(GAUC, nDCG@5) computed only by the organizer's evaluate.py; beat the official FM
baseline (validation primary 0.6016) autonomously, with auditable per-iteration logs.

Three zones (CLAUDE.md is the constitution):
- **frozen**: `starter_kit/` (organizer files, byte-for-byte), `prepare.py` (data + metric +
  submissions), `harness/` (the loop runner: git, watchdog, convergence, ledger), `tests/`.
- **mutable** (the research agent's whole surface): `train.py` + `recsys/` (features, models,
  losses, blending, eda).
- **human-owned**: `program.md` (the agent's runtime instructions), docs, reports.

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

A scored run is kicked off with the message in IMPLEMENTATION.md Appendix C and nothing else;
every later human message counts as a manual intervention (`python -m harness intervene`).

Re-score a finished submission locally (validation only; test labels never enter memory):

```bash
PYTHONUTF8=1 uv run python starter_kit/submit.py --check --split test --data_dir data/raw submissions/<tag>/final.csv
PYTHONUTF8=1 uv run python starter_kit/submit.py --score --split valid --data_dir data/raw <a valid-split csv>
```

Model zoo baselines: `uv run python -m recsys.zoo bench --budget 300` (reports/zoo_baselines.md;
current best rung: blend 0.6071 vs fm 0.6016 on validation). Dry-run rehearsal:
`uv run python -m tests.scripted_agent.run --fixture small` (a scripted 9-step agent against the
real harness in a temporary clone) or `--fixture full`.

## Limitations / with more time

- TODO: the within-user signal saturates ~0.606–0.607 for single 300 s models (see
  reports/phase3.md); the largest untapped directions are positive-history attention
  (DIN over long_view-only sequences) and cross-model distillation into one budget-fit model.
- TODO: the 27k dataset loader samples users (`prepare.SAMPLE_27K_MOD`) and has not been run
  end to end; 1k is wired and verified.
- TODO: GPU is auto-detected but untested in this build (CPU-only machine); token accounting
  relies on agent-side reporting (`finish --tokens-in/--tokens-out`).
- TODO: LightGBM categorical handling warns about -1 fills (treated as NaN); worth a cleaner
  encoding pass if lgbm becomes the kept model family.

## Team contributions

Solo build (harness, features, zoo, tests, reports) on top of the organizer starter kit and
the public KuaiRand dataset; see reports/decisions.md for every decision made along the way.
