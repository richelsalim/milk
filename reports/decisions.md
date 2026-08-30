# Decisions

Running log of decisions made without confirmation (IMPLEMENTATION.md section 0).

## Pre-made (IMPLEMENTATION.md section 2)
- CPU-first. Everything runs on one CPU core inside the time budget; GPU is auto-detected and optional.
- Python 3.11, uv, ruff, pytest. numpy, polars, pyarrow, lightgbm, torch (CPU wheels), scikit-learn for utilities only.
- One experiment per iteration, 300 s training wall-clock per experiment by default.
- The harness does all git operations during a research run. One commit per iteration.
- HISTORY_END defaults to 20220421. Validation labels are never used for fitting except in the optional, clearly labelled refit at finish.
- No transductive features: nothing is aggregated across rows of the validation or test split.
- train.py defaults to the FM baseline (model fm, feature spec fm5) so a research run's iteration 1 reproduces the official baseline.

## Made during the build

### 2026-08-30 phase 0
- **Starter kit source**: `kuairand-starter-kit.zip` is no longer in the repo (commit 5fcf12b removed it as "obsolete" and committed its extracted contents at `kuairand-starter-kit/kuairand-starter-kit/`). The extracted directory is byte-identical in role to the zip's contents, so Phase 1 vendors `starter_kit/` from it. Not treated as a block.
- **Raw data**: `data/raw/` was absent and no copy existed anywhere on disk. The vendored starter kit README names the canonical source (`https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`, 47 MB, no registration). Downloaded from there instead of blocking; integrity is checked against the pinned split sizes (1,141,112 / 124,909 / 170,588) in `prepare.py --verify`. Would have written BLOCKED.md only if the download or the row counts failed.
- **Windows host**: `python3` does not exist on PATH (Windows). All code invokes `sys.executable`; documented gate commands use `python` via `uv run`. Same commands, same semantics.
- **Extra deps beyond the section-2 list**: `matplotlib` (Phase 6 trajectory.png) and `psutil` (Phase 4 watchdog RSS sampling; Windows has no `resource` module). Both pinned in pyproject.toml.
- **Convergence rule as shipped**: the starter kit ships the rule as *parameters* (`baseline_scores.json: convergence_rule = {epsilon: 0.002, N: 3}`), not as code. harness/convergence.py implements the rule and reads eps/N defaults from that file.

### 2026-08-30 phase 1
- **Popularity parity rung formula**: IMPLEMENTATION.md pins "training-window long_view *count* per video, 0 for unseen" with expected primary in [0.55, 0.60] — but on the real validation split the count variant scores **0.5435**, outside the guide's own range. The published popularity rung (valid 0.5807) is the starter kit's `run_pop`: smoothed long_view *rate* `(pos + 20*gmean)/(imp + 20)`, global mean for unseen. Ground-truth precedence (starter-kit code > IMPLEMENTATION.md) resolves the inconsistency: `prepare.popularity_scores` implements the kit formula and the [0.55, 0.60] assertion is kept. Not a block — the Phase 1 blocking condition is FM parity, which passed (0.6015 / 0.6671 / 0.5358 vs published 0.6016 / 0.6674 / 0.5357).
- **Fixture**: 315 users sampled (seed 0, users present in all three splits, greedy fill to 20k rows) -> 15,753 / 1,877 / 2,528 rows. The fixture is a full parallel data root (raw CSVs + cache) under `data/cache/fixture_small/`, selected via `KUAIRAND_DATA_ROOT`, so `submit.py --check`, `prepare.*` and later the whole harness run on it unchanged.
- **`--verify` idempotence + speed**: the three parity rungs are cached in `data/cache/parity.json`, keyed on the starter-kit and cache manifests (plus formula tag). First `--verify` computes and writes it (~4 min, FM subprocess); later runs are read-only and instant. The second-run-makes-no-changes property holds from run 2 onward.
