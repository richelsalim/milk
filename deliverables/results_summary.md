# Results summary & resource usage (Deliverable 4)

## Required benchmark: KuaiRand-Pure — validation-best vs the official baseline

| metric | validation-best (blend) | official baseline (valid) | **absolute delta** |
|---|---|---|---|
| GAUC | 0.6743 | 0.6674 | **+0.0069** |
| nDCG@5 | 0.5398 | 0.5357 | **+0.0041** |
| **primary = mean(GAUC, nDCG@5)** | **0.6071** | **0.6016** | **+0.0055** |

Judging formula (`score_dataset = mean over m of delta(m)` with m ∈ {GAUC, nDCG@5}):
**(+0.0069 + 0.0041) / 2 = +0.0055.** For scale: the baseline's 5-seed std is 0.0008
(so this is ~7σ) and the convergence threshold is ε = 0.002 (2.75ε).

Reference ladder on validation (all reproduced by this repo's zoo,
[reports/zoo_baselines.md](../reports/zoo_baselines.md)):

| rung | primary |
|---|---|
| random | 0.4827 |
| item popularity | 0.5807 |
| official FM baseline | 0.6016 (published 0.6016 — reproduced exactly) |
| **blend (deepfm + ple weighted rank-average)** | **0.6071** |

Hidden-test reference (published, scored once on `final.csv` by the organizers):
baseline GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946; oracle ceiling 0.8645.
The submission in [submission/](submission/) is this configuration's hidden-test
output in the starter-kit schema, validated by `submit.py --check`.

## Bonus benchmark: KuaiRand-1k

Wired end to end (streaming cache build, verified splits: 5,055,984 / 2,524,980 /
4,132,081 rows). `lgbm_lambdarank` reaches **validation primary 0.6573 (GAUC 0.6888 /
nDCG@5 0.6257)** in 356 s. No official 1k baseline is published, so no delta is
claimed. KuaiRand-27k: loader with a documented deterministic user-sampling policy
(`prepare.SAMPLE_27K_MOD`), not run end to end.

## Resource usage

Reported per run by `harness finish` into `runs/<tag>/resources.json`
(tokens in+out, agent wall-clock, iterations of the 50 cap, GPU-hours, intervention
count, and the iteration-counting interpretation).

Demonstration run (scripted rehearsal, `runs/scripted/resources.json`):

| | |
|---|---|
| iterations used | 8 of 50 (7 scored, 1 abandoned) — converged by the ε=0.002 / N=3 rule |
| manual interventions | **0** |
| agent wall-clock | 0.021 h (fixture-scale) |
| GPU-hours | 0.0 — every model in this repo is CPU-only |
| tokens | reported at finish via `--tokens-in/--tokens-out` (agent-side accounting) |

Environment cost profile: official baseline reproduces in ~40 s on CPU; every zoo
model fits within a 300 s training budget on a 12-core desktop with no GPU; the full
12-rung bench takes ~35 min. The scored autonomous run's own resources.json replaces
the demonstration row above once it completes (staged in IMPLEMENTATION_V2.md).
