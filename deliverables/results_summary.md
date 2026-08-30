# Results summary & resource usage (Deliverable 4)

## Required benchmark: KuaiRand-Pure — validation-best vs the official baseline

| metric | validation-best (run aug30, blend) | official baseline (valid) | **absolute delta** |
|---|---|---|---|
| GAUC | 0.6751 | 0.6674 | **+0.0077** |
| nDCG@5 | 0.5399 | 0.5357 | **+0.0042** |
| **primary = mean(GAUC, nDCG@5)** | **0.6075** | **0.6016** | **+0.0059** |

Judging formula (`score_dataset = mean over m of delta(m)` with m ∈ {GAUC, nDCG@5}):
**(+0.0077 + 0.0042) / 2 = +0.0059.** For scale: the baseline's 5-seed std is 0.0008
(so this is ~7σ) and the convergence threshold is ε = 0.002 (~3ε). Produced by the
first scored autonomous run (`runs/aug30/`, branch `autoresearch/aug30`): iteration 1
reproduced the baseline to the 4th decimal (0.6016), iteration 2 switched to the
diverse snapshot blend and banked +0.0059, iterations 3-5 tested mechanism-backed
follow-ups (multi-task pruning, embedding width, a video x tab cross) and were
reverted by the harness; the run then converged under the shipped ε/N rule.

Reference ladder on validation (all reproduced by this repo's zoo,
[reports/zoo_baselines.md](../reports/zoo_baselines.md)):

| rung | primary |
|---|---|
| random | 0.4827 |
| item popularity | 0.5807 |
| official FM baseline | 0.6016 (published 0.6016 — reproduced exactly) |
| **blend (deepfm + ple, snapshot bases, weight grid — run aug30 best)** | **0.6075** |

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

Scored autonomous run (`runs/aug30/resources.json`):

| | |
|---|---|
| iterations used | 5 of 50 (5 scored: 2 kept, 3 reverted, 0 abandoned) — converged by the ε=0.002 / N=3 rule |
| manual interventions | **0** (`runs/aug30/interventions.jsonl` is empty) |
| agent wall-clock | 0.409 h |
| GPU-hours | 0.0 — every model in this repo is CPU-only |
| tokens | ~16.5k for the run (in ~14k / out ~2.5k), measured from the session's metered token pool at kickoff vs finish; the pool does not separate input/output, so the split is an 85/15 estimate (method in [reports/v2.md](../reports/v2.md)) |

Demonstration rehearsal (fixture-scale, `runs/scripted/resources.json`): 8 iterations
(7 scored, 1 abandoned), 0 interventions, 0.021 h — the adversarial dry run that
validated the loop before any scored run.

Environment cost profile: official baseline reproduces in ~40 s on CPU; every zoo
model fits within a 300 s training budget on a 12-core desktop with no GPU; the full
12-rung bench takes ~35 min.
