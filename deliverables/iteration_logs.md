# Run & iteration logs (Deliverable 3)

Every iteration of a research run leaves the exact per-iteration record the problem
statement requires — hypothesis, code diff, metrics, error/recovery events — written
by the frozen harness (the agent cannot edit its own ledger), one git commit per
iteration.

## Anatomy of the ledger

| artifact | written by | contents |
|---|---|---|
| `runs/<tag>/iterations/<i>/hypothesis.md` | the agent, before running | what it intends to try and why (stage targeted, mechanism, expected delta, rollback) |
| `runs/<tag>/iterations/<i>/diff.patch` | harness | the code diff vs the last kept commit (`train.py` + `recsys/`) |
| `runs/<tag>/iterations/<i>/metrics.json` | harness | GAUC / nDCG@5 / primary **recomputed by the harness** through the organizer's evaluate.py (train.py's own printout is ignored), per-seed, timing, peak RSS, decision |
| `runs/<tag>/iterations/<i>/events.jsonl` | harness | typed error/recovery events: error, timeout, oom, nan, shape, missing, divergence, stop — with the action taken (retry / abandon / warn / stop) |
| `results.tsv` (git-tracked) | harness | one row per iteration: iter, commit, keep/revert/abandoned, metrics, delta vs baseline, seeds, description |
| `runs/<tag>/interventions.jsonl` | the human, via `harness intervene` | every manual touch — the count feeds Autonomy scoring |
| `runs/<tag>/resources.json` | `harness finish` | iterations, wall-clock, tokens, GPU-hours, **intervention count** |

Failure handling policy (also in the events): a failed attempt (crash, timeout, OOM,
NaN, wrong shape, missing output) does not consume an iteration number; after 3
attempts the iteration is abandoned — counting toward the 50-iteration cap but not
toward the ε/N convergence window, because it produced no score.

## The demonstration run (scripted rehearsal)

A full adversarial rehearsal of the loop — including deliberate crashes, an infinite
loop, NaN outputs, and an equal-score decision — ran against the real harness:

- Raw ledger: [`runs/scripted/`](../runs/scripted/) (8 iterations: 3 keeps, 4 reverts,
  1 abandoned; converged at iteration 8 by the ε = 0.002 / N = 3 rule).
- Rendered, judge-facing view: [`reports/scripted/iteration_log.md`](../reports/scripted/iteration_log.md)
  (one section per iteration: hypothesis → diff → metrics → events → how each was
  handled), plus [`results_table.md`](../reports/scripted/results_table.md),
  [`resources.md`](../reports/scripted/resources.md) and the score trajectory
  [`trajectory.png`](../reports/scripted/trajectory.png).
- Events across the run: 1 error (fixed on attempt 2), 3 timeouts (iteration abandoned),
  1 NaN (fixed on attempt 2), 1 convergence stop — every one recovered or routed
  around; the run neither crashed, stalled, nor diverged.

**Manual interventions: 0** (`runs/scripted/resources.json`).

## The scored autonomous run: `aug30`

The first scored run executed on 2026-08-30 (kickoff per IMPLEMENTATION.md Appendix C;
branch `autoresearch/aug30`, one commit per iteration, ledger also merged to main):

- Raw ledger: [`runs/aug30/`](../runs/aug30/) — per-iteration hypothesis.md,
  diff.patch, metrics.json, events.jsonl, config.json, plus notes.md (the agent's
  reflection after every iteration) and resources.json.
- Results table: [`results.tsv`](../results.tsv) — 5 scored iterations: baseline
  reproduced at 0.6016, diverse snapshot blend KEPT at **0.6075 (+0.0059)**, then
  three mechanism-backed follow-ups (ple aux-head pruning, deepfm dim 24, video x tab
  cross) each REVERTED by the harness at -0.0008/-0.0012/-0.0016 vs best.
- Rendered, judge-facing view: [`reports/aug30/`](../reports/aug30/) —
  iteration_log.md (hypothesis → diff → metrics → events per iteration),
  results_table.md, resources.md, trajectory.png.
- Stop: `STOP <converged>` under the shipped ε = 0.002 / N = 3 rule after the
  post-blend window read [-0.0008, -0.0012, -0.0016].

**Manual interventions: 0** (`runs/aug30/resources.json`; no interventions.jsonl
entries — no human message arrived during the run).
