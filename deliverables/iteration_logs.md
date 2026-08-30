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

## Where the scored run's logs will land

The scored autonomous run (kickoff message in IMPLEMENTATION.md Appendix C; staged by
IMPLEMENTATION_V2.md) produces the same ledger under `runs/<tag>/` on branch
`autoresearch/<tag>`, rendered by `python -m harness report --run-id <tag>`. This file
then points at that run.
