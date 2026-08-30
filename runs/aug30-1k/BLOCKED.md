# Run aug30-1k: BLOCKED (frozen layer)

The KuaiRand-1k scored run cannot proceed: the frozen submission checker only
understands the Pure dataset, and the frozen harness validates a submission on every
kept iteration, so no 1k iteration can ever be kept. Full diagnosis with evidence and
an owner-fix suggestion in [HARNESS_BUGS.md](HARNESS_BUGS.md).

State at stop:
- Iteration 1 (baseline reproduction, fm/fm5 on 1k, 900 s budget): training completed
  (rounds=2, 959.7 s) and produced finite, shape-correct score arrays; the failure is
  exclusively the pure-only checker on the KEEP path. Attempt count left at 1 failed
  attempt; no iteration was scored, results.tsv has no rows for this run.
- The 1k split cache was rebuilt in verified raw-CSV order and its manifest
  regenerated (HARNESS_BUGS.md finding 2) — the environment is healthier than before
  the attempt, and phase-6 1k zoo scores stand.
- Per program.md, a broken frozen layer is the one legitimate early stop:
  this file is that stop for run aug30-1k. No human message arrived during the run
  (interventions: 0).
