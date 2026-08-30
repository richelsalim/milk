# Phase 5 report: scripted dry run and fault injection

The scripted driver (tests/scripted_agent/run.py) plays the research agent against the
real harness in a temporary clone (`git clone <repo> <tmp>/scripted-<ts>`), on the 20k
fixture with --time-budget 5 and the QUALITY-knob train stub (q x time-safe popularity
+ (1-q) x seeded noise). The 9-step sequence from IMPLEMENTATION.md runs verbatim:
keep, keep, error-then-fixed revert, keep, triple-timeout abandon, nan-then-fixed
revert, equal-not-better revert, below-best revert + STOP converged, refused-after-STOP.

## Gate commands and real output

```
$ uv run pytest -q tests/test_fault_injection.py
.....                                                                    [100%]
5 passed in 25.90s
```

```
$ uv run python -m tests.scripted_agent.run --fixture small
34/34 checks passed (expected-vs-observed table, exit 0):
keeps at 01/02/04 only; error->fixed revert at 03; triple-timeout abandon at 05;
nan->fixed revert at 06; equal-not-better revert at 07; revert + STOP <converged> at 08;
iterate refused with STOP at 09; final.csv == iter_4 submission and passes --check;
events: 1 error / 3 timeout / 1 nan / 1 stop; mutable surface byte-equal to the iter-4
commit; every iteration has its artifact set; one commit per iteration; tree clean.
```

## Scripted run ledger (from the clone)

```
iter  commit   status     primary  gauc    ndcg5   delta     train_sec seed description
1     0589b23  keep       0.4608   0.4838  0.4377  -0.1408   0.6       0    q=0.20 baseline
2     4ca23c0  keep       0.4657   0.4896  0.4418  -0.1359   0.6       0    q=0.30
3     895014f  revert     0.4591   0.4811  0.4371  -0.1425   0.6       0    fixed, q=0.10
4     4cfd7a6  keep       0.5144   0.5618  0.4670  -0.0872   0.6       0    q=0.90
5     bc5d5f4  abandoned                                     0              abandoned after 3 attempts (timeout)
6     90425a2  revert     0.4792   0.5059  0.4526  -0.1224   0.6       0    fixed, q=0.50
7     9672881  revert     0.5144   0.5618  0.4670  -0.0872   0.6       0    q=0.90 again
8     7eae20e  revert     0.5101   0.5570  0.4633  -0.0915   0.6       0    q=0.85
```

Events across the run: 1 error (iter 3, attempt 1), 3 timeouts (iter 5), 1 nan (iter 6, attempt 1), 1 stop (converged at iter 8). The deltas are negative because the fixture stub's ceiling sits below the real-data baseline number recorded in run.json — the KEEP/REVERT logic only compares iterations against each other.

Git log (one commit per iteration):

```
be27aff run scripted: finish (converged)
7eae20e iter 8: q=0.85 (primary=0.5101)
9672881 iter 7: q=0.90 again (primary=0.5144)
90425a2 iter 6: fixed, q=0.50 (primary=0.4792)
bc5d5f4 iter 5: abandoned after 3 attempts (timeout)
4cfd7a6 iter 4: q=0.90 (primary=0.5144)
895014f iter 3: fixed, q=0.10 (primary=0.4591)
4ca23c0 iter 2: q=0.30 (primary=0.4657)
0589b23 iter 1: q=0.20 baseline (primary=0.4608)
f16abaf run scripted: start
98475d4 scripted: stub train.py
```

## Full-mode rehearsal (real data, real train.py, 300 s budget)

```
$ uv run python -m tests.scripted_agent.run --fixture full
11/11 checks passed, exit 0 — baseline keep (fm), unchanged rerun reverts on the exact
equal score (deterministic), syntax-error event then fixed rerun scored, finish ok,
final.csv passes --check, artifacts + one commit per iteration, tree clean.
```
