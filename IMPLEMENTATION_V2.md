# IMPLEMENTATION_V2.md: pushing past 0.6071 and running the scored run

## 0. Read me first

v1 (IMPLEMENTATION.md, tag `v1.0`, all boxes ticked) built and validated the
environment: frozen data/eval layer with exact baseline parity (0.6016), time-safe
feature layer, 12-rung model zoo (validation-best: blend 0.6071 = fm + 0.0055), the
autoresearch harness (34/34 scripted checks, fault-injection green), and the
deliverables pack under `deliverables/`. **Do not modify IMPLEMENTATION.md — it is the
v1 record.** This guide is v2: raise the environment's validation ceiling with the
specific levers v1's investigation surfaced, then run the first scored autonomous run
and refresh the deliverables from its converged ledger.

How to work through this file
- Phases V2.0 to V2.3 (V2.4 optional), in order. Each phase lists deliverables, a gate,
  commit points, and appends a section to a single running report `reports/v2.md`
  (do not create per-phase report files).
- Progress lives in the Master checklist at the bottom. Tick a box only after its gate
  passed; commit the tick with the deliverable. On a new session: read this checklist,
  `git log --oneline -20`, `reports/v2.md`, then resume at the first unticked item.
- Do not ask for confirmation. Decide, append the decision to reports/decisions.md
  (new "v2" section), continue.
- **Two kinds of gates.** *Environment-correctness gates* (tests green, parity intact,
  budgets respected, artifacts produced) block on failure: write reports/BLOCKED.md,
  commit, push, stop. *Research-gain targets* (score X) never block: if missed, write
  the analysis into reports/v2.md, keep the best config found, and continue to the next
  phase. v1's zoo experience is the model: first guesses clustered at 0.60; the fixes
  came from diagnosis, not from insisting.
- Blocking conditions (the only stops): `python prepare.py --verify` failing; the
  40-test suite failing in a way a mechanical fix cannot recover; the scored run unable
  to start (git/branch corruption). Everything else: decide and continue.

Commit and push protocol
- Environment phases work on main. Messages `v2.<n>: <what>`. Push at each phase end,
  tag `v2-phase-<n>`. The scored run manages its own `autoresearch/<tag>` branch via
  the harness — never commit manually during it.
- Same tracked-file rules as v1 (nothing over 20 MB; data/, checkpoints/, out dirs,
  stdout logs, tarballs stay untracked).

Ground rules carried over: CLAUDE.md is the constitution (data policy, zones,
precedence); the frozen layer stays frozen (starter_kit/, prepare.py, harness/,
tests/ — v2's environment work happens in the MUTABLE zone: train.py and recsys/**,
plus new tests are allowed to be ADDED under tests/ for new mutable-zone behavior);
every model must fit + predict within the 300 s training budget on CPU; every number
quoted in a report comes from a run, never from memory.

## 1. Starting state (verify before phase V2.0)

`git describe --tags` reaches v1.0's lineage; `uv run python prepare.py --verify`
prints parity fm 0.6015 / random 0.4827 / popularity 0.5807 and exits 0;
`uv run pytest -q` = 40 passed; reports/zoo_baselines.md's last block ends with
blend 0.6071. If any of these fail → blocking condition.

## 2. What v1 learned (the evidence behind v2's levers)

From reports/phase3.md and reports/decisions.md:
- User-constant features cancel in within-user ranking; the signal is item, context
  and user × item interaction. Trees waste capacity on them; embeddings don't.
- Full-data training (not the 500k subsample) was worth +0.004 on every torch model.
  The best single model is ple 0.6062 in ~150 s of its 300 s budget — **half the
  budget is currently unused** because epochs 5-6 plateau.
- Snapshot ensembling (top-3 epoch checkpoints) reached 0.6065 but was implemented
  wastefully (re-predicted for selection); the per-epoch val predictions already exist
  from early stopping, so selection is nearly free if reused.
- Diverse rank-average blending is the strongest lever found: deepfm+ple = 0.6071;
  offline math showed fm/deepfm/ple/ple-variant combinations at 0.6065–0.6072, and
  adding a third base is budget-feasible (fm costs only ~33 s).
- Losses (listwise/bpr/mixed) never beat bce on this data; min_day filtering hurt;
  the user × author cross is dead (~5-impression lists, authors rarely repeat).
- The sequence feature mixes ~66 % negatives; classic DIN attends over POSITIVE
  history only — untested here and the statement's own "user history" lead.
- Recency: the decayed item-popularity rung beat the plain one (0.5816 vs 0.5807);
  recency-weighted TRAINING (sample weights by row date) is untested.

## 3. Phases

### Phase V2.0: free budget — snapshots, throughput, recency weights

Deliverables (mutable zone: recsys/models/, train.py; plus added tests)
1. Budget-free snapshot selection: `_TorchRec` already predicts validation every epoch
   for early stopping — cache those prediction vectors, select the top-k epochs by
   validation primary from the CACHE (no re-forward), keep the k state dicts for
   test-time averaging. The selection may add only metric-evaluation time (seconds),
   not forward passes. `snapshot_k=3` becomes the default for deepfm/dcnv2/mmoe/ple/
   din_lite. Record per-model in info whether the ensemble beat the single best.
2. Trainer throughput: profile one ple epoch; precompute per-user row-index arrays
   once per fit (shuffle user order per epoch, not rebuild), and try batch 8192 with
   lr scaled. Target: ≥ 8 full-data ple epochs inside 300 s (v1 did 6).
3. Recency sample weights: config knob `recency_half_life_days` (None off) — lgbm via
   Dataset(weight=...), torch via per-row weight on the bce term; weight =
   0.5 ** ((train_end - row_date)/half_life). Grid {None, 3, 7} on ple and
   lgbm_pointwise, one bench each.
4. Unit checks added to tests/ for: snapshot selection uses cached predictions
   (no extra forward), recency weights alter the loss (weighted vs unweighted differ),
   and determinism still holds (two fits → identical predictions).

Gate (environment-correctness): `uv run pytest -q` all green (40 + the new tests);
`python -m recsys.zoo bench --budget 300 --models ple,deepfm` completes with total
per-model wall ≤ 330 s. Research target (non-blocking): ple ≥ 0.6065.
Commits: `v2.0: budget-free snapshots`, `v2.0: trainer throughput`, `v2.0: recency
weights`, `v2.0: tests + bench`. Push, tag v2-phase-0. Append to reports/v2.md.

### Phase V2.1: positive-history attention

Deliverables
1. recsys/features/blocks.py: new `seq_pos` block — the user's last 20 POSITIVE
   (long_view == 1) impressions before this row (video/author/tag ids, duration,
   watch ratio), strict-past for train rows, training-window for val/test, padded.
   Same leakage contract as `seq`; docstring states it.
2. Spec `full_seqpos` (v1) = full + seq_pos. din_lite gains a config flag
   `history="all"|"positive"` selecting which sq_/sp_ columns it attends over
   (registry default stays as-is until benched).
3. tests/test_features.py extended: time-safety for seq_pos (shuffling later labels
   leaves earlier rows' seq_pos unchanged; flipping an earlier positive changes it),
   plus group order and no-label checks for the new spec.

Gate: pytest green; `python -m recsys.features build --spec full_seqpos --split val`
runs; bench `din_lite` with history=positive at --budget 300. Research target
(non-blocking): beat din_lite 0.6048; if it beats ple 0.6062, promote it into the
blend candidates of V2.2. Commits: `v2.1: seq_pos block + spec`, `v2.1: din positive
history + tests + bench`. Push, tag v2-phase-1.

### Phase V2.2: blend v2

Deliverables
1. Three-base blends within one 300 s budget, using V2.0/V2.1 winners. Candidate sets
   (shares must sum ≤ 1.0 with measured fit times): {fm 0.12, deepfm 0.38, ple 0.50};
   {deepfm, ple, din_pos} if din_pos earned it; {fm, ple, ple(seed 1)} as a bagging
   control. Rank-average with a small weight grid (≤ 12 combos) chosen on validation
   (same precedent as the mmoe head grid — document the selection cost).
2. `python -m recsys.zoo bench --budget 300 --models blend` with the new default =
   best found; keep v1's pair as config fallback. Every candidate's number appended to
   reports/zoo_baselines.md (bench them via --models with temporary configs or a small
   scratch script whose results are transcribed into reports/v2.md).

Gate: tests green; blend bench completes ≤ 330 s wall. Research target (non-blocking):
blend ≥ 0.6076 (v1 + 0.0005); stretch 0.6091 (+ one ε). If missed: one page of
analysis in reports/v2.md (what was tried, per-candidate numbers, why the ceiling
holds), keep the best config as default. Commits: `v2.2: blend v2 (+ number)`.
Push, tag v2-phase-2.

### Phase V2.3: the scored autonomous run + deliverables refresh

This is the point of everything. Environment work stops; the research agent runs.

1. Preflight: clean main, all v2 phases pushed; `python prepare.py --verify` exit 0;
   `uv run pytest -q` green; re-read program.md (do not edit it).
2. Kick off exactly per IMPLEMENTATION.md Appendix C: read program.md and run it end
   to end — `python -m harness start --run-id <tag from today's date>`, iteration 1 =
   unmodified train.py ("baseline reproduction"), then iterate until the harness
   prints STOP. The harness owns git; you own hypothesis.md and notes.md. Any human
   message that arrives mid-run is logged with `python -m harness intervene`.
   Note: train.py's registry defaults now start at the v2 blend — iteration 1 still
   reproduces the FM baseline because train.py's contract defaults are model fm /
   spec fm5; the zoo's stronger rungs are the agent's second and later iterations.
3. `python -m harness finish --tokens-in <N> --tokens-out <N>` using this session's
   actual token usage (estimate from the client's usage display; state the estimation
   method in reports/v2.md), then `python -m harness report --run-id <tag>`.
4. Deliverables refresh (on main, after merging or checking out the run's artifacts —
   the run branch stays intact for audit): update deliverables/results_summary.md and
   the headline blocks of deliverables/README.md and the root README.md with the run's
   converged validation-best and its deltas; point deliverables/iteration_logs.md at
   `runs/<tag>/` and `reports/<tag>/`; replace deliverables/submission/ contents with
   the run's `submissions/<tag>/final.csv` (+ its config.json); update the resources
   table from `runs/<tag>/resources.json` (tokens, wall-clock, iterations,
   0 GPU-hours, intervention count).

Gate (environment-correctness, blocking): the run reaches STOP with status
converged|cap|ceiling; final.csv passes `submit.py --check`; resources.json written;
one commit per iteration on the run branch; deliverables updated and consistent with
the run ledger. Commits (main): `v2.3: deliverables refresh from run <tag>`.
Push, tag v2.0-release.

### Phase V2.4 (optional): 1k bonus run

Only if wall-clock allows after V2.3. `python -m harness start --run-id <tag>-1k
--dataset 1k --time-budget 900`, same loop, same rules; deliverables gain a 1k row.
Skipping this phase is recorded in the checklist, not a failure.

## 4. Master checklist

Phase V2.0: free budget
- [x] budget-free snapshot selection (cached predictions), default snapshot_k=3
- [x] trainer throughput: >= 8 full-data ple epochs in 300 s (or measured best, documented)
- [x] recency_half_life_days knob (lgbm + torch), grid benched
- [x] new unit tests green; full pytest green; bench walls <= 330 s
- [x] reports/v2.md section; pushed, tagged v2-phase-0

Phase V2.1: positive-history attention
- [x] seq_pos block + full_seqpos spec, time-safety tests green
- [x] din_lite history=positive benched
- [x] reports/v2.md section; pushed, tagged v2-phase-1

Phase V2.2: blend v2
- [x] three-base candidates benched within one 300 s budget; weight grid on validation
- [x] blend default updated to the best found (numbers in zoo_baselines.md + v2.md)
- [x] reports/v2.md section; pushed, tagged v2-phase-2

Phase V2.3: scored run + deliverables
- [x] preflight green (verify, pytest, clean main)
- [ ] scored run completed to STOP per program.md; interventions logged (target 0)
- [ ] finish with real token numbers; report generated
- [ ] deliverables/ + README headlines refreshed from the run ledger; final.csv replaced
- [ ] pushed, tagged v2.0-release

Phase V2.4 (optional): 1k bonus run
- [ ] run completed and deliverables updated — or explicitly skipped, noted here

## Appendix: goal prompt for the new chat

Paste exactly this (as a /goal if available, else as the first message):

```
Read IMPLEMENTATION_V2.md and execute it phase by phase, in order. Tick each item in
its Master checklist only after the gate command has passed, commit after every
completed deliverable, push and tag at the end of every phase. Do not skip a gate.
Environment-correctness gates block per the guide; research-gain targets never block —
if one is missed, write the analysis into reports/v2.md, keep the best configuration
found, and continue. Phase V2.3's scored run follows program.md exactly: the harness
owns git and the ledger, you write hypothesis.md before every iterate and notes.md
after, and you run until the harness prints STOP — never stopping early and never
asking me anything. Log any message I send during the scored run with
`python -m harness intervene` before acting on it. Report real token numbers at
finish. Stop only for the blocking conditions listed in the guide; otherwise do not
ask me anything until the last checklist box is ticked.
```
