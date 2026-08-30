# IMPLEMENTATION_V3.md: temporal dynamics (DMD) — build, falsify, then run scored

## 0. Read me first

v1 (IMPLEMENTATION.md, tag `v1.0`) built the environment; v2 (IMPLEMENTATION_V2.md,
tags `v2-phase-0..2`, `v2.0-release`, `v2-phase-4`) raised the ceiling to validation
primary **0.6075** (deterministic deepfm+ple snapshot blend) and produced the first
scored run (`aug30`: converged, 0 interventions). **Do not modify IMPLEMENTATION.md or
IMPLEMENTATION_V2.md — they are records.** This guide is v3: give the environment a
*temporal-dynamics* capability — Dynamic Mode Decomposition features per
[docs/dmd_handoff.md](docs/dmd_handoff.md) (self-contained; no textbook needed) —
falsify it honestly against a plain temporal control, and run the next scored
autonomous run. The competition scores the delta on the hidden test, the autonomy of
the loop, and the auditability of the ledger; every phase below serves one of those.

How to work through this file
- Phases V3.0 to V3.4, in order. Each phase lists deliverables, a gate, commit
  points, and appends a section to a single running report `reports/v3.md` (do not
  create per-phase files). Decisions without confirmation go to reports/decisions.md
  (new "v3" section).
- Progress lives in the Master checklist at the bottom. Tick a box only after its
  gate passed; commit the tick with the deliverable. On a new session: read this
  checklist, `git log --oneline -20`, `reports/v3.md`, then resume at the first
  unticked item.
- **Two kinds of gates**, exactly as in v2. *Environment-correctness gates* (tests
  green, parity intact, budgets respected, artifacts produced, leakage tests green)
  block on failure: write reports/BLOCKED.md, commit, push, stop. *Research-gain
  targets* (score X) never block: if missed, write the analysis into reports/v3.md,
  keep the best configuration found, continue. v2's recency-weights result is the
  model: implemented, benched, rejected as default, documented — that is a completed
  deliverable, not a failure.
- Blocking conditions (the only stops): `python prepare.py --verify` failing; the
  test suite failing in a way a mechanical fix cannot recover; the scored run unable
  to start (git/branch corruption). Everything else: decide, log, continue.

Commit and push protocol
- Environment phases work on main. Messages `v3.<n>: <what>`. Push at each phase
  end, tag `v3-phase-<n>`. The scored run manages its own `autoresearch/<tag>`
  branch via the harness — never commit manually during it (post-STOP artifact
  commits, as in v2.3/v2.4, are allowed and documented).
- Tracked-file rules unchanged: nothing over 20 MB; data/, checkpoints/, out dirs,
  stdout logs, bundles stay untracked.

Ground rules carried over (the competition requirements, condensed)
- CLAUDE.md is the constitution. Frozen: starter_kit/, prepare.py, harness/,
  tests/ (new tests may be ADDED under tests/ for new mutable-zone behavior).
  Mutable: train.py, recsys/**.
- Data policy is non-negotiable and DMD is exactly the kind of feature it exists
  for: **no label after HISTORY_END (20220421) may be used to fit anything** — DMD
  operators, snapshot statistics, standardization constants, mode selections, all of
  it. No cross-row aggregation inside an evaluation split: DMD features for a
  val/test row come from an operator fitted on the training window and *forecast
  forward* to the row's date; the val/test rows themselves never enter a snapshot.
  Only KuaiRand files feed KuaiRand models; numpy is the only linear-algebra
  dependency (numpy.linalg has svd/eig/pinv — no new packages).
- Every model fits + predicts within the 300 s budget on CPU; feature building runs
  inside train.py's process, and the watchdog kills at 2x budget + 120 s — so the
  dmd block must build in tens of seconds, not minutes (it will: the SVDs are on
  matrices with at most 13 columns).
- Never reimplement the metric; the harness recomputes metrics from saved arrays.
- Scored runs are **pure-only**: the frozen starter-kit checker validates against
  Pure regardless of dataset (runs/aug30-1k/HARNESS_BUGS.md). Do not attempt a 1k
  scored run in v3.
- Determinism: fixed seed + fixed inputs → identical outputs. For DMD specifically:
  order eigenpairs canonically (by −|λ|, then −Re λ, then sign of Im λ) and define
  per-entity features **sign-invariantly** (through Φ Λ^h b products or |φ_k[e]·b_k|
  magnitudes), so BLAS eigenvector sign/order quirks cannot leak into features.
  v2's other determinism lesson also applies: no wall-clock-dependent stops in
  anything benched as a default.

## 1. Starting state (verify before phase V3.0)

`git describe --tags` reaches v2-phase-4; `uv run python prepare.py --verify` prints
parity fm 0.6015 / random 0.4827 / popularity 0.5807 and exits 0; `uv run pytest -q`
= 51 passed; reports/zoo_baselines.md's blend rows show 0.6075; train.py's contract
defaults are still model fm / spec fm5 (a run's iteration 1 must reproduce the
baseline). If any of these fail → blocking condition.

## 2. Why DMD, and the honest priors (read before building)

The hypothesis (docs/dmd_handoff.md §1, §22): user/item behavior is a time-evolving
system; low-rank linear dynamics fitted on daily snapshots of the training window
may carry signal about *where* activity is moving that static aggregates miss.

What the existing evidence says, for and against:
- For: the split is chronological; the decayed item-popularity rung beat the plain
  one (0.5816 vs 0.5807); hidden test extends 10 days past training, so a feature
  that extrapolates forward addresses the actual deployment gap.
- Against: v2's recency-weighted *training* was uniformly negative (this window has
  little drift worth paying data for); the training window is 14 days → at day
  grain the snapshot matrix has 13 transitions, so every DMD here lives in a
  rank-≤12 space fitted on 13 points. Modest expectations are the calibrated ones:
  v1/v2 found no single feature family worth more than ~+0.001; treat any
  reproducible +0.001 as success and anything more as a surprise to verify.
- Where score lives (unchanged from v1): within-user ranking cancels anything
  constant in a user's ~5-impression list. Day-level global dynamics are therefore
  worthless on their own; the load-bearing DMD features are the **item-side** ones
  (per-video / per-author / per-tag / per-duration-bucket forecasts), which vary
  across a user's list. Global spectral summaries (spectral radius, mode counts)
  are near-constant per day — do not spend columns on them.
- Attribution discipline (handoff §15): a DMD gain is only a DMD gain if it beats a
  *plain temporal control* given the same information. Every DMD bench in v3 is
  therefore read against `full_roll` (below), not against `full`.
- Out of scope for v3, recorded here so nobody re-litigates: DMDc (logged exposures
  are policy-confounded, handoff §9), HMM comparison, kernel/extended DMD, mrDMD
  beyond a 2-level probe (13 snapshots cannot support deep recursion), compressed
  DMD, and "DMD on the agent's own experiment trajectory" (5-iteration runs are far
  too short). Any of these may return in v4 if v3 finds real temporal signal.

## 3. Phases

### Phase V3.0: snapshots + temporal EDA

Deliverables (mutable zone: new module recsys/dmd.py; tests added)
1. `recsys/dmd.py::snapshots(dataset="pure")` → per-day, per-entity state built
   ONLY from `prepare.load("train")`: for each axis in {video top-500 (+pooled
   "other"), author top-200 (+other), tag (~46), dur_bucket (10), tab (~15)} and
   each train day d, the smoothed long_view rate ((pos + 20·gmean_prefix)/(n + 20))
   and z-scored log1p impression count. Returns the joint matrix (n_state x 14),
   axis index maps, and a `prefix(d)` view for strict-past fits. Top-K memberships
   and standardization constants are computed from the fitting window only (the
   prefix, for prefix fits). Deterministic; pure polars+numpy.
2. Temporal EDA appended to reports/v3.md: per-axis lag-1/2/3 autocorrelation of
   daily rate vectors; singular-value spectrum of the joint X (energy retained at
   r ∈ {4, 8, 12}); and a pre-registered internal probe — fit on days 0–10,
   predict days 11–13, report DMD 1-step MAE vs the persistence baseline
   (yesterday's rate) per axis. These numbers pick the default state/rank and are
   informational: a weak probe narrows V3.2's scope, it does not block.
3. tests/test_v3_dmd.py: snapshot shapes and finiteness; determinism (two builds
   byte-identical); structural leakage guard (the builder consumes a frame whose
   max date ≤ HISTORY_END — assert it).

Gate: `uv run pytest -q` all green (51 + new); EDA numbers present in reports/v3.md.
Commits: `v3.0: snapshot builder + temporal eda`. Push, tag v3-phase-0.

### Phase V3.1: DMD core (classical + delay), synthetic-verified

Deliverables
1. `recsys/dmd.py::fit(S, r, s=1, damp=None)` implementing the handoff §2.2/§20
   algorithm exactly (truncated SVD → Ã = Uᵣ*X'VᵣΣᵣ⁻¹ → eig → Φ = X'VᵣΣᵣ⁻¹W), with
   s>1 delay-stacking per §3, canonical eigen ordering, amplitudes anchored at the
   window's LAST state (b = Φ⁺x_last, so forecasts extrapolate from where the
   window ends), and optional eigenvalue damping λ ← λ/max(1,|λ|) as the
   noise-defense knob (§8's spirit; rank truncation is the first defense).
2. `forecast(model, h)` → real(Φ Λ^h b), rates clamped to [0, 1] after inverse
   scaling (a 17-day extrapolation with |λ|>1 must not explode into features).
3. `entity_features(model, axis, h)` → per-entity [pred_rate, trend (pred − last
   observed), growth (|λ| of the entity's dominant mode by |φ_k[e]·b_k|), osc
   (|arg λ| of that mode)] — all sign-invariant.
4. Synthetic tests: plant a linear system with known eigenstructure (real decaying
   0.7, real growing 1.05, oscillatory pair 0.95·e^{±iθ}) plus small noise: fit
   recovers each |λ| within 0.05 and beats persistence on 3-step forecast MSE;
   delay s=3 recovers dynamics when only 1 of 2 coupled dims is observed;
   determinism across two fits.

Gate: pytest green. Commits: `v3.1: low-rank + delay DMD core (synthetic-verified)`.
Push, tag v3-phase-1.

### Phase V3.2: feature blocks, specs, leakage tests, the E-ladder bench

Deliverables
1. blocks.py `roll` block — the plain temporal control (handoff E1): per-entity
   last-3-day smoothed rate and its delta vs the full-window rate, for
   video/author/tag/dur_bucket (8 columns). Strict past for train rows at DAY grain
   (a row on day d uses days < d only — stricter than the time_ms contract, never
   looser), full training window for val/test. Same padding/fill conventions as
   existing blocks; docstring states the contract.
2. blocks.py `dmd` block: for the same 4 axes, the 4 entity features from V3.1 =
   16 columns. Train rows on day d: features from `fit(prefix(d))` (days < 3 get
   fill values — same convention as thin strict-past histories). Val/test rows:
   one fit on all 14 train days, h = row date − 20220421 (1..17). Defaults from
   V3.0's EDA (state = joint, r = 8, s = 1 unless the probe said otherwise);
   knobs live in the block and any change bumps the spec version.
3. Specs: `full_roll` (v1) = full + roll; `full_dmd` (v1) = full + roll + dmd —
   so full_dmd vs full_roll isolates DMD's marginal exactly.
4. tests/test_features.py extended: later-label shuffle leaves earlier rows'
   roll/dmd features unchanged; flipping an earlier day's positive changes a later
   row's features; no-label/group-order checks for both specs; two-build
   determinism for full_dmd; full_dmd train build ≤ 120 s single-core.
5. Bench the ladder (budget 300, seed 0): lgbm_pointwise and deepfm on full_roll
   and full_dmd (4 rows; `full` reference rows exist: lgbm 0.6012, deepfm 0.6066).
   Promotion rule: if (full_dmd − full_roll) ≥ +0.0010 on either model, also bench
   ple and blend on full_dmd (registry defaults change only if they win).

Gate: pytest green; benches complete with walls ≤ 330 s; all numbers in
zoo_baselines.md and transcribed with deltas into reports/v3.md. Research targets
(non-blocking): H1 full_roll ≥ full + 0.0005 (temporal signal exists at all);
H2 full_dmd ≥ full_roll + 0.0010 (dynamics beat plain recency). If H2 misses on
both models, the analysis page in reports/v3.md must say which of the four feature
families was closest to useful and why (feature importances from the lgbm run are
the cheapest evidence). Commits: `v3.2: roll + dmd blocks, specs, leakage tests`,
`v3.2: E-ladder bench (+numbers)`. Push, tag v3-phase-2.

### Phase V3.3: variants — expand signal or falsify cleanly

If V3.2 showed H2 signal (either model): up to 6 single-change benches, one commit
each, chosen from: delay s ∈ {2, 3}; rank r ∈ {4, 12}; damped vs clamped forecasts;
per-axis separate DMDs vs the joint state; amplitude-sparse mode selection (keep
top-q modes by |b_k|·‖φ_k‖ — the cheap proxy for handoff §6's L1 program; document
the deviation); half-day snapshots (28 columns) if rank starved. Promote winners
into the blend's spec and re-bench blend once.

If V3.2 showed no H2 signal: exactly 2 falsification probes (delay s=3 — the
handoff's highest-priority extension — and r=12), then a one-page negative result
in reports/v3.md: what was tried, per-variant numbers, why 13 daily snapshots of
this window do not contain exploitable linear dynamics beyond recency. Registry
defaults stay unchanged. This outcome is a legitimate, complete phase.

Gate: pytest green; each bench ≤ 330 s wall. Commits: `v3.3: dmd variants
(+numbers)` or `v3.3: negative result — dynamics ≤ recency here`. Push, tag
v3-phase-3.

### Phase V3.4: the scored autonomous run + conditional deliverables refresh

1. Preflight: clean pushed main; `python prepare.py --verify` exit 0;
   `uv run pytest -q` green; re-read program.md (do not edit it); train.py defaults
   still fm/fm5.
2. Kick off exactly per IMPLEMENTATION.md Appendix C: pick a run tag from today's
   date, `python -m harness start --run-id <tag>` (pure, default budgets),
   iteration 1 = unmodified train.py ("baseline reproduction"), then iterate until
   the harness prints STOP. The harness owns git and the ledger; you write
   hypothesis.md before every iterate and notes.md after. Any human message that
   arrives mid-run is logged with `python -m harness intervene` before acting.
   Expected-value ordering with the v3 zoo evidence in hand: bank the strongest
   known configuration early (the convergence window gives roughly three
   post-banking shots), then spend those shots on the best-evidenced DMD/temporal
   hypotheses from V3.2/V3.3 — with real mechanisms in hypothesis.md, not hope.
   Do not game ε with micro-gains; reverts that teach something are good ledger.
3. `python -m harness finish --tokens-in <N> --tokens-out <N>` with numbers read
   off the session's metered token pool (markers at kickoff and finish; state the
   split method in reports/v3.md, as v2 did), then
   `python -m harness report --run-id <tag>`.
4. Deliverables: if the run's converged best beats 0.6075, refresh
   deliverables/results_summary.md, both README headline blocks,
   deliverables/iteration_logs.md, submission/ (final.csv + config.json), and the
   resources table from runs/<tag>/resources.json — exactly the v2.3 procedure.
   If it does not beat 0.6075, do NOT demote the headline: add the run as an
   additional audited run (iteration_logs.md section + resources row) and say so
   in reports/v3.md. Either way the run branch stays intact and pushed.

Gate (blocking): run reaches STOP with status converged|cap|ceiling; final.csv
passes submit.py --check; resources.json written with real token numbers; one
commit per iteration on the run branch; main's deliverables state consistent with
the decision rule above. Commits (main): `v3.4: deliverables from run <tag>` or
`v3.4: run <tag> recorded (headline unchanged)`. Push, tag v3.0-release.

## 4. Master checklist

Phase V3.0: snapshots + temporal EDA
- [ ] recsys/dmd.py snapshot builder (train-window only, deterministic, tested)
- [ ] temporal EDA in reports/v3.md (autocorr, sv spectrum, persistence-vs-DMD probe)
- [ ] pytest green; pushed, tagged v3-phase-0

Phase V3.1: DMD core
- [ ] fit/forecast/entity_features (classical + delay, damping knob, canonical order)
- [ ] synthetic recovery + determinism tests green; pushed, tagged v3-phase-1

Phase V3.2: blocks, specs, E-ladder
- [ ] roll + dmd blocks with day-grain strict-past/train-window contract + leakage tests
- [ ] full_roll / full_dmd specs; build ≤ 120 s test; determinism test
- [ ] E-ladder benched (lgbm + deepfm x full_roll/full_dmd; promotions per rule)
- [ ] reports/v3.md section with H1/H2 verdicts; pushed, tagged v3-phase-2

Phase V3.3: variants or clean falsification
- [ ] variant benches per the signal rule (or 2 probes + negative-result page)
- [ ] registry defaults updated only where a variant won; pushed, tagged v3-phase-3

Phase V3.4: scored run + deliverables
- [ ] preflight green (verify, pytest, clean main, fm defaults)
- [ ] scored run to STOP per program.md; interventions logged (target 0)
- [ ] finish with real token numbers; report generated
- [ ] deliverables refreshed or run recorded per the beat-0.6075 rule
- [ ] pushed, tagged v3.0-release

## Appendix: goal prompt for the executing chat

Paste exactly this (as a /goal if available, else as the first message):

```
Read IMPLEMENTATION_V3.md and execute it phase by phase, in order. Tick each item in
its Master checklist only after the gate command has passed, commit after every
completed deliverable, push and tag at the end of every phase. Do not skip a gate.
Environment-correctness gates block per the guide; research-gain targets never block
— if one is missed, write the analysis into reports/v3.md, keep the best
configuration found, and continue: a cleanly falsified DMD hypothesis is a completed
deliverable. Respect the data policy everywhere: nothing fitted on labels after
HISTORY_END, DMD operators and snapshot statistics included; val/test rows never
enter a snapshot. Phase V3.4's scored run follows program.md exactly: the harness
owns git and the ledger, you write hypothesis.md before every iterate and notes.md
after, and you run until the harness prints STOP — never stopping early and never
asking me anything. Log any message I send during the scored run with
`python -m harness intervene` before acting on it. Report real token numbers at
finish. Stop only for the blocking conditions listed in the guide; otherwise do not
ask me anything until the last checklist box is ticked.
```
