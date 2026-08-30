# IMPLEMENTATION.md: KuaiRand-Pure autoresearch harness

## 0. Read me first

What I am building: an autoresearch-style research harness (karpathy/autoresearch) for the ByteDance "Autonomous ML Research Agent for Recommender Systems" track, on the KuaiRand-Pure ranking benchmark. autoresearch has three files that matter: prepare.py (frozen: data and evaluation), train.py (the one file the research agent edits), program.md (the agent's instructions, owned by the human). I keep that shape and add what the challenge scores: a recsys/ environment library (features, models, losses, blending) so the research agent composes instead of writing from scratch, and a harness/ that runs the loop, does the git work, writes the per-iteration logs the judges read (hypothesis, diff, metrics, error and recovery events), enforces the convergence rule and the caps, and always keeps a validated submission for the validation-best checkpoint.

How to work through this file
- Phases 0 to 7, in order. Each phase lists deliverables, a gate (commands that must pass), the commit points, and the phase report.
- Progress lives in the Master checklist at the bottom of this file. Tick a box only after its gate command has passed. Commit the tick together with the deliverable. On any new session: read the checklist, `git log --oneline -30`, and the latest reports/phase*.md, then resume at the first unticked item.
- Do not ask me for confirmation between phases or deliverables. Decide, write the decision in reports/decisions.md, continue.
- Blocking conditions, the only reasons to stop: inputs missing (section 1); baseline parity failure (Phase 1 gate); zoo gate failure (Phase 3 gate); a starter kit file that contradicts Appendix A. On a block: write reports/BLOCKED.md with the exact output, commit, push, stop.

Commit and push protocol
- Work on main. Commit after every completed deliverable with message `phase<N>: <what>`. Push at the end of every phase and tag `phase-<N>`. If a push fails (no network, no remote), keep committing locally, retry at the next phase end, and note it in the checklist.
- Never commit data/, checkpoints/, iteration out/ directories, stdout logs, or any file over 20 MB. Tracked: results.tsv, runs/**/{hypothesis.md,diff.patch,metrics.json,events.jsonl,config.json,notes.md,interventions.jsonl,run.json,resources.json}, submissions/**/final*.csv.
- Every phase ends with reports/phase<N>.md containing the gate commands and the tail of their real output.

Context hygiene: redirect long output to files and tail them. Never cat data files, caches, or full training logs. Every script prints a short summary and writes the rest to disk.

## 1. Inputs and environment

Expected in the repo before I start (I placed them; they are gitignored):
- ./kuairand-starter-kit.zip, the organizer starter kit (data.py, baseline.py, evaluate.py, submit.py, maybe more).
- ./data/raw/: log_standard_4_08_to_4_21_pure.csv, log_standard_4_22_to_5_08_pure.csv, user_features_pure.csv, video_features_basic_pure.csv, video_features_statistic_pure.csv. log_random_4_22_to_5_08_pure.csv may be present and is never read.
- Python 3.11, uv (install with `curl -LsSf https://astral.sh/uv/install.sh | sh` if missing), git with origin configured.
Missing zip or missing raw CSVs is a blocking condition.

## 2. Spec

The pinned numbers, the three zones, the data policy, the train.py contract, the definition of an iteration and the ledger schema are all in Appendix A (CLAUDE.md). Appendix A is the spec; this file is the build order. Appendix B is program.md, the research agent's runtime instructions, written verbatim in Phase 4.

Known contradiction in the problem statement: its "Limits" row says NDCG@10 / Recall@50 with click as positive; the Starter Kit section, the Benchmarks table and Appendix A.4 of the statement say long_view with GAUC / nDCG@5. The starter kit's evaluate.py is the ground truth. If baseline.py does not reproduce primary 0.6016 on validation, that is the Phase 1 block.

Decisions already made
- CPU-first. Everything runs on one CPU core inside the time budget; GPU is auto-detected and optional.
- Python 3.11, uv, ruff, pytest. numpy, polars, pyarrow, lightgbm, torch (CPU wheels), scikit-learn for utilities only.
- One experiment per iteration, 300 s training wall-clock per experiment by default.
- The harness does all git operations during a research run. One commit per iteration.
- HISTORY_END defaults to 20220421. Validation labels are never used for fitting except in the optional, clearly labelled refit at finish.
- No transductive features: nothing is aggregated across rows of the validation or test split.
- train.py defaults to the FM baseline (model fm, feature spec fm5) so a research run's iteration 1 reproduces the official baseline.

## 3. Target layout

```
CLAUDE.md  program.md  IMPLEMENTATION.md  README.md  pyproject.toml  Makefile  results.tsv
starter_kit/      organizer files, vendored unchanged + MANIFEST.sha256 + NOTES.md
prepare.py        frozen: load(), tables(), evaluate(), write_submission(), --build, --verify
train.py          mutable: composes features + model + loss + training
recsys/           mutable: features/  models/  losses.py  zoo.py  eda.py
harness/          frozen: __main__.py run.py iterate.py git_ops.py convergence.py ledger.py watchdog.py report.py
tests/            frozen for the research agent: parity, guards, features, models, harness, fault injection, scripted_agent/, fixtures/
docs/             autoresearch/ (vendored README.md + program.md), devpost.md
reports/          eda.md zoo_baselines.md phase*.md decisions.md <run_id>/
data/             raw/ cache/ (gitignored)
runs/ checkpoints/ submissions/
```

## 4. Phases

### Phase 0: scaffold and constitution

Deliverables
1. pyproject.toml (uv, Python 3.11, deps above, ruff and pytest config), .python-version, .gitignore (data/, checkpoints/, runs/*/iterations/*/out/, runs/*/iterations/*/stdout.log, submissions/*/iter_*.csv, *.tar.gz), Makefile with targets setup, build, verify, test, bench, iterate, report.
2. CLAUDE.md written verbatim from Appendix A.
3. docs/autoresearch/README.md and docs/autoresearch/program.md vendored from https://github.com/karpathy/autoresearch (MIT). If the network is off, leave a one-line note in docs/autoresearch/MISSING.md and continue.
4. reports/decisions.md started with the decisions list from section 2.
5. Verify inputs from section 1 exist (blocking if not).

Gate: `uv sync` succeeds; `ruff check .` passes on the empty scaffold.
Commits: `phase0: scaffold`, `phase0: CLAUDE.md`, `phase0: vendor autoresearch docs`. Push, tag phase-0.

### Phase 1: frozen layer, baseline parity, guards, EDA

Deliverables
1. starter_kit/: unzip the starter kit here unchanged. Write starter_kit/MANIFEST.sha256. Read every file and write starter_kit/NOTES.md: exact columns data.load() returns per split, the row-ordering rule, which 5 categorical fields the FM uses, whether the convergence rule ships as code, submit.py's CLI, and any contradiction with Appendix A (a contradiction is a block).
2. prepare.py:
   - load(split, dataset="pure", history_end=None) -> polars frame in the exact row order of starter_kit data.load(). For "test", drop every feedback column: is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter, and anything NOTES.md identifies as post-impression.
   - tables() -> the static KuaiRand tables (user_features_pure, video_features_basic_pure, video_features_statistic_pure).
   - evaluate(split, scores) -> dict(gauc, ndcg5, primary) by calling starter_kit.evaluate on the split's labels. Raises on split == "test".
   - write_submission(split, scores, path): pinned CSV schema, then `python3 starter_kit/submit.py --check`; raises on failure.
   - --build: data/cache/{train,val,test}.parquet, static tables, data/cache/MANIFEST.sha256. --verify: starter kit manifest, cache manifest, split sizes (1,141,112 / 124,909 / 170,588), test frame has no label columns, the three parity rungs below. Idempotent, exit 0 only when all pass.
3. tests/fixtures/make_fixture.py: deterministic 20k-row subsample of the cache by sampling users (train, val, test slices with the same date rule), written to data/cache/fixture_small/. All later unit tests run on it.
4. tests/test_parity.py:
   - `python3 starter_kit/baseline.py --model fm` via subprocess, parse its validation numbers: |primary - 0.6016| <= 0.003, |gauc - 0.6674| <= 0.003, |ndcg5 - 0.5357| <= 0.003.
   - random scores on validation: primary in [0.46, 0.49].
   - item popularity (training-window long_view count per video, 0 for unseen) on validation: primary in [0.55, 0.60].
   - hand-built 3-user toy example exercising the three pinned evaluate conventions, expected values as literals with the arithmetic in a comment.
5. tests/test_guards.py: prepare.load("test") has no feedback columns; static scan that nothing under train.py or recsys/ opens data/raw/ or calls a CSV reader on a raw path; the string log_random appears nowhere under train.py or recsys/; prepare.evaluate("test", ...) raises.
6. recsys/eda.py -> reports/eda.md (reads only through prepare): label rate overall and by duration decile, tab, hour, is_rand; impressions per user in validation (median, p90); share of validation users with zero positives and with all positives; repeated (user, video) pairs in validation; user and item overlap between train and validation; play_time_ms against duration_ms for long_view 1 versus 0.

Gate
```
python prepare.py --build
python prepare.py --verify
python tests/fixtures/make_fixture.py
pytest -q tests/test_parity.py tests/test_guards.py
test -f reports/eda.md
```
Second `--verify` run makes no changes. Commits: `phase1: vendor starter kit + NOTES`, `phase1: prepare.py`, `phase1: fixture`, `phase1: parity + guards`, `phase1: eda`. reports/phase1.md shows baseline validation numbers next to the published ones. Push, tag phase-1.

### Phase 2: time-safe feature layer

Deliverables, recsys/features/
1. spec.py: FeatureSpec = named, versioned list of blocks. build(spec, split, history_end=HISTORY_END, dataset="pure") -> (X float32 matrix or polars frame, meta with column names and categorical indices, group = user_id per row in split order). Cache at data/cache/features/<spec>-<hash>.parquet keyed on spec version, split, history_end, input manifests.
2. Blocks, each a pure function with a docstring naming the files it reads and its leakage argument in one line:
   - ctx: tab, hour of day, day of week, is_rand, days since split start.
   - item_static: from video_features_basic_pure (duration_ms, video_type, upload_type, upload age at impression time, music_type, tag ids). Sub-block item_stats from video_features_statistic_pure, switchable off, with a docstring noting its aggregation window is not pinned by KuaiRand.
   - user_static: user_features_pure columns.
   - hist_user: training-window aggregates per user (impressions, long_view rate, click rate, mean watch ratio, mean duration, distinct authors). Out-of-time for training rows (only rows strictly earlier than the row's time_ms, via cumulative sums on a time-sorted frame); full training window for validation and test rows.
   - hist_item: same per video, per author, per tag, per music_id, plus an exponentially decayed variant with a half-life parameter in days.
   - cross: user x item statistics that survive within-user ranking: user's historical long_view rate on this author, this tag, this duration bucket; user's mean watch ratio on this duration bucket; item's long_view rate among users with the same user_active_degree.
   - seq: last 20 impressions of the user before this one (video ids, author ids, tag ids, durations, watch ratios), padded.
   - target_enc: smoothed target encoding of high-cardinality ids with time-safe folds.
   - Specs registered: fm5 (the baseline's 5 fields), full (everything), full_nostats (full minus item_stats).
3. recsys/features/README.md listing every block with inputs and leakage argument, and stating up front: evaluation ranks within a user, so features constant within a user only help through interactions or a tree model.

Tests, tests/test_features.py: time safety on a synthetic 3-user log (features of a row unchanged when later labels are shuffled, changed when an earlier label flips); validation rows equal a brute-force training-window recomputation; no label column in X for any split; group equals prepare.load() user order; full spec builds on the real data in under 60 s on one core (print the time); two builds give identical hashes.

Gate
```
pytest -q tests/test_features.py
python -m recsys.features build --spec full --split val
```
Commits: `phase2: feature spec + cache`, `phase2: blocks ctx/static/hist`, `phase2: blocks cross/seq/target_enc`, `phase2: feature tests + README`. Push, tag phase-2.

### Phase 3: recommender environment (model zoo) and train.py

Deliverables
1. recsys/models/base.py:
   ```
   class Recommender:
       fit(X_train, y_train, groups_train, aux_train=None, X_val=None, y_val=None, groups_val=None, time_budget=300, seed=0)
       predict(X, groups) -> np.ndarray float32
       save(dir) / load(dir)
   ```
   y is long_view; aux carries the other 11 feedback columns plus play_time_ms and duration_ms; groups is user_id per row; validation arrays are for early stopping only and are always scored through prepare.evaluate.
2. recsys/models/__init__.py registry: name -> (class, default config, default spec). recsys/losses.py: bce, bpr_pairwise_within_user, listwise_softmax_within_user, mixed(weight).
3. Rungs, each its own commit, each appended to reports/zoo_baselines.md with its validation score:
   - random, popularity (must match Phase 1).
   - fm: numpy port of the official baseline on fm5; reproduces 0.6016 within 0.003.
   - lgbm_pointwise: LightGBM binary on full, early stopping on validation primary through a prepare.evaluate callback.
   - lgbm_lambdarank: LightGBM lambdarank, query groups = user (sort by user, keep the permutation, unsort predictions), eval_at [5]; the default label_gain already equals 2^rel - 1. Expected to beat fm.
   - deepfm, dcnv2: torch, id embeddings (user, video, author, tag, music) plus dense features, any loss from losses.py.
   - mmoe, ple: shared-bottom multi-task over long_view, click, like, follow, comment, forward, hate, and a watch-ratio regression head; score = long_view head by default, optional grid-searched combination of heads chosen on validation.
   - cwm: counterfactual watch time (statement reference [4]); censored regression on play_time_ms truncated at duration_ms (one-sided loss where play_time equals duration); rank by predicted watch time normalized by duration; pure torch.
   - din_lite: target attention over the seq block on top of dcnv2.
   - blend: rank-average of registered models, and a linear stacker fitted on out-of-fold training-window predictions built by date folds; never fitted on validation.
   Rules: every model finishes fit + predict on the full training split under 300 s of training wall-clock on one core (early stopping, subsampling, fewer epochs; record what was subsampled in its config); seed-deterministic; reads no raw files; torch auto-detects CUDA or MPS and never requires it.
4. recsys/zoo.py: `python -m recsys.zoo list` and `python -m recsys.zoo bench --budget 300 --seeds 0 [--models a,b] [--dataset pure]` -> reports/zoo_baselines.md (name, primary, gauc, ndcg5, train_sec, peak_rss_mb, spec, notes) and checkpoints/zoo/<name>/val_scores.npy.
5. train.py satisfying the Appendix A contract: parse args, build the spec, instantiate the model, fit with the time budget, predict validation and test, write val_scores.npy, test_scores.npy, config.json (model, spec, hyperparameters, rounds or epochs actually used, seed, subsampling). Defaults: model fm, spec fm5.

Tests, tests/test_models.py on the fixture: each registered model fits, predicts finite float32 of the right length, save/load round-trips to identical predictions, and fit(time_budget=10) finishes under 20 s. tests/test_train.py: `python train.py --out /tmp/t --time-budget 20` on the fixture writes the three files with the right lengths.

Gate
```
pytest -q tests/test_models.py tests/test_train.py
python -m recsys.zoo bench --budget 300
```
Zoo gate: at least one non-fm rung beats fm on validation by 0.005 or more. If not, write reports/zoo_debug.md (feature importances, per-user error analysis of the best rung, label-versus-duration from eda.md) and block. Commits: one per rung, then `phase3: zoo bench`, `phase3: train.py`. Push, tag phase-3.

### Phase 4: the autoresearch loop

Read docs/autoresearch/README.md and docs/autoresearch/program.md first. Keep their shape: one mutable surface, one metric, one branch per run, keep-or-revert, results.tsv.

Deliverables, harness/ (CLI in harness/__main__.py)
1. `start --run-id <tag> [--dataset pure] [--history-end 20220421] [--time-budget 300] [--max-iters 50] [--wall-clock-hours 6] [--eps 0.002] [--patience 3] [--seeds 0] [--resume]`: creates branch autoresearch/<tag> from main (refuses if it exists without --resume, or if git is dirty); runs `python prepare.py --verify`; writes runs/<tag>/run.json (all config, started_at, base_commit, baseline numbers, status running, best null) and the results.tsv header; prints the next command.
2. `iterate --desc "<text>" [--seeds 0,1,2] [--tokens-in N --tokens-out N]`:
   - refuses unless runs/<tag>/iterations/<next>/hypothesis.md exists and is non-empty; refuses with STOP and the reason if a stop condition already fired.
   - writes diff.patch = `git diff <last kept commit> -- train.py recsys/`.
   - runs `python train.py --out runs/<tag>/iterations/<i>/out --seed <s> --time-budget <b>` in a subprocess (harness/watchdog.py): stdout and stderr to stdout.log, hard timeout 2 x budget + 120 s, RSS sampler with a configurable cap (default 16 GB).
   - success: loads val_scores.npy, recomputes metrics with prepare.evaluate (never trusts train.py's own numbers), averages across seeds if several, writes metrics.json, decides KEEP if primary > best so far else REVERT. On KEEP: copies config, model artifacts, val_scores and test_scores to checkpoints/<tag>/best/, writes submissions/<tag>/iter_<i>.csv through prepare.write_submission (which runs submit.py --check), and points submissions/<tag>/final.csv at it.
   - git (harness/git_ops.py): exactly one commit per iteration. KEEP commits the mutable surface plus the ledger; REVERT checks out the mutable surface from the last kept commit, then commits only the ledger. `git status` is clean after every iterate.
   - failure (nonzero exit, timeout, oom, nan or inf, wrong length, missing output): appends to events.jsonl, does not consume the iteration number, allows up to 3 attempts; after the third failure it reverts the mutable surface, records the iteration as abandoned (counts toward the 50 cap, not toward the convergence window), and commits the ledger.
   - prints exactly one summary block, fixed and greppable:
     ```
     === ITERATION 7 (attempt 1) ===
     DECISION: KEEP
     primary: 0.6412  gauc: 0.7011  ndcg5: 0.5813
     delta_vs_baseline: +0.0396  delta_vs_best: +0.0021
     best_so_far: iter 7 @ a1b2c3d (0.6412)
     convergence: window [+0.0000, +0.0008, +0.0021] max +0.0021 > eps 0.0020 -> continuing
     budget: iterations 7/50  elapsed 0h52m/6h00m
     next: write runs/<tag>/iterations/8/hypothesis.md, then: python -m harness iterate --desc "..."
     ```
     On failure the block starts with `ERROR <type> (attempt k/3)` and the last 5 lines of stdout.log. On a stop condition it starts with `STOP <converged|cap|ceiling>`.
3. `abandon --reason "<why>"`, `revert` (manual fallback), `status`, `intervene --note "<what I did>"` (appends to runs/<tag>/interventions.jsonl; I run this, not the agent; it is how the manual-intervention count is produced), `finish [--tokens-in N --tokens-out N] [--also-refit]`: re-checks final.csv, writes runs/<tag>/resources.json (iterations used, scored, abandoned, wall-clock, tokens or null with a note, GPU-hours from an nvidia-smi sampler if present, intervention count, the iteration-counting interpretation), bundles out/ dirs and stdout logs into runs/<tag>/bundle.tar.gz, final ledger commit. --also-refit re-runs the best config with --history-end 20220428 and the recorded rounds or epochs and writes submissions/<tag>/final_refit_trainval.csv, clearly labelled, never replacing final.csv.
4. harness/convergence.py: if the starter kit ships the rule as code, wrap it; otherwise implement exactly this and document it: converged when the best validation primary over the last 3 scored iterations does not exceed the best before that window by more than eps. Also stop at max-iters or at the wall-clock ceiling measured from started_at, whichever comes first.
5. Divergence guard: 5 consecutive revert or abandoned iterations print a DIVERGENCE warning with the best commit hash and a pointer to the recovery advice in program.md; it never stops the run.
6. Schemas. results.tsv columns: iter, commit, status (keep | revert | abandoned), primary, gauc, ndcg5, delta_vs_baseline, train_sec, seed, description. metrics.json: iter, attempt, seeds, primary, gauc, ndcg5, per_seed, train_sec, total_sec, peak_rss_mb, n_val, n_test, decision, best_before, delta_vs_best, delta_vs_baseline. events.jsonl rows: ts, iter, attempt, type (error | timeout | oom | nan | shape | missing | divergence | stop), detail, action.
7. program.md written verbatim from Appendix B.
8. `report --run-id <tag>` is Phase 6; leave a stub that says so.

Tests, tests/test_harness.py on the fixture, under 3 minutes total: convergence sequences that must and must not converge, an abandoned iteration inside the window, a gain of exactly 0.002 (no improvement); iterate refuses without hypothesis.md and after STOP; metrics are recomputed (plant a train.py that prints primary 0.99; the ledger records the true value); KEEP produces a submission that passes --check; REVERT restores the mutable surface byte for byte; one commit per iteration and git clean afterwards; failure paths (syntax error, sleep past the timeout, NaN scores, wrong length, missing output, allocation past the RSS cap) yield the right events type, the harness never raises, and the run continues; wall-clock ceiling with a fake clock.

Gate
```
pytest -q tests/test_harness.py
python -m harness start --run-id smoke --max-iters 3
# write runs/smoke/iterations/1/hypothesis.md ("baseline reproduction"), then:
python -m harness iterate --desc "baseline reproduction"
python -m harness finish
```
The smoke results.tsv row equals the fm rung within 0.003 and submissions/smoke/final.csv passes --check. Then `git checkout main && git branch -D autoresearch/smoke`. Commits: `phase4: run state + git ops`, `phase4: watchdog + iterate`, `phase4: convergence + finish`, `phase4: harness tests`, `phase4: program.md`. Push, tag phase-4.

### Phase 5: scripted dry run and fault injection

Deliverables
1. tests/scripted_agent/: a driver that plays the research agent. It works in a temporary clone (`git clone . /tmp/scripted-<ts>`) so the real repo is never dirtied, on the fixture with --time-budget 20, using tests/scripted_agent/train_stub.py copied over train.py. The stub produces val and test scores from a deterministic rule with a QUALITY constant q in [0, 1] (for example q x a time-safe popularity feature + (1 - q) x seeded noise) so that primary rises monotonically with q and every KEEP/REVERT is predictable. Sequence:
   ```
   01 q=0.20                                   -> keep (baseline)
   02 q=0.30                                   -> keep
   03 syntax error, fixed on attempt 2, q=0.10 -> error event, then revert
   04 q=0.90                                   -> keep
   05 infinite loop                            -> timeout x 3, abandoned
   06 NaN on attempt 1, fixed, q=0.50          -> nan event, then revert
   07 q=0.90                                   -> revert (equal, not better)
   08 q=0.85                                   -> revert, then STOP converged (window 06, 07, 08 against best at 04)
   09 driver calls iterate again               -> harness refuses with STOP
   ```
   The driver asserts: results.tsv has 8 rows (7 scored, 05 abandoned); kept commits are 01, 02, 04 only; final.csv is the 04 submission and passes --check; events across the run: one error, three timeouts, one nan, one stop; run.json status converged at iteration 8; mutable surface at the end equals the 04 commit byte for byte; every iteration directory has the full artifact set; exactly one commit per iteration. It prints an expected-versus-observed table and exits nonzero on any mismatch.
2. `--fixture full` mode: the same driver on the real data with the real train.py and a 300 s budget, asserting only behaviour (run completes, artifacts exist, final.csv passes, one commit per iteration). One rehearsal before a scored run.
3. tests/test_fault_injection.py, parametrized over the failure classes from Phase 4 plus OOM (allocate 64 GB), corrupted checkpoint on load, git dirty at start, disk full simulated on the submission write, train.py deleting its own output: no traceback escapes the harness, and the run can continue or finish cleanly afterwards.

Gate
```
pytest -q tests/test_fault_injection.py
python -m tests.scripted_agent.run --fixture small
```
Commits: `phase5: train stub + scripted driver`, `phase5: fault injection tests`. reports/phase5.md contains the scripted run's results.tsv, events and git log. Push, tag phase-5.

### Phase 6: deliverables pack and bonus datasets

Deliverables
1. `python -m harness report --run-id <tag>` -> reports/<tag>/: results_table.md (validation-best GAUC / nDCG@5 / primary, absolute deltas against 0.6674 / 0.5357 / 0.6016, iterations used of 50, plus the published hidden-test baseline row for reference), resources.md (tokens, wall-clock, iterations, GPU-hours, intervention count, iteration-counting interpretation), interventions.md, iteration_log.md (one section per iteration: hypothesis, diff summary linking diff.patch, metrics, events and how they were handled), trajectory.png (validation primary per iteration with baseline and best lines).
2. README.md with the required sections: overview, setup, reproduce (exact commands from start to finish and how to re-score final.csv), limitations and what I would do with more time (TODO markers for me), team contributions (solo).
3. docs/devpost.md skeleton: how the solution addresses the problem statement, development tools, APIs, libraries and frameworks, datasets and assets.
4. Bonus datasets: prepare.py --dataset {pure,1k,27k} with the same date boundaries; polars streaming for 1k (11.7M rows); a documented user-sampling policy for 27k (322M rows); run.json records the dataset. Do not build 27k end to end unless 1k is green.

Gate
```
python -m harness report --run-id <scripted run tag from the phase 5 clone, copied under runs/>
python prepare.py --dataset 1k --build && python prepare.py --dataset 1k --verify
python -m recsys.zoo bench --dataset 1k --models lgbm_lambdarank --budget 900
```
Skip the 1k gates only if the 1k raw files are absent; note it in the checklist. Commits: `phase6: report`, `phase6: README + devpost`, `phase6: bonus datasets`. Push, tag phase-6.

### Phase 7: final validation and release

Run the full go/no-go list and record every output in reports/final.md:
```
python prepare.py --verify
pytest -q
python -m tests.scripted_agent.run --fixture small
python -m tests.scripted_agent.run --fixture full
python -m recsys.zoo bench --budget 300
git status            # clean on main, no autoresearch/* branches left
```
Then tag v1.0, push, and tick the last checklist box. The repo is now ready for a scored run, which starts with the kickoff message in Appendix C and nothing else.

## Appendix A: CLAUDE.md (written verbatim in Phase 0)

# CLAUDE.md

I am competing in the "Autonomous ML Research Agent for Recommender Systems" track (ByteDance, KuaiRand). This repo is an autoresearch-style harness (karpathy/autoresearch) for the KuaiRand-Pure ranking benchmark. Read this file fully before touching anything.

## The task in one line
Rank each user's logged impressions in the evaluation split by P(long_view). Score = mean(GAUC, nDCG@5), computed only by the organizer's evaluate.py. Beat the official Factorization Machine baseline on the hidden test set, autonomously, with auditable per-iteration logs.

## Pinned numbers
- Splits, date-based, from log_standard_4_08_to_4_21_pure.csv and log_standard_4_22_to_5_08_pure.csv: train 20220408 to 20220421 (1,141,112 rows), validation 20220422 to 20220428 (124,909 rows), test 20220429 to 20220508 (170,588 rows).
- Official baseline: Factorization Machine, k=16, lr=0.001, 5 categorical fields, numpy only, about 40 s on one CPU core. Validation GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016. Hidden test GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946 (5-seed std 0.0008).
- Reference rungs on the hidden test: random 0.4753, item popularity 0.5715, perfect ranking 0.8645 (GAUC 1.0000, nDCG@5 0.7289, because 27.1% of test users have no positive and 9.2% are all-positive). Judge progress against 0.8645, not 1.0.
- Convergence rule: a run is converged when the validation primary has not improved by more than eps = 0.002 over the last N = 3 consecutive scored iterations. Hard caps: 50 iterations per run and a 6 h wall-clock ceiling. Whichever fires first stops the run.
- Submission: CSV with header row_id,user_id,video_id,score, one line per evaluation-split row. row_id is the 0-based index into the split in data.load() order. (user_id, video_id) is not unique (3.06% of test rows are repeated pairs, up to 12 times), so row_id is the only key. NaN and Inf are rejected. `python3 submit.py --check` is the validator; `--make` produces an example.
- evaluate.py conventions: nDCG gain is 2^rel - 1; users with zero positives score nDCG 0 and are included in the mean; GAUC uses only users with 0 < positives < impressions and weights each user by positive count.
- Each user has about 5 logged impressions in the evaluation split. Recall@50 is meaningless here and is not scored.

## Three zones
FROZEN for the research agent (built under my instructions; `python prepare.py --verify` checks their hashes):
- starter_kit/**: organizer files, vendored byte for byte.
- prepare.py: loads splits through starter_kit data.load(), builds caches, exposes evaluate and submission helpers, strips test labels.
- harness/**: the loop runner.
- tests/**: parity, guards, features, models, harness and fault-injection tests.
MUTABLE (the research agent's whole surface):
- train.py: composes features + model + loss + training. Must satisfy the contract below.
- recsys/**: the environment library (features, models, losses, blending, eda).
HUMAN-OWNED (edited by me between runs, never by the research agent):
- program.md, CLAUDE.md, IMPLEMENTATION.md, README.md, docs/**, reports/**.

## Data policy (non-negotiable)
1. Test labels never enter memory. prepare.load("test") returns features only; every feedback column is dropped before return. A test asserts this.
2. No label from any row dated after HISTORY_END may be used to fit anything. Default HISTORY_END = 20220421 (train end). This excludes log_random_*.csv entirely by default, because that file lives in the validation and test date range.
3. No cross-row aggregation inside an evaluation split (no test-time popularity, no counting duplicates across rows). Features for a row may use: the training-window logs, the static user and video feature files shipped with KuaiRand, and the row's own columns.
4. Only KuaiRand files feed KuaiRand models. No external data, no pretrained weights that saw these benchmarks' test labels.
5. Never reimplement the metric. Call starter_kit.evaluate through prepare.evaluate. The harness recomputes metrics from the saved score arrays and ignores anything train.py prints.

## train.py contract
`python train.py --out <dir> [--seed S] [--time-budget SEC] [--model NAME] [--features NAME]`
Writes to <dir>: val_scores.npy (float32, length 124,909, data.load() order), test_scores.npy (float32, length 170,588, same rule), config.json (everything needed to re-run: model, feature spec, hyperparameters, number of rounds or epochs actually used, seed, subsampling), and exits 0. Respects --time-budget as training wall-clock (default 300 s); the harness hard-kills at 2x budget + 120 s. Deterministic for a fixed seed on CPU.

## What an iteration is
An iteration is one scored experiment: one hypothesis, one change, one train.py run, one validation score. A failed attempt (crash, timeout, bad output) does not consume an iteration number; the same iteration may be retried up to 3 attempts, after which it is abandoned. Abandoned iterations count toward the 50 cap but not toward the convergence window, because they produced no score. This interpretation is written into reports/ so the judges can see it.

## Iteration ledger
- results.tsv: tab-separated, tracked by git, one row per iteration: iter, commit, status (keep | revert | abandoned), primary, gauc, ndcg5, delta_vs_baseline, train_sec, seed, description. Never edited by hand.
- runs/<run_id>/iterations/<iter>/: hypothesis.md (written by the agent before running), diff.patch, metrics.json, events.jsonl, config.json, stdout.log. The harness writes everything except hypothesis.md.
- runs/<run_id>/notes.md: the agent's reflection after each iteration. runs/<run_id>/interventions.jsonl: every human touch, logged by me.

## Coding conventions
Python 3.11, uv, ruff. CPU-first; torch models auto-detect CUDA or MPS but never require it. Redirect long outputs to files; never let training logs flood the context. One idea per commit. Commit message format during research runs: `iter <n>: <short description> (primary=<x>)`.

## Ground truth precedence
starter_kit/evaluate.py > starter_kit/data.py > this file > program.md > the problem statement. If two of these disagree, stop and report the discrepancy to me instead of guessing.

## Appendix B: program.md (written verbatim in Phase 4)

# program.md: KuaiRand-Pure autoresearch

You are an autonomous ML research agent. Your job is to run the loop (read the problem, inspect data, engineer features, train and tune, evaluate, reflect and revise) on the KuaiRand-Pure ranking benchmark and push the validation primary score (mean of GAUC and nDCG@5) as high as possible before the harness stops you. I am not available during the run. You never ask me anything; you decide, act, and log.

## Setup
1. Pick a run tag from today's date (for example sep03). Run `python -m harness start --run-id <tag>`. It creates the branch, verifies the frozen layer and prints the next command. If it fails, fix only things outside the frozen layer. If the frozen layer itself is broken, write the diagnosis to runs/<tag>/BLOCKED.md and stop; that is the one legitimate early stop.
2. Read, in this order: CLAUDE.md, reports/eda.md, recsys/features/README.md, reports/zoo_baselines.md, results.tsv (previous runs), the latest runs/*/notes.md if any, then train.py.
3. Iteration 1 is always the unmodified train.py so the ledger starts at the reproduced baseline. Its hypothesis.md says "baseline reproduction".

## What you can change
- train.py and anything under recsys/: features, model, loss, sampling, training schedule, ensembling, post-processing of scores. Every stage of the loop is fair game, not only the model.
- Dependencies: only what is already in pyproject.toml. If an idea needs a new package, write it in notes.md as deferred and move on.

## What you cannot change or do
- starter_kit/, prepare.py, harness/, tests/, this file.
- Read data/raw/ directly, read log_random, use any label dated after HISTORY_END, aggregate anything across rows of the validation or test split, or reimplement the metric.
- Run more than one experiment per iteration, or run an experiment longer than the time budget. If training will not fit, subsample or stop early; do not raise the budget.
- Ask me to continue, confirm or choose. If you are stuck, write it in notes.md and pick the next hypothesis.
- Edit results.tsv or the harness. If you find a harness bug, log it in runs/<tag>/HARNESS_BUGS.md, route around it, continue.

## The loop
The harness numbers iterations and does the git work. Repeat until iterate prints STOP:
1. Reflect. Read the summary block of the last iteration and notes.md. Choose the single most valuable thing to test next by expected gain per minute, risk of a dud, and what the last results taught you.
2. Write runs/<tag>/iterations/<i>/hypothesis.md, 5 to 12 lines: which stage this targets (data, features, model, loss, training, evaluation, ensembling), what exactly changes, why it should raise primary (mechanism, not hope), expected delta, source if any (paper, public solution, textbook chapter), rollback plan.
3. Edit the mutable surface. Small legible diffs. One idea per iteration.
4. `python -m harness iterate --desc "<12 words max>" > iterate.log 2>&1` then `tail -n 40 iterate.log`. Never read stdout.log in full; the harness has already redirected training output there.
5. The block says KEEP or REVERT and the harness has already committed or reverted. Confirm with `git status` that the tree is clean.
6. If the block shows an error: read the last 40 lines of that iteration's stdout.log, fix the cause if it is mechanical (typo, shape, dtype, missing key, import) and run iterate again for the same iteration (attempt 2, then 3). If the idea itself is broken, `python -m harness abandon --reason "<why>"` and move on. Never spend more than 3 attempts on one iteration.
7. Append 2 to 4 lines to runs/<tag>/notes.md: what was learned, what it rules out, the next two candidates.
8. When iterate prints STOP (converged, iteration cap or wall-clock), run `python -m harness finish` then `python -m harness report --run-id <tag>`. Then stop.

## Decision rules
- KEEP means the validation primary beat the best so far. The harness decides; you do not override it.
- Noise floor is about 0.001 (the baseline's seed std is 0.0008). If a candidate lands within 0.002 of the best either way and the change is cheap, re-run it with `--seeds 0,1,2` before deciding your next move, but only while the run has at least 10 iterations of headroom.
- The convergence rule ends the run after three consecutive scored iterations without a gain above 0.002 over the previous best. Order experiments so high-probability gains come first and speculative ones come after you have banked a margin. Do not try to game the rule with artificial micro-gains; the judges read the logs.
- Simplicity criterion, as in autoresearch: an improvement that adds ugly complexity is worth less than its number suggests; a simplification that keeps the score is a win. Weigh complexity against delta before choosing.
- Judge progress against the 0.8645 ceiling, not 1.0. The baseline holds about 31% of the attainable range.

## Where the score lives (read before choosing hypotheses)
- Ranking is within each user's impressions over the whole split, about 5 per user. User-level main effects cancel. Item, context and user x item interaction signal is what moves GAUC and nDCG@5. Features constant within a user only help through interactions or a tree model.
- 27.1% of test users have no positive (nDCG fixed at 0) and 9.2% are all-positive (nDCG fixed at 1, excluded from GAUC). Only mixed users move the score. Per-user losses that focus on them (lambdarank, pairwise or listwise within user) are metric-aligned; pointwise BCE is not.
- long_view is a function of play_time and duration, and duration is known at impression time. Watch-time and watch-ratio modelling (multi-task heads, censored regression as in the CWM reference) are the most direct auxiliary signals. Look at reports/eda.md for how the label depends on duration before designing anything.
- The split is temporal. Recency of item statistics matters; test the decayed hist_item variant early.
- 3.06% of eval rows are repeated (user, video) pairs. Ties between duplicates are fine. Do not break them with anything that reads across rows of the split.

## Seed ideas, roughly ordered by expected value per minute
1. lgbm_lambdarank on the full spec, tuned: num_leaves, min_data_in_leaf, feature_fraction, learning rate, early stopping on validation primary through prepare.evaluate.
2. Decayed hist_item rates and duration-bucket cross features; switch the item_stats sub-block off (spec full_nostats) and compare.
3. Multi-task heads (mmoe, ple) with watch-ratio regression; score = long_view head, then a validated combination of heads.
4. cwm censored watch-time model, rank-averaged with lambdarank.
5. din_lite target attention over the last-20 history.
6. Pairwise or listwise losses for the torch models.
7. Linear stacker over kept models' out-of-fold predictions.
8. Training-window weighting toward the last 3 to 5 days; reweighting of watched-but-short impressions as hard negatives.
9. Textbook leads if you run dry (Aggarwal, Recommender Systems: The Textbook): learning to rank 13.2, factorization machines 8.5.2, implicit feedback in latent factor models 3.6.4.6, ensembles and hybrids chapter 6, recency and decay 9.2.1, ranking metrics 7.5.3 and 7.5.4.

## Logging
The harness writes results.tsv, diff.patch, metrics.json, events.jsonl, config.json, stdout.log, and the commits. You write hypothesis.md before every iterate call and notes.md after. These logs are what the judges score for autonomy and robustness, so write hypotheses that a reviewer can follow.

## Never stop early
After setup you run until the harness prints STOP. Not "this seems like a good place to pause", not "should I continue", not "I have run out of ideas". If you run out of ideas: re-read reports/eda.md and the seed list, look at per-user errors of the best checkpoint (checkpoints/<tag>/best/val_scores.npy against the validation labels through prepare.load), combine earlier near-misses, or try a bolder change. The only reasons a run ends are the three stop conditions and a BLOCKED.md on a broken frozen layer.

## Appendix C: kickoff message for a scored run (not part of the build)

Read program.md and run it end to end for KuaiRand-Pure. Do the setup, then iterate until the harness prints STOP, then finish and report. Every human message after this one counts as a manual intervention, so I will not send any.

## Master checklist

Tick only after the gate command passed. Commit the tick with the deliverable.

Phase 0: scaffold and constitution
- [x] pyproject.toml, .python-version, .gitignore, Makefile; `uv sync` and `ruff check .` pass
- [x] CLAUDE.md written verbatim from Appendix A
- [x] docs/autoresearch/ vendored (or MISSING.md noted)
- [x] reports/decisions.md started; inputs from section 1 verified
- [x] pushed, tagged phase-0

Phase 1: frozen layer, parity, guards, EDA
- [x] starter_kit/ vendored unchanged, MANIFEST.sha256, NOTES.md
- [x] prepare.py: load, tables, evaluate, write_submission, --build, --verify (idempotent)
- [x] tests/fixtures/make_fixture.py produces data/cache/fixture_small/
- [x] tests/test_parity.py green (baseline within 0.003 of 0.6016 / 0.6674 / 0.5357; random and popularity rungs in range; toy example)
- [x] tests/test_guards.py green
- [x] reports/eda.md generated
- [x] reports/phase1.md written; pushed, tagged phase-1

Phase 2: feature layer
- [x] spec.py with cache; specs fm5, full, full_nostats
- [x] blocks ctx, item_static (+ item_stats), user_static, hist_user, hist_item (+ decayed), cross, seq, target_enc
- [x] tests/test_features.py green (time safety, brute-force equality, no labels, group order, under 60 s, deterministic)
- [x] recsys/features/README.md written
- [x] reports/phase2.md written; pushed, tagged phase-2

Phase 3: model zoo and train.py
- [x] base.py, registry, losses.py
- [x] rungs: random, popularity, fm (parity), lgbm_pointwise, lgbm_lambdarank, deepfm, dcnv2, mmoe, ple, cwm, din_lite, blend
- [x] recsys/zoo.py list and bench; reports/zoo_baselines.md
- [x] train.py meets the Appendix A contract with defaults fm / fm5
- [x] tests/test_models.py and tests/test_train.py green
- [x] zoo gate: a non-fm rung beats fm by 0.005 or more on validation (blend 0.6071 = fm + 0.0055)
- [x] reports/phase3.md written; pushed, tagged phase-3

Phase 4: autoresearch loop
- [x] harness start, iterate, abandon, revert, status, intervene, finish
- [x] watchdog (timeout, RSS cap), metrics recomputed by the harness, KEEP writes a checked submission
- [x] git ops: one commit per iteration, clean tree after iterate
- [x] convergence.py (wrapping the starter kit rule if it ships) plus cap and ceiling; divergence guard
- [x] schemas for results.tsv, metrics.json, events.jsonl, run.json, resources.json
- [x] program.md written verbatim from Appendix B
- [x] tests/test_harness.py green
- [x] smoke run: one iteration matches the fm rung, final.csv passes --check, smoke branch deleted
- [x] reports/phase4.md written; pushed, tagged phase-4

Phase 5: scripted dry run and fault injection
- [x] train_stub.py with the QUALITY knob; driver runs in a temporary clone
- [x] `python -m tests.scripted_agent.run --fixture small` exits 0 with CONVERGED at iteration 8 and all assertions matched (34/34)
- [x] `--fixture full` mode implemented (rehearsed: 11/11, exit 0)
- [x] tests/test_fault_injection.py green
- [x] reports/phase5.md written; pushed, tagged phase-5

Phase 6: deliverables and bonus datasets
- [x] harness report writes results_table.md, resources.md, interventions.md, iteration_log.md, trajectory.png
- [x] README.md with all required sections; docs/devpost.md skeleton
- [x] prepare.py --dataset 1k builds and verifies; lambdarank bench passes on 1k (0.6573; 1k downloaded from the organizer Zenodo record)
- [x] 27k loader with documented sampling policy (not run end to end; 27k raw files not downloaded — 1k is green so the precondition holds if ever needed)
- [x] reports/phase6.md written; pushed, tagged phase-6

Phase 7: final validation and release
- [ ] full go/no-go list green, outputs in reports/final.md
- [ ] main clean, no autoresearch/* branches, tagged v1.0, pushed
