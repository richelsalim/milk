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
