# aug30: iteration log

## Iteration 1 — keep/revert: **keep**, primary 0.6016 (delta vs baseline +0.0000)

**Hypothesis**

> # Iteration 1: baseline reproduction
> 
> - Stage: none (control).
> - Change: none — unmodified train.py (model fm, spec fm5, seed 0, 300 s budget).
> - Why: the ledger starts at the reproduced official Factorization Machine baseline so
>   every later delta is measured against a number this harness produced, not a quoted
>   one. program.md setup step 3.
> - Expected: primary ~0.6015 on validation (published 0.6016, seed std 0.0008).
> - Rollback: nothing to roll back.

Diff: 0 lines ([diff.patch](../../runs/aug30/iterations/1/diff.patch))

Metrics: GAUC 0.6674, nDCG@5 0.5358, primary 0.6016, train 46s, peak RSS 824 MB, seeds [0]

## Iteration 2 — keep/revert: **keep**, primary 0.6075 (delta vs baseline +0.0059)

**Hypothesis**

> # Iteration 2: switch to the diverse snapshot blend
> 
> - Stage: model + ensembling.
> - Change: train.py --model default fm -> blend. The registry blend is a weighted
>   rank-average of deepfm (share .45, epochs<=6) and ple (share .55, epochs=3), both
>   full-data with top-3 snapshot ensembling from cached per-epoch val predictions,
>   plus a 6-point weight grid on validation. All stops are label-driven (patience or
>   epoch cap), so the result is machine-speed independent.
> - Why it should raise primary: within-user ranking rewards item/context/interaction
>   signal that embedding models capture and trees/FM under-use; deepfm and ple make
>   decorrelated errors (architecture + objective differ) and rank-averaging keeps
>   each model's within-user order where it is confident. Snapshot averaging adds a
>   cheap variance reduction on top (zoo: deepfm 0.6057->0.6066, ple 0.6062->0.6065).
> - Expected delta: +0.006 vs baseline (zoo bench: 0.6075, reproduced twice bit-equal).
> - Source: reports/zoo_baselines.md + reports/v2.md phase V2.2.
> - Rollback: harness reverts train.py to the kept commit on a worse score.

Diff: 13 lines ([diff.patch](../../runs/aug30/iterations/2/diff.patch))

Metrics: GAUC 0.6751, nDCG@5 0.5399, primary 0.6075, train 287s, peak RSS 2878 MB, seeds [0]

## Iteration 3 — keep/revert: **revert**, primary 0.6067 (delta vs baseline +0.0051)

**Hypothesis**

> # Iteration 3: prune ple's rare-label aux heads, spend the savings on a 4th epoch
> 
> - Stage: model (multi-task structure) + training schedule.
> - Change: in the registry blend, the ple base's cfg gains
>   aux_tasks=["is_click","is_like"] (drops is_follow/is_comment/is_forward/is_hate
>   heads and their own-experts/gates/towers) and epochs 3 -> 4. deepfm base untouched.
> - Why it should raise primary: the dropped labels are extremely rare in train
>   (follow/forward/hate well under 1%), so their heads contribute mostly noise
>   gradients and task interference while costing ~4/8 of the tower/expert compute.
>   A leaner ple epoch (~38 s vs ~47 s) lets 4 full-data epochs + snapshot top-3 fit
>   the same 165 s share — one more epoch of the main task usually beats four dead
>   auxiliary towers (v1 evidence: ple best epochs are 4-6 at full budget).
> - Expected delta: +0.0005 to +0.0015 on the blend; risk: losing multi-task
>   regularization from the rare heads could cost up to -0.001.
> - Source: reports/phase3.md multi-task findings; PLE paper's task-interference
>   argument (Tang et al. 2020).
> - Rollback: harness reverts the registry edit if the blend scores below 0.6075.

Diff: 15 lines ([diff.patch](../../runs/aug30/iterations/3/diff.patch))

Metrics: GAUC 0.6743, nDCG@5 0.5391, primary 0.6067, train 268s, peak RSS 2904 MB, seeds [0]

## Iteration 4 — keep/revert: **revert**, primary 0.6063 (delta vs baseline +0.0047)

**Hypothesis**

> # Iteration 4: deepfm embedding dim 16 -> 24 inside its blend share
> 
> - Stage: model capacity.
> - Change: registry blend's deepfm base cfg gains dim=24 and epochs<=5 (its natural
>   patience stop was 5 of 6, so the cap only re-pins label-driven stopping while the
>   wider epoch (~26 s vs ~19 s) still fits the 135 s share: 5 x 26 + evals ~ 132 s).
>   ple base untouched.
> - Why it should raise primary: within-user ranking lives on user x item interaction
>   terms; the FM component's expressiveness scales with embedding dim, and dim 16 was
>   inherited from the official baseline, never tuned (v1/v2 swept data size, losses,
>   blends — not width). deepfm is the budget-cheap base, so this is the only place
>   capacity is free.
> - Expected delta: +0.0005 to +0.0015 on the blend; risk: wider embeddings on 1.14M
>   rows can overfit day-tail noise (-0.001), mitigated by snapshot top-3 averaging.
> - Source: FM capacity discussion, Aggarwal 8.5.2; reports/decisions.md v1 zoo notes.
> - Rollback: harness revert on a sub-0.6075 score.

Diff: 13 lines ([diff.patch](../../runs/aug30/iterations/4/diff.patch))

Metrics: GAUC 0.6736, nDCG@5 0.5390, primary 0.6063, train 317s, peak RSS 2884 MB, seeds [0]

## Iteration 5 — keep/revert: **revert**, primary 0.6059 (delta vs baseline +0.0043)

**Hypothesis**

> # Iteration 5: video x tab historical-rate cross feature
> 
> - Stage: features.
> - Change: recsys/features/blocks.py cross block gains a (video_id, tab) family —
>   x_vt_impressions and smoothed x_vt_lv_rate (prior 20, train-window gmean), same
>   strict-past/train-window contract as every cross. SPECS["full"] version 1 -> 2
>   (cache key bump; other specs untouched this iteration). Model config unchanged
>   (registry blend, both bases as in iter 2).
> - Why it should raise primary: eda shows tab is the strongest context main effect in
>   the data (long_view rate 0.004 on tab 3 vs 0.489 on tab 4), and the same video
>   surfaces on different tabs with different intents. hi_vid_lv_rate pools across
>   tabs, so a video's per-tab rate is new item-side information that varies within a
>   user's list (different videos, sometimes different tabs) — exactly the kind of
>   signal within-user ranking can use. Both embedding bases consume it as a dense
>   column.
> - Expected delta: +0.0005 to +0.002; risk: sparsity of (video, tab) cells adds noise
>   the prior may not fully absorb (-0.0005).
> - Source: reports/eda.md tab table; cross-feature precedent x_vd (video x
>   user_active_degree) from v1.
> - Rollback: harness revert restores blocks.py and spec.py.

Diff: 30 lines ([diff.patch](../../runs/aug30/iterations/5/diff.patch))

Metrics: GAUC 0.6729, nDCG@5 0.5390, primary 0.6059, train 322s, peak RSS 3837 MB, seeds [0]

Events:
- `stop` (attempt 1): converged -> stop

