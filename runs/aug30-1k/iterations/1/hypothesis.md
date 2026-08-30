# Iteration 1: baseline reproduction (KuaiRand-1k)

- Stage: none (control).
- Change: none — unmodified train.py (model fm, spec fm5, seed 0), dataset selected
  by KUAIRAND_DATASET=1k in the calling shell (train.py env plumbing, commit
  0c8e46f's parent). 900 s budget.
- Why: no official 1k baseline is published, so this run's floor is the same FM
  architecture on the 1k split — every later delta is measured against a number this
  harness produced. Note: the frozen harness prints delta_vs_baseline against the
  Pure baseline (0.6016) because baseline_scores.json only ships Pure numbers; that
  column is cosmetic here, deltas that matter are vs this iteration's score.
- Expected: unknown a priori; 1k logs are denser per user (~1k users, 5.06M train
  rows), zoo's lgbm_lambdarank reached 0.6573, so FM plausibly lands 0.55-0.62.
- Rollback: nothing to roll back.
