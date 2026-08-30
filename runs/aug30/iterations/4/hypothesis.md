# Iteration 4: deepfm embedding dim 16 -> 24 inside its blend share

- Stage: model capacity.
- Change: registry blend's deepfm base cfg gains dim=24 and epochs<=5 (its natural
  patience stop was 5 of 6, so the cap only re-pins label-driven stopping while the
  wider epoch (~26 s vs ~19 s) still fits the 135 s share: 5 x 26 + evals ~ 132 s).
  ple base untouched.
- Why it should raise primary: within-user ranking lives on user x item interaction
  terms; the FM component's expressiveness scales with embedding dim, and dim 16 was
  inherited from the official baseline, never tuned (v1/v2 swept data size, losses,
  blends — not width). deepfm is the budget-cheap base, so this is the only place
  capacity is free.
- Expected delta: +0.0005 to +0.0015 on the blend; risk: wider embeddings on 1.14M
  rows can overfit day-tail noise (-0.001), mitigated by snapshot top-3 averaging.
- Source: FM capacity discussion, Aggarwal 8.5.2; reports/decisions.md v1 zoo notes.
- Rollback: harness revert on a sub-0.6075 score.
