# Iteration 3: prune ple's rare-label aux heads, spend the savings on a 4th epoch

- Stage: model (multi-task structure) + training schedule.
- Change: in the registry blend, the ple base's cfg gains
  aux_tasks=["is_click","is_like"] (drops is_follow/is_comment/is_forward/is_hate
  heads and their own-experts/gates/towers) and epochs 3 -> 4. deepfm base untouched.
- Why it should raise primary: the dropped labels are extremely rare in train
  (follow/forward/hate well under 1%), so their heads contribute mostly noise
  gradients and task interference while costing ~4/8 of the tower/expert compute.
  A leaner ple epoch (~38 s vs ~47 s) lets 4 full-data epochs + snapshot top-3 fit
  the same 165 s share — one more epoch of the main task usually beats four dead
  auxiliary towers (v1 evidence: ple best epochs are 4-6 at full budget).
- Expected delta: +0.0005 to +0.0015 on the blend; risk: losing multi-task
  regularization from the rare heads could cost up to -0.001.
- Source: reports/phase3.md multi-task findings; PLE paper's task-interference
  argument (Tang et al. 2020).
- Rollback: harness reverts the registry edit if the blend scores below 0.6075.
