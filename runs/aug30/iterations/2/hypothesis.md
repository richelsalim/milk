# Iteration 2: switch to the diverse snapshot blend

- Stage: model + ensembling.
- Change: train.py --model default fm -> blend. The registry blend is a weighted
  rank-average of deepfm (share .45, epochs<=6) and ple (share .55, epochs=3), both
  full-data with top-3 snapshot ensembling from cached per-epoch val predictions,
  plus a 6-point weight grid on validation. All stops are label-driven (patience or
  epoch cap), so the result is machine-speed independent.
- Why it should raise primary: within-user ranking rewards item/context/interaction
  signal that embedding models capture and trees/FM under-use; deepfm and ple make
  decorrelated errors (architecture + objective differ) and rank-averaging keeps
  each model's within-user order where it is confident. Snapshot averaging adds a
  cheap variance reduction on top (zoo: deepfm 0.6057->0.6066, ple 0.6062->0.6065).
- Expected delta: +0.006 vs baseline (zoo bench: 0.6075, reproduced twice bit-equal).
- Source: reports/zoo_baselines.md + reports/v2.md phase V2.2.
- Rollback: harness reverts train.py to the kept commit on a worse score.
