# Final submission — KuaiRand-Pure (Deliverable 4)

- `final.csv` — hidden-test submission in the starter-kit schema
  (`row_id,user_id,video_id,score`, 170,588 rows), written by the harness from run
  `aug30`'s kept-best checkpoint (**blend**: weighted rank-average of DeepFM + PLE
  with top-3 snapshot ensembling and a validation weight grid; validation primary
  **0.6075** = official baseline + 0.0059) and re-validated by
  `starter_kit/submit.py --check` at `harness finish`.
- `config.json` — the run's best-iteration config: model, feature spec, per-base
  configs and budget shares, epochs actually used, seed, timing.

The trained checkpoint (`model.pkl`, ~20 MB with snapshot state dicts) lives
untracked at `checkpoints/aug30/best/`; training is seed-deterministic with
label-driven stops, so a clean checkout rebuilds the identical checkpoint and CSV
(CPU-only, ~5 min):

```bash
uv run python prepare.py --build
uv run python train.py --out out_blend --model blend --time-budget 300
uv run python - <<'EOF'
import numpy as np, prepare
prepare.write_submission("test", np.load("out_blend/test_scores.npy"),
                         "deliverables/submission/final.csv")  # runs submit.py --check
EOF
```

Provenance: run ledger `runs/aug30/` (branch `autoresearch/aug30`, one commit per
iteration), rendered report `reports/aug30/`.
