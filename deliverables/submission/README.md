# Final submission — KuaiRand-Pure (Deliverable 4)

- `final.csv` — hidden-test submission in the starter-kit schema
  (`row_id,user_id,video_id,score`, 170,588 rows), produced by the repo's current
  validation-best configuration (**blend**: weighted rank-average of DeepFM + PLE,
  validation primary 0.6071 = official baseline + 0.0055) and validated by
  `starter_kit/submit.py --check` at write time.
- `config.json` — everything needed to re-run it (model, feature spec, per-base
  configs and budget shares, epochs actually used, seed, timing).
- `model.pkl` — the trained checkpoint itself (8.5 MB; load with
  `recsys.models.Recommender.load` and score any split via its `predict`).

Regenerate from a clean checkout (CPU-only, ~6 min):

```bash
uv run python prepare.py --build
uv run python train.py --out out_blend --model blend --time-budget 300
uv run python - <<'EOF'
import numpy as np, prepare
prepare.write_submission("test", np.load("out_blend/test_scores.npy"),
                         "deliverables/submission/final.csv")  # runs submit.py --check
EOF
```

The training is seed-deterministic, so the command above rebuilds the identical
checkpoint and CSV. When the scored autonomous run (IMPLEMENTATION_V2.md, phase V2.3)
completes, its `submissions/<tag>/final.csv` replaces this file.
