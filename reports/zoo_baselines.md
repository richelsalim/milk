# Zoo baselines (validation)

| model | primary | gauc | ndcg5 | train_sec | peak_rss_mb | spec | notes |
|---|---|---|---|---|---|---|---|
| random | 0.4827 | 0.4990 | 0.4663 | 0 | 1913 | fm5 |  |
| popularity | 0.5807 | 0.6387 | 0.5227 | 0 | 2009 | full |  |
| fm | 0.6016 | 0.6674 | 0.5358 | 33 | 863 | fm5 |  |
| lgbm_pointwise | 0.6012 | 0.6670 | 0.5354 | 154 | 2199 | full |  |
| lgbm_lambdarank | 0.5992 | 0.6642 | 0.5341 | 128 | 2328 | full |  |
