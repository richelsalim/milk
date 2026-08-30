# Iteration 5: video x tab historical-rate cross feature

- Stage: features.
- Change: recsys/features/blocks.py cross block gains a (video_id, tab) family —
  x_vt_impressions and smoothed x_vt_lv_rate (prior 20, train-window gmean), same
  strict-past/train-window contract as every cross. SPECS["full"] version 1 -> 2
  (cache key bump; other specs untouched this iteration). Model config unchanged
  (registry blend, both bases as in iter 2).
- Why it should raise primary: eda shows tab is the strongest context main effect in
  the data (long_view rate 0.004 on tab 3 vs 0.489 on tab 4), and the same video
  surfaces on different tabs with different intents. hi_vid_lv_rate pools across
  tabs, so a video's per-tab rate is new item-side information that varies within a
  user's list (different videos, sometimes different tabs) — exactly the kind of
  signal within-user ranking can use. Both embedding bases consume it as a dense
  column.
- Expected delta: +0.0005 to +0.002; risk: sparsity of (video, tab) cells adds noise
  the prior may not fully absorb (-0.0005).
- Source: reports/eda.md tab table; cross-feature precedent x_vd (video x
  user_active_degree) from v1.
- Rollback: harness revert restores blocks.py and spec.py.
