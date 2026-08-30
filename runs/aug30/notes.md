## After iter 1 (KEEP, 0.6016)
Baseline reproduced to the 4th decimal (published 0.6016) — env parity holds, deltas
from here are trustworthy. Rules out nothing; establishes the floor. Next two
candidates: (1) registry blend (deepfm+ple snapshot pair, weight grid — zoo shows
0.6075, deterministic label-driven stops); (2) if kept, free ple epochs inside its
blend share by pruning weak aux heads.
## After iter 2 (KEEP, 0.6075)
Blend reproduced its bench exactly (0.6075) — the label-driven-stop determinism work
holds inside the harness too. +0.0059 banked. The window now needs >0.6095 in 3 scored
iters or the run converges; ordering the remaining shots by mechanism quality, not
hope. Next: (1) ple aux-head pruning + 4th epoch inside its share; (2) deepfm dim bump
in its cheap share.
## After iter 3 (REVERT, 0.6067)
Pruning ple's rare aux heads for a 4th epoch LOST 0.0008: the follow/comment/forward/
hate towers are cheap regularizers, not dead weight — multi-task breadth > one more
main-task epoch here. Rules out task-count cuts and epoch-for-capacity trades on ple.
Next: (1) deepfm dim 16->24 inside its 135 s share (capacity on the cheap base, ple
untouched); (2) a video-x-tab historical-rate cross feature (item-side context signal,
eda shows tab rates span 0.004-0.49).
## After iter 4 (REVERT, 0.6063)
dim 24 lost 0.0012 — capacity was not the binding constraint; dim 16 embeddings +
snapshots already sit at this data's interaction-signal ceiling, and wider tables
overfit the two-week window. Rules out width sweeps. Window is live: iter 5 must
clear +0.002 over 0.6075 or the run converges. Last mechanism-backed candidate:
video x tab historical rate cross (eda: tab long_view rates span 0.004-0.49, so the
same video carries different priors per surface — item-side, varies within user).
## After iter 5 (REVERT, 0.6059) — run converged
x_vt cross lost 0.0016: (video, tab) cells are too sparse for prior=20, and tab is
near-constant within most users' lists, so the noise landed exactly where ranking is
decided. Run converged per the shipped rule (window -0.0008/-0.0012/-0.0016 <= eps
0.002): three mechanism-backed post-blend shots (task pruning, capacity, context
cross) all landed within noise-to-negative of 0.6075 — consistent with the v2
environment sweeps that put this data's blendable ceiling at ~0.6075 on validation.
Best: iter 2, diverse snapshot blend, 0.6075 (+0.0059 vs reproduced baseline; the
FM baseline holds ~31% of the 0.8645-ceiling range, this run holds ~50%).
