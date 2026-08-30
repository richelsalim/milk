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
