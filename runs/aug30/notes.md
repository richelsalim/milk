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
