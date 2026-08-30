# Phase 7: final validation (go/no-go)

Every command from the IMPLEMENTATION.md phase-7 list, with its real output, run on main
at the release candidate.

```
=== python prepare.py --verify
parity: fm 0.6015 | random 0.4827 | popularity 0.5807
verify OK (pure @ D:\milk\data)
EXIT=0

=== pytest -q
........................................                                 [100%]
40 passed in 245.60s (0:04:05)

=== python -m tests.scripted_agent.run --fixture small
34/34 checks passed, exit 0
(workspace C:\Users\rahul\AppData\Local\Temp\scripted-20260830-201819)

=== python -m tests.scripted_agent.run --fixture full
11/11 checks passed, exit 0
(workspace C:\Users\rahul\AppData\Local\Temp\scripted-20260830-201919)

=== python -m recsys.zoo bench --budget 300
random: primary 0.4827 gauc 0.4990 ndcg5 0.4663 (0s, 697 MB)
popularity: primary 0.5807 gauc 0.6387 ndcg5 0.5227 (0s, 1810 MB)
fm: primary 0.6016 gauc 0.6674 ndcg5 0.5358 (31s, 952 MB)
lgbm_pointwise: primary 0.6012 gauc 0.6670 ndcg5 0.5354 (151s, 2385 MB)
lgbm_lambdarank: primary 0.5992 gauc 0.6642 ndcg5 0.5341 (122s, 2553 MB)
deepfm: primary 0.6057 gauc 0.6728 ndcg5 0.5386 (112s, 2999 MB)
dcnv2: primary 0.6043 gauc 0.6717 ndcg5 0.5370 (240s, 3326 MB)
mmoe: primary 0.6051 gauc 0.6724 ndcg5 0.5378 (236s, 3664 MB)
ple: primary 0.6062 gauc 0.6733 ndcg5 0.5392 (318s, 3398 MB)
cwm: primary 0.5625 gauc 0.6132 ndcg5 0.5118 (66s, 3342 MB)
din_lite: primary 0.6048 gauc 0.6721 ndcg5 0.5376 (306s, 5970 MB)
blend: primary 0.6071 gauc 0.6743 ndcg5 0.5398 (310s, 3375 MB)
EXIT=0
  -> every rung reproduced its previous number exactly (seed-deterministic),
     zoo gate still holds: blend 0.6071 = fm + 0.0055 >= +0.005

=== git status (before the final-report commit)
 M reports/zoo_baselines.md      (the bench rows above)
?? reports/final.md              (this file)
autoresearch/* branches: none
```

## One defect found and fixed during this list

The first scripted-small pass scored 30/34: phase 6 commits the scripted dry run's ledger
under runs/scripted, so the driver's fresh clone inherited the old per-iteration
events.jsonl files and the appended error/timeout/nan/stop counts doubled. Fix (committed
as `phase7: scripted driver resets a committed prior ledger in its clone`): the driver now
`git rm`s a pre-existing runs/scripted in its clone before starting. Both modes were
re-run after the fix — 34/34 and 11/11 with exit 0, as recorded above.

## Verdict

GO. Main is clean after this commit, no autoresearch/* branches remain, tagged v1.0.
The repo is ready for a scored run, which starts with the kickoff message in
IMPLEMENTATION.md Appendix C and nothing else.
