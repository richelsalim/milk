# Phase 1 report: frozen layer, baseline parity, guards, EDA

## Baseline parity — reproduced vs published (validation)

| metric | reproduced (seed 0) | published | delta |
|---|---|---|---|
| GAUC | 0.6671 | 0.6674 | -0.0003 |
| nDCG@5 | 0.5358 | 0.5357 | +0.0001 |
| **primary** | **0.6015** | **0.6016** | **-0.0001** |

Rungs: random 0.4827 (published 0.4834), popularity 0.5807 (published 0.5807; kit smoothed-rate
formula — see reports/decisions.md for why the literal count variant was replaced).

## Gate commands and real output

```
$ uv run python prepare.py --build
built cache at D:\milk\data\cache: sizes {'train': 1141112, 'val': 124909, 'test': 170588}

$ uv run python prepare.py --verify   (first run computes parity, later runs are read-only)
parity: fm 0.6015 | random 0.4827 | popularity 0.5807
verify OK (pure @ D:\milk\data)
EXIT=0

$ uv run python tests/fixtures/make_fixture.py
built cache at D:\milk\data\cache\fixture_small\cache: sizes {'train': 15753, 'val': 1877, 'test': 2528}
fixture: 315 users, 20158 log rows -> D:\milk\data\cache\fixture_small

$ uv run pytest -q tests/test_parity.py tests/test_guards.py
.......                                                                  [100%]
7 passed in 62.74s (0:01:02)

$ test -f reports/eda.md && echo OK
OK

second --verify run: IDEMPOTENT (no cache file changed; checked by hashing data/cache before/after)
```

## Notes
- Raw data was absent at build start; downloaded from the organizer-referenced Zenodo record
  (starter kit README) and verified against the pinned split sizes — exact match.
- EDA highlights (reports/eda.md): tab dominates label rate (0.004–0.61 across tabs);
  median watch_ratio 0.032 (long_view=0) vs 0.978 (long_view=1); val users 30.3% zero-positive,
  11.9% all-positive; 98.1% user / 99.9% item overlap between train and validation.
