# EDA: KuaiRand-Pure

Overall long_view rate: train 0.3366, val 0.3133

## Label rate by duration decile (train; edges from train quantiles)

| dur_decile | rows | long_view rate |
|---|---|---|
| 0 | 114,203 | 0.2810 |
| 1 | 114,540 | 0.2730 |
| 2 | 113,966 | 0.3479 |
| 3 | 113,867 | 0.3670 |
| 4 | 114,093 | 0.3629 |
| 5 | 114,020 | 0.3592 |
| 6 | 114,147 | 0.3760 |
| 7 | 114,364 | 0.3445 |
| 8 | 116,975 | 0.3367 |
| 9 | 110,937 | 0.3179 |

## Label rate by tab (train)

| tab | rows | long_view rate |
|---|---|---|
| 0 | 150,013 | 0.0422 |
| 1 | 834,876 | 0.3861 |
| 2 | 39,291 | 0.3805 |
| 3 | 3,574 | 0.0042 |
| 4 | 75,524 | 0.4893 |
| 5 | 3,402 | 0.1699 |
| 6 | 29,671 | 0.0870 |
| 7 | 333 | 0.2132 |
| 8 | 2,551 | 0.0176 |
| 9 | 252 | 0.1667 |
| 10 | 80 | 0.6125 |
| 11 | 417 | 0.1151 |
| 12 | 834 | 0.0959 |
| 13 | 283 | 0.1802 |
| 14 | 11 | 0.0000 |

## Label rate by hour of day (train)

| hour | rows | long_view rate |
|---|---|---|
| 0 | 40,846 | 0.3293 |
| 1 | 23,389 | 0.3418 |
| 2 | 14,528 | 0.3467 |
| 3 | 9,770 | 0.3355 |
| 4 | 7,754 | 0.3496 |
| 5 | 9,518 | 0.3441 |
| 6 | 19,230 | 0.3680 |
| 7 | 32,665 | 0.3764 |
| 8 | 39,374 | 0.3745 |
| 9 | 43,760 | 0.3501 |
| 10 | 47,579 | 0.3361 |
| 11 | 52,024 | 0.3350 |
| 12 | 62,586 | 0.3502 |
| 13 | 62,303 | 0.3526 |
| 14 | 51,557 | 0.3387 |
| 15 | 50,658 | 0.3269 |
| 16 | 55,574 | 0.3178 |
| 17 | 60,868 | 0.3233 |
| 18 | 66,039 | 0.3347 |
| 19 | 69,574 | 0.3279 |
| 20 | 76,517 | 0.3252 |
| 21 | 85,935 | 0.3263 |
| 22 | 89,105 | 0.3300 |
| 23 | 69,959 | 0.3289 |

## Label rate by is_rand (train)

| is_rand | rows | long_view rate |
|---|---|---|
| 0 | 1,141,112 | 0.3366 |

## Validation split shape

- impressions per user: median 4, p90 12 (users: 22,377)
- users with zero positives: 30.3% (nDCG pinned at 0, excluded from GAUC)
- users with all positives: 11.9% (nDCG pinned at 1, excluded from GAUC)
- discriminative users: 57.8%
- repeated (user, video) pairs: 3,513 pairs covering 7,085 rows (5.67% of val), max multiplicity 7

## Train/validation overlap

- users: 21,955 of 22,377 val users seen in train (98.1%)
- items: 5,944 of 5,951 val videos seen in train (99.9%)

## play_time_ms vs duration_ms by long_view (train)

| long_view | rows | mean play_ms | median play_ms | mean dur_ms | mean watch_ratio | median watch_ratio |
|---|---|---|---|---|---|---|
| 0 | 756,991 | 3,546 | 2,027 | 97,644 | 94.329 | 0.032 |
| 1 | 384,121 | 62,113 | 42,419 | 98,344 | 156.511 | 0.978 |

long_view is a deterministic function of play_time and duration (KuaiRand defines it as play_time >= duration for short videos, >= 18s for longer ones), and duration_ms is known at impression time — watch-time/watch-ratio modelling is a direct auxiliary signal.

Evaluation ranks within a user (~5 impressions each), so user-constant features only help through interactions or a tree model; item, context and user x item signal is what moves GAUC / nDCG@5.
