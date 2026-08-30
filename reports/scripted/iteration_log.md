# scripted: iteration log

## Iteration 1 — keep/revert: **keep**, primary 0.4608 (delta vs baseline -0.1408)

**Hypothesis**

> baseline reproduction

Diff: 13 lines ([diff.patch](../../runs/scripted/iterations/1/diff.patch))

Metrics: GAUC 0.4838, nDCG@5 0.4377, primary 0.4608, train 1s, peak RSS 51 MB, seeds [0]

## Iteration 2 — keep/revert: **keep**, primary 0.4657 (delta vs baseline -0.1359)

**Hypothesis**

> raise quality to 0.30

Diff: 13 lines ([diff.patch](../../runs/scripted/iterations/2/diff.patch))

Metrics: GAUC 0.4896, nDCG@5 0.4418, primary 0.4657, train 1s, peak RSS 51 MB, seeds [0]

## Iteration 3 — keep/revert: **revert**, primary 0.4591 (delta vs baseline -0.1425)

**Hypothesis**

> broken change, then fixed at 0.10

Diff: 13 lines ([diff.patch](../../runs/scripted/iterations/3/diff.patch))

Metrics: GAUC 0.4811, nDCG@5 0.4371, primary 0.4591, train 1s, peak RSS 50 MB, seeds [0]

Events:
- `error` (attempt 1): seed 0: rc=1 after 0.2s -> retry

## Iteration 4 — keep/revert: **keep**, primary 0.5144 (delta vs baseline -0.0872)

**Hypothesis**

> raise quality to 0.90

Diff: 13 lines ([diff.patch](../../runs/scripted/iterations/4/diff.patch))

Metrics: GAUC 0.5618, nDCG@5 0.4670, primary 0.5144, train 1s, peak RSS 52 MB, seeds [0]

## Iteration 5 — **abandoned**

**Hypothesis**

> infinite loop idea

Diff: 16 lines ([diff.patch](../../runs/scripted/iterations/5/diff.patch))

Events:
- `timeout` (attempt 1): seed 0: rc=15 after 13.2s -> retry
- `timeout` (attempt 2): seed 0: rc=15 after 13.2s -> retry
- `timeout` (attempt 3): seed 0: rc=15 after 13.1s -> abandon

## Iteration 6 — keep/revert: **revert**, primary 0.4792 (delta vs baseline -0.1224)

**Hypothesis**

> nan bug, then fixed at 0.50

Diff: 13 lines ([diff.patch](../../runs/scripted/iterations/6/diff.patch))

Metrics: GAUC 0.5059, nDCG@5 0.4526, primary 0.4792, train 1s, peak RSS 51 MB, seeds [0]

Events:
- `nan` (attempt 1): seed 0: NaN/Inf in scores -> retry

## Iteration 7 — keep/revert: **revert**, primary 0.5144 (delta vs baseline -0.0872)

**Hypothesis**

> retry 0.90 (equal, not better)

Diff: 0 lines ([diff.patch](../../runs/scripted/iterations/7/diff.patch))

Metrics: GAUC 0.5618, nDCG@5 0.4670, primary 0.5144, train 1s, peak RSS 51 MB, seeds [0]

## Iteration 8 — keep/revert: **revert**, primary 0.5101 (delta vs baseline -0.0915)

**Hypothesis**

> slightly below best

Diff: 13 lines ([diff.patch](../../runs/scripted/iterations/8/diff.patch))

Metrics: GAUC 0.5570, nDCG@5 0.4633, primary 0.5101, train 1s, peak RSS 51 MB, seeds [0]

Events:
- `stop` (attempt 1): converged -> stop

