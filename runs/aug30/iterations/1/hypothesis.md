# Iteration 1: baseline reproduction

- Stage: none (control).
- Change: none — unmodified train.py (model fm, spec fm5, seed 0, 300 s budget).
- Why: the ledger starts at the reproduced official Factorization Machine baseline so
  every later delta is measured against a number this harness produced, not a quoted
  one. program.md setup step 3.
- Expected: primary ~0.6015 on validation (published 0.6016, seed std 0.0008).
- Rollback: nothing to roll back.
