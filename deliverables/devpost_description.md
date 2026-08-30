# Devpost project description (Deliverable 1 — paste-ready)

**Headline: +0.0055 validation primary over the official FM baseline (+0.0069 GAUC,
+0.0041 nDCG@5) on KuaiRand-Pure — CPU-only, inside a 300 s training budget, with a
fully autonomous, auditable research loop (0 manual interventions in rehearsal).**

## How the solution addresses the problem statement

We built an autoresearch-style autonomous ML research agent for the KuaiRand-Pure
ranking benchmark, split into three zones so the agent can experiment aggressively
without being able to cheat:

- **Frozen** (hash-verified): the organizer starter kit vendored byte-for-byte;
  `prepare.py`, which pins the date splits, strips test labels before they can enter
  memory, and scores everything through the organizer's own `evaluate.py`; the harness;
  and the test suite (baseline parity, leakage guards, time-safety proofs, fault
  injection).
- **Mutable** (the agent's whole surface): `train.py` plus a composable environment
  library `recsys/` — a time-safe feature layer (strict-past aggregates for training
  rows, full-training-window joins for validation/test), a 12-rung model zoo (exact
  numpy FM parity port, LightGBM pointwise/lambdarank, DeepFM, DCNv2, MMoE, PLE,
  censored watch-time regression per the CWM reference, DIN-style sequence attention,
  and blending), and within-user ranking losses (BPR, listwise softmax).
- **Human-owned**: `program.md`, the agent's runtime instructions.

The harness runs the Figure-1 loop end to end: the agent writes a hypothesis, edits the
mutable surface, and calls `iterate`; the harness runs training under a watchdog (hard
timeout, RSS cap), **recomputes** GAUC/nDCG@5 from the saved score arrays (never
trusting the training script's printout), decides KEEP or REVERT against the best
validation primary, writes a validated submission for every kept iteration, commits
exactly once per iteration, and stops on the pinned convergence rule (ε = 0.002,
N = 3), the 50-iteration cap, or the 6 h ceiling. Reproduction of the official
baseline is exact (validation primary 0.6016), and the environment's current
validation-best — a budget-feasible weighted rank-average of DeepFM and PLE — reaches
**0.6071 (+0.0055)** within a single 300 s CPU budget.

Robustness is tested adversarially, not asserted: a scripted 9-step agent rehearsal
(deliberate syntax error, infinite loop, NaN scores, equal-score decisions,
convergence stop — 34/34 checks) and a fault-injection suite (OOM, disk-full on the
submission write, corrupted checkpoint, dirty git, self-deleting outputs) all pass;
failures become typed events with retry-then-abandon semantics and the run continues.

## Development tools used

Claude Code (the LLM research/coding agent), uv, ruff, pytest, git, Windows 11
(CPU-only, 12 logical cores — no GPU anywhere in the pipeline).

## APIs used

None at runtime — the pipeline is fully local. The LLM agent driving development and
research runs is Claude (Anthropic); its token usage is reported per run via
`harness finish --tokens-in/--tokens-out` into `resources.json`.

## Libraries and frameworks used

Python 3.11, numpy, polars, pyarrow, LightGBM, PyTorch (CPU wheels), scikit-learn
(utilities only), matplotlib (reports), psutil (watchdog RSS sampling).

## Datasets and assets used

- KuaiRand-Pure (required) and KuaiRand-1k (bonus, wired end to end), from the
  organizer-referenced Zenodo record (https://kuairand.com); the organizer starter kit
  vendored unchanged under `starter_kit/` with a SHA-256 manifest.
- karpathy/autoresearch README + program.md vendored under `docs/autoresearch/` (MIT)
  as the loop's design reference.
- No external training data and no pretrained weights — training uses only the
  KuaiRand files (the one hard rule), enforced by static leakage guards in the frozen
  test suite.
