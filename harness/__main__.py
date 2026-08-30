"""Harness CLI (FROZEN): start, iterate, abandon, revert, status, intervene, finish, report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import prepare  # noqa: E402
from harness import convergence, git_ops, iterate, ledger  # noqa: E402
from harness.run import Run, now  # noqa: E402

ITERATION_NOTE = ("An iteration is one scored experiment. Failed attempts (max 3) do not "
                  "consume an iteration number; after 3 the iteration is abandoned. "
                  "Abandoned iterations count toward the 50 cap but not toward the "
                  "convergence window, because they produced no score.")


def cmd_start(a) -> int:
    branch = f"autoresearch/{a.run_id}"
    if git_ops.is_dirty():
        print("REFUSED: git is dirty; commit or stash first")
        return 2
    if git_ops.branch_exists(branch):
        if not a.resume:
            print(f"REFUSED: branch {branch} exists (use --resume)")
            return 2
        git_ops.checkout_branch(branch)
        run = Run.load(a.run_id)
        print(f"resumed {branch} at iteration {run.state['next_iter']}")
        return 0
    base = git_ops.git("rev-parse", "--short", "main")
    proc = subprocess.run([sys.executable, "prepare.py", "--verify"], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print(f"REFUSED: prepare.py --verify failed\n{proc.stdout}{proc.stderr}")
        return 2
    git_ops.create_branch(branch, "main")
    eps_default, n_default = convergence.shipped_defaults()
    baseline = json.loads((REPO / "starter_kit" / "baseline_scores.json").read_text(encoding="utf-8"))
    fmv = baseline["scores"]["fm_official"]["valid"]
    run = Run(a.run_id)
    run.state = {
        "run_id": a.run_id, "dataset": a.dataset, "history_end": a.history_end,
        "time_budget": a.time_budget, "max_iters": a.max_iters,
        "wall_clock_hours": a.wall_clock_hours,
        "eps": a.eps if a.eps is not None else eps_default,
        "patience": a.patience if a.patience is not None else n_default,
        "seeds": [int(s) for s in str(a.seeds).split(",")],
        "rss_cap_gb": a.rss_cap_gb,
        "started_at": now(), "base_commit": base,
        "baseline": {"gauc": fmv["GAUC"], "ndcg5": fmv["nDCG@5"], "primary": fmv["primary"]},
        "status": "running", "best": None, "last_kept_iter": None,
        "next_iter": 1, "attempt": 1, "consecutive_bad": 0,
        "scored_primaries": [], "counts": {"scored": 0, "kept": 0, "reverted": 0, "abandoned": 0},
    }
    run.save()
    ledger.init_results()
    git_ops.commit_iteration(f"run {a.run_id}: start", iterate._ledger_paths(run))
    print(f"started {branch} @ {base} (baseline primary {fmv['primary']})")
    print(f"next: write runs/{a.run_id}/iterations/1/hypothesis.md "
          f'("baseline reproduction"), then: python -m harness iterate --desc "baseline reproduction"')
    return 0


def cmd_iterate(a) -> int:
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    return iterate.iterate(a.run_id, a.desc, seeds)


def cmd_abandon(a) -> int:
    run = Run.load(a.run_id) if a.run_id else Run.latest()
    if run.state["status"] != "running":
        print(f"run already stopped ({run.state['status']})")
        return 2
    i = run.state["next_iter"]
    ledger.append_event(run.it_dir(i), i, run.state["attempt"], "error",
                        f"manual abandon: {a.reason}", "abandon")
    return iterate.abandon_iteration(run, i, run.state["attempt"], f"abandoned: {a.reason}")


def cmd_revert(a) -> int:
    run = Run.load(a.run_id) if a.run_id else Run.latest()
    git_ops.restore_mutable(iterate.kept_commit(run))
    print(f"mutable surface restored to {iterate.kept_commit(run)} (working tree, no commit)")
    return 0


def cmd_status(a) -> int:
    run = Run.load(a.run_id) if a.run_id else Run.latest()
    st = run.state
    best = st["best"]
    print(f"run {run.run_id} on {run.branch}: {st['status']}")
    print(f"iterations: {run.iterations_used()}/{st['max_iters']} "
          f"(kept {st['counts']['kept']}, reverted {st['counts']['reverted']}, "
          f"abandoned {st['counts']['abandoned']})  next: {st['next_iter']} "
          f"attempt {st['attempt']}")
    if best:
        print(f"best: iter {best['iter']} @ {git_ops.find_iteration_commit(best['iter'])} "
              f"primary {best['primary']:.4f} "
              f"(baseline {st['baseline']['primary']:.4f})")
    print(f"elapsed: {run.elapsed() / 3600:.2f}h of {st['wall_clock_hours']}h")
    return 0


def cmd_intervene(a) -> int:
    run = Run.load(a.run_id) if a.run_id else Run.latest()
    path = run.dir / "interventions.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": a.note}) + "\n")
    print(f"logged intervention #{sum(1 for _ in open(path))}")
    return 0


def cmd_finish(a) -> int:
    run = Run.load(a.run_id) if a.run_id else Run.latest()
    st = run.state
    final = REPO / "submissions" / run.run_id / "final.csv"
    if final.exists():
        import numpy as np
        best_dir = REPO / "checkpoints" / run.run_id / "best"
        prepare.write_submission("test", np.load(best_dir / "test_scores.npy"), final,
                                 dataset=st["dataset"])
        print("final.csv re-validated (submit.py --check passed)")
    else:
        print("WARNING: no final.csv (no kept iteration)")

    interventions = run.dir / "interventions.jsonl"
    n_int = sum(1 for _ in open(interventions)) if interventions.exists() else 0
    gpu_note = ("no nvidia-smi on PATH; CPU-only run, GPU-hours 0"
                if not _which("nvidia-smi") else
                "nvidia-smi present but not sampled during the run; models ran on CPU")
    resources = {
        "iterations_used": run.iterations_used(),
        "scored": st["counts"]["scored"], "kept": st["counts"]["kept"],
        "reverted": st["counts"]["reverted"], "abandoned": st["counts"]["abandoned"],
        "wall_clock_hours": round(run.elapsed() / 3600, 3),
        "tokens": ({"in": a.tokens_in, "out": a.tokens_out}
                   if a.tokens_in is not None else None),
        "tokens_note": None if a.tokens_in is not None else
        "not tracked by the harness; agent-side accounting only",
        "gpu_hours": 0.0, "gpu_note": gpu_note,
        "interventions": n_int,
        "iteration_interpretation": ITERATION_NOTE,
    }
    (run.dir / "resources.json").write_text(json.dumps(resources, indent=1), encoding="utf-8")

    bundle = run.dir / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in sorted(run.dir.glob("iterations/*/out")):
            tar.add(p, arcname=str(p.relative_to(run.dir)))
        for p in sorted(run.dir.glob("iterations/*/stdout.log")):
            tar.add(p, arcname=str(p.relative_to(run.dir)))
    print(f"bundled {bundle.name} ({bundle.stat().st_size / 1e6:.1f} MB, untracked)")

    if a.also_refit and final.exists():
        try:
            _refit(run, st)
        except Exception as e:  # a corrupt checkpoint must not break finish
            print(f"refit failed (final.csv unaffected): {type(e).__name__}: {e}")

    if st["status"] == "running":
        st["status"] = "finished"
    run.save()
    ledger.backfill_commits()
    git_ops.commit_iteration(f"run {run.run_id}: finish ({st['status']})",
                             iterate._ledger_paths(run))
    print(f"run {run.run_id} finished with status {st['status']}; "
          f"next: python -m harness report --run-id {run.run_id}")
    return 0


def _refit(run, st):
    cfg = json.loads((REPO / "checkpoints" / run.run_id / "best" / "config.json").read_text(encoding="utf-8"))
    rounds = cfg["info"].get("rounds_used")
    extra = ["--config-json", json.dumps({"rounds": rounds}), "--history-end", "20220428",
             "--model", cfg["model"], "--features", cfg["features"],
             "--seed", str(cfg["seed"]), "--time-budget", str(st["time_budget"]),
             "--out", str(run.dir / "refit_out")]
    proc = subprocess.run([sys.executable, "train.py", *extra], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode == 0:
        import numpy as np
        scores = np.load(run.dir / "refit_out" / "test_scores.npy")
        prepare.write_submission("test", scores,
                                 REPO / "submissions" / run.run_id / "final_refit_trainval.csv",
                                 dataset=st["dataset"])
        print("wrote final_refit_trainval.csv (train+val refit, clearly labelled; "
              "final.csv unchanged)")
    else:
        print(f"refit failed (final.csv unaffected):\n{proc.stdout[-1000:]}")


def _which(name: str):
    import shutil
    return shutil.which(name)


def cmd_report(a) -> int:
    from harness import report
    return report.report(a.run_id)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start")
    s.add_argument("--run-id", required=True)
    s.add_argument("--dataset", default="pure")
    s.add_argument("--history-end", type=int, default=20220421)
    s.add_argument("--time-budget", type=int, default=300)
    s.add_argument("--max-iters", type=int, default=50)
    s.add_argument("--wall-clock-hours", type=float, default=6)
    s.add_argument("--eps", type=float, default=None)
    s.add_argument("--patience", type=int, default=None)
    s.add_argument("--seeds", default="0")
    s.add_argument("--rss-cap-gb", type=float, default=16)
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_start)

    s = sub.add_parser("iterate")
    s.add_argument("--desc", required=True)
    s.add_argument("--run-id", default=None)
    s.add_argument("--seeds", default=None)
    s.add_argument("--tokens-in", type=int, default=None)
    s.add_argument("--tokens-out", type=int, default=None)
    s.set_defaults(fn=cmd_iterate)

    s = sub.add_parser("abandon")
    s.add_argument("--reason", required=True)
    s.add_argument("--run-id", default=None)
    s.set_defaults(fn=cmd_abandon)

    s = sub.add_parser("revert")
    s.add_argument("--run-id", default=None)
    s.set_defaults(fn=cmd_revert)

    s = sub.add_parser("status")
    s.add_argument("--run-id", default=None)
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("intervene")
    s.add_argument("--note", required=True)
    s.add_argument("--run-id", default=None)
    s.set_defaults(fn=cmd_intervene)

    s = sub.add_parser("finish")
    s.add_argument("--run-id", default=None)
    s.add_argument("--tokens-in", type=int, default=None)
    s.add_argument("--tokens-out", type=int, default=None)
    s.add_argument("--also-refit", action="store_true")
    s.set_defaults(fn=cmd_finish)

    s = sub.add_parser("report")
    s.add_argument("--run-id", required=True)
    s.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
