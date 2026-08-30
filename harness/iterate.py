"""The iterate flow (FROZEN): one hypothesis, one experiment, one commit.

Failure attempts (crash/timeout/oom/nan/shape/missing) do not consume the iteration
number; after 3 failed attempts the iteration is abandoned (counts toward the cap,
not toward the convergence window). Metrics are always recomputed by the harness
from val_scores.npy through prepare.evaluate — train.py's own printout is ignored.

Commit hashes are never stored in state (a file cannot carry the hash of the commit
that contains it): the kept/best commit is derived from the git log by message
prefix `iter <n>:`, and results.tsv commit cells are backfilled the same way.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import prepare  # noqa: E402
from harness import convergence, git_ops, ledger, watchdog  # noqa: E402
from harness.run import Run  # noqa: E402

MAX_ATTEMPTS = 3
DIVERGENCE_AFTER = 5


def kept_commit(run: Run) -> str:
    ki = run.state.get("last_kept_iter")
    if ki:
        h = git_ops.find_iteration_commit(ki)
        if h:
            return h
    return run.state["base_commit"]


def _fmt_h(sec: float) -> str:
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m"


def _print_stop(reason: str, run: Run) -> None:
    best = run.state["best"]
    print(f"STOP <{reason}>")
    if best:
        print(f"best: iter {best['iter']} @ {git_ops.find_iteration_commit(best['iter'])} "
              f"({best['primary']:.4f})")
    print(f"budget: iterations {run.iterations_used()}/{run.state['max_iters']}  "
          f"elapsed {_fmt_h(run.elapsed())}/{_fmt_h(run.state['wall_clock_hours'] * 3600)}")
    print("next: python -m harness finish")


def _tail(path: Path, n: int = 5) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]:
            print(f"  | {line}")


def _ledger_paths(run: Run) -> list[str]:
    paths = ["results.tsv", run.dir.relative_to(REPO).as_posix()]
    sub = REPO / "submissions" / run.run_id
    if sub.exists():
        paths.append(sub.relative_to(REPO).as_posix())
    return paths


def _fail(run: Run, i: int, attempt: int, etype: str, detail: str, stdout_log: Path) -> int:
    it_dir = run.it_dir(i)
    if attempt < MAX_ATTEMPTS:
        ledger.append_event(it_dir, i, attempt, etype, detail, "retry")
        run.state["attempt"] = attempt + 1
        run.save()
        print(f"ERROR {etype} (attempt {attempt}/{MAX_ATTEMPTS})")
        _tail(stdout_log)
        print(f"next: fix the cause and rerun iterate (attempt {attempt + 1}), "
              f'or: python -m harness abandon --reason "..."')
        return 1
    ledger.append_event(it_dir, i, attempt, etype, detail, "abandon")
    return abandon_iteration(run, i, attempt, f"abandoned after {MAX_ATTEMPTS} attempts ({etype})")


def abandon_iteration(run: Run, i: int, attempt: int, description: str) -> int:
    st = run.state
    git_ops.restore_mutable(kept_commit(run))
    st["consecutive_bad"] = st.get("consecutive_bad", 0) + 1
    st["counts"]["abandoned"] += 1
    ledger.append_result(i, "abandoned", None, st["seeds"], description, None, None)
    _advance_and_commit(run, i, attempt, f"iter {i}: {description}")
    _summary(run, i, attempt, "abandoned", None, None)
    return 1


def _advance_and_commit(run: Run, i: int, attempt: int, message: str) -> str:
    st = run.state
    if st.get("consecutive_bad", 0) >= DIVERGENCE_AFTER and st["best"]:
        ledger.append_event(run.it_dir(i), i, attempt, "divergence",
                            f"{st['consecutive_bad']} consecutive bad iterations", "warn")
    reason = _check_stop(run, i, attempt)
    if reason:
        st["status"] = reason
    st["next_iter"] = i + 1
    st["attempt"] = 1
    run.save()
    return git_ops.commit_iteration(message, _ledger_paths(run))


def _check_stop(run: Run, i: int, attempt: int) -> str | None:
    st = run.state
    reason = convergence.stop_reason(
        st["scored_primaries"], run.iterations_used(), run.elapsed(),
        eps=st["eps"], n=st["patience"], max_iters=st["max_iters"],
        wall_clock_hours=st["wall_clock_hours"])
    if reason:
        ledger.append_event(run.it_dir(i), i, attempt, "stop", reason, "stop")
    return reason


def _summary(run: Run, i: int, attempt: int, status: str, metrics, best_before) -> None:
    st = run.state
    print(f"=== ITERATION {i} (attempt {attempt}) ===")
    if status == "abandoned":
        print("DECISION: ABANDONED")
    else:
        print(f"DECISION: {status.upper()}")
        print(f"primary: {metrics['primary']:.4f}  gauc: {metrics['gauc']:.4f}  "
              f"ndcg5: {metrics['ndcg5']:.4f}")
        base = st["baseline"]["primary"]
        ref = best_before if best_before is not None else base
        print(f"delta_vs_baseline: {metrics['primary'] - base:+.4f}  "
              f"delta_vs_best: {metrics['primary'] - ref:+.4f}")
    best = st["best"]
    if best:
        print(f"best_so_far: iter {best['iter']} @ {git_ops.find_iteration_commit(best['iter'])} "
              f"({best['primary']:.4f})")
    n = st["patience"]
    deltas = convergence.window_deltas(st["scored_primaries"], n)
    if deltas:
        mx = max(deltas)
        conv = st["status"] == "converged"
        print(f"convergence: window [{', '.join(f'{d:+.4f}' for d in deltas)}] "
              f"max {mx:+.4f} {'<=' if conv else '>'} eps {st['eps']:.4f} -> "
              f"{'converged' if conv else 'continuing'}")
    else:
        print(f"convergence: window needs {n + 1}+ scored iterations -> continuing")
    print(f"budget: iterations {run.iterations_used()}/{st['max_iters']}  "
          f"elapsed {_fmt_h(run.elapsed())}/{_fmt_h(st['wall_clock_hours'] * 3600)}")
    if st.get("consecutive_bad", 0) >= DIVERGENCE_AFTER and best:
        print(f"DIVERGENCE: {st['consecutive_bad']} consecutive revert/abandoned iterations. "
              f"Best is iter {best['iter']} @ {git_ops.find_iteration_commit(best['iter'])}; "
              f"see the recovery advice in program.md.")
    if st["status"] != "running":
        print(f"STOP <{st['status']}>")
        print("next: python -m harness finish")
    else:
        print(f"next: write runs/{run.run_id}/iterations/{i + 1}/hypothesis.md, "
              f'then: python -m harness iterate --desc "..."')


def iterate(run_id: str | None, desc: str, seeds: list[int] | None) -> int:
    run = Run.load(run_id) if run_id else Run.latest()
    st = run.state
    if st["status"] != "running":
        _print_stop(st["status"], run)
        return 2
    reason = _check_stop(run, st["next_iter"], st["attempt"])
    if reason:
        st["status"] = reason
        run.save()
        ledger.backfill_commits()
        git_ops.commit_iteration(f"run {run.run_id}: stop ({reason})", _ledger_paths(run))
        _print_stop(reason, run)
        return 2

    i = st["next_iter"]
    attempt = st["attempt"]
    seeds = seeds or st["seeds"]
    it_dir = run.it_dir(i)
    hyp = it_dir / "hypothesis.md"
    if not hyp.exists() or not hyp.read_text(encoding="utf-8", errors="replace").strip():
        print(f"REFUSED: write a non-empty {hyp.relative_to(REPO)} first")
        return 2

    try:
        return _run_iteration(run, i, attempt, seeds, desc)
    except Exception:  # no traceback may escape the harness
        import traceback
        detail = traceback.format_exc().strip().splitlines()[-1]
        return _fail(run, i, attempt, "error", f"harness exception: {detail}",
                     run.it_dir(i) / "stdout.log")


def _run_iteration(run: Run, i: int, attempt: int, seeds: list[int], desc: str) -> int:
    st = run.state
    it_dir = run.it_dir(i)
    it_dir.mkdir(parents=True, exist_ok=True)
    (it_dir / "diff.patch").write_text(git_ops.diff_since(kept_commit(run)), encoding="utf-8")
    stdout_log = it_dir / "stdout.log"

    per_seed_metrics, val_scores, test_scores = [], [], []
    total_train, peak = 0.0, 0.0
    n_val = prepare.load("val", dataset=st["dataset"]).height
    n_test = prepare.load("test", dataset=st["dataset"]).height
    for seed in seeds:
        out = it_dir / "out" / f"s{seed}"
        res = watchdog.run_train(out, seed, st["time_budget"], st["rss_cap_gb"], stdout_log)
        total_train += res["elapsed"]
        peak = max(peak, res["peak_rss_mb"])
        if res["status"] != "ok":
            return _fail(run, i, attempt, res["status"],
                         f"seed {seed}: rc={res['returncode']} after {res['elapsed']}s",
                         stdout_log)
        vp, tp = out / "val_scores.npy", out / "test_scores.npy"
        if not vp.exists() or not tp.exists():
            return _fail(run, i, attempt, "missing", f"seed {seed}: missing output arrays",
                         stdout_log)
        v, t = np.load(vp), np.load(tp)
        if len(v) != n_val or len(t) != n_test:
            return _fail(run, i, attempt, "shape",
                         f"seed {seed}: val {len(v)}/{n_val} test {len(t)}/{n_test}",
                         stdout_log)
        if not (np.isfinite(v).all() and np.isfinite(t).all()):
            return _fail(run, i, attempt, "nan", f"seed {seed}: NaN/Inf in scores", stdout_log)
        per_seed_metrics.append(prepare.evaluate("val", v, dataset=st["dataset"]))
        val_scores.append(v.astype(np.float64))
        test_scores.append(t.astype(np.float64))

    metrics = {k: float(np.mean([m[k] for m in per_seed_metrics]))
               for k in ("primary", "gauc", "ndcg5")}
    best_before = st["best"]["primary"] if st["best"] else None
    decision = "keep" if (best_before is None or metrics["primary"] > best_before) else "revert"

    if decision == "keep":
        best_dir = REPO / "checkpoints" / run.run_id / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(it_dir / "out" / f"s{seeds[0]}", best_dir)
        mean_val = np.mean(val_scores, axis=0).astype(np.float32)
        mean_test = np.mean(test_scores, axis=0).astype(np.float32)
        np.save(best_dir / "val_scores.npy", mean_val)
        np.save(best_dir / "test_scores.npy", mean_test)
        sub_dir = REPO / "submissions" / run.run_id
        sub_path = sub_dir / f"iter_{i}.csv"
        prepare.write_submission("test", mean_test, sub_path, dataset=st["dataset"])
        shutil.copyfile(sub_path, sub_dir / "final.csv")
        st["best"] = {"iter": i, "primary": metrics["primary"]}
        st["last_kept_iter"] = i
        st["consecutive_bad"] = 0
        st["counts"]["kept"] += 1
    else:
        git_ops.restore_mutable(kept_commit(run))
        st["consecutive_bad"] = st.get("consecutive_bad", 0) + 1
        st["counts"]["reverted"] += 1

    st["counts"]["scored"] += 1
    st["scored_primaries"].append(metrics["primary"])
    base = st["baseline"]["primary"]
    ledger.append_result(i, decision, metrics, seeds, desc, metrics["primary"] - base,
                         total_train)
    ledger.write_metrics(it_dir, {
        "iter": i, "attempt": attempt, "seeds": seeds,
        "primary": metrics["primary"], "gauc": metrics["gauc"], "ndcg5": metrics["ndcg5"],
        "per_seed": per_seed_metrics, "train_sec": total_train, "total_sec": total_train,
        "peak_rss_mb": peak, "n_val": n_val, "n_test": n_test, "decision": decision,
        "best_before": best_before,
        "delta_vs_best": metrics["primary"] - (best_before if best_before is not None else base),
        "delta_vs_baseline": metrics["primary"] - base,
    })
    _advance_and_commit(run, i, attempt,
                        f"iter {i}: {desc} (primary={metrics['primary']:.4f})")
    _summary(run, i, attempt, decision, metrics, best_before)
    return 0
