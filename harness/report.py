"""`python -m harness report --run-id <tag>` -> reports/<tag>/ (FROZEN, phase 6).

Reads runs/<tag>/ (run.json, iterations/*/, interventions.jsonl, resources.json) and
results.tsv when present; the iteration data itself comes from the per-iteration
artifacts so a run directory copied from another clone still reports fully.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HIDDEN_TEST_BASELINE = {"gauc": 0.6610, "ndcg5": 0.5282, "primary": 0.5946}


def _iterations(run_dir: Path) -> list[dict]:
    out = []
    it_root = run_dir / "iterations"
    if not it_root.exists():
        return out
    for d in sorted(it_root.iterdir(), key=lambda p: int(p.name)):
        item = {"iter": int(d.name), "dir": d}
        m = d / "metrics.json"
        item["metrics"] = json.loads(m.read_text(encoding="utf-8")) if m.exists() else None
        h = d / "hypothesis.md"
        item["hypothesis"] = h.read_text(encoding="utf-8").strip() if h.exists() else "(missing)"
        e = d / "events.jsonl"
        item["events"] = ([json.loads(line) for line in e.read_text(encoding="utf-8").splitlines()]
                          if e.exists() else [])
        p = d / "diff.patch"
        item["diff_lines"] = len(p.read_text(encoding="utf-8", errors="replace").splitlines()) if p.exists() else 0
        out.append(item)
    return out


def report(run_id: str) -> int:
    run_dir = REPO / "runs" / run_id
    state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    iters = _iterations(run_dir)
    out_dir = REPO / "reports" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    base = state["baseline"]
    scored = [i for i in iters if i["metrics"]]
    best = max(scored, key=lambda i: i["metrics"]["primary"], default=None)

    # results_table.md -------------------------------------------------------
    lines = [f"# {run_id}: results", ""]
    if best:
        bm = best["metrics"]
        lines += [
            "| metric | validation best | official baseline (valid) | delta |",
            "|---|---|---|---|",
            f"| GAUC | {bm['gauc']:.4f} | {base['gauc']:.4f} | {bm['gauc'] - base['gauc']:+.4f} |",
            f"| nDCG@5 | {bm['ndcg5']:.4f} | {base['ndcg5']:.4f} | {bm['ndcg5'] - base['ndcg5']:+.4f} |",
            f"| **primary** | **{bm['primary']:.4f}** | **{base['primary']:.4f}** | "
            f"**{bm['primary'] - base['primary']:+.4f}** |",
            "",
            f"Best iteration: {best['iter']}. Iterations used: "
            f"{len(iters)} of {state['max_iters']} (scored {len(scored)}, "
            f"abandoned {len(iters) - len(scored)}). Run status: {state['status']}.",
            "",
            "For reference, the published hidden-test baseline (not our score): "
            f"GAUC {HIDDEN_TEST_BASELINE['gauc']:.4f} / nDCG@5 {HIDDEN_TEST_BASELINE['ndcg5']:.4f} "
            f"/ primary {HIDDEN_TEST_BASELINE['primary']:.4f}; final.csv holds this run's "
            "hidden-test submission.",
        ]
    else:
        lines += ["No scored iterations."]
    (out_dir / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # resources.md -----------------------------------------------------------
    res_path = run_dir / "resources.json"
    lines = [f"# {run_id}: resources", ""]
    if res_path.exists():
        res = json.loads(res_path.read_text(encoding="utf-8"))
        for k, v in res.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("resources.json missing — run `python -m harness finish` first.")
    (out_dir / "resources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # interventions.md -------------------------------------------------------
    ipath = run_dir / "interventions.jsonl"
    lines = [f"# {run_id}: manual interventions", ""]
    if ipath.exists():
        rows = [json.loads(line) for line in ipath.read_text(encoding="utf-8").splitlines()]
        lines += [f"- {r['ts']}: {r['note']}" for r in rows] or ["(none logged)"]
        lines.append(f"\nTotal: {len(rows)}")
    else:
        lines.append("(none logged)")
    (out_dir / "interventions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # iteration_log.md -------------------------------------------------------
    lines = [f"# {run_id}: iteration log", ""]
    for it in iters:
        m = it["metrics"]
        head = (f"keep/revert: **{m['decision']}**, primary {m['primary']:.4f} "
                f"(delta vs baseline {m['delta_vs_baseline']:+.4f})" if m else "**abandoned**")
        lines += [f"## Iteration {it['iter']} — {head}", "",
                  "**Hypothesis**", "",
                  *(f"> {line}" for line in it["hypothesis"].splitlines()), ""]
        lines += [f"Diff: {it['diff_lines']} lines "
                  f"([diff.patch](../../runs/{run_id}/iterations/{it['iter']}/diff.patch))", ""]
        if m:
            lines += [f"Metrics: GAUC {m['gauc']:.4f}, nDCG@5 {m['ndcg5']:.4f}, "
                      f"primary {m['primary']:.4f}, train {m['train_sec']:.0f}s, "
                      f"peak RSS {m['peak_rss_mb']:.0f} MB, seeds {m['seeds']}", ""]
        if it["events"]:
            lines.append("Events:")
            for e in it["events"]:
                lines.append(f"- `{e['type']}` (attempt {e['attempt']}): {e['detail']} "
                             f"-> {e['action']}")
            lines.append("")
    (out_dir / "iteration_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # trajectory.png ---------------------------------------------------------
    if scored:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [i["iter"] for i in scored]
        ys = [i["metrics"]["primary"] for i in scored]
        run_best = []
        cur = -1
        for y in ys:
            cur = max(cur, y)
            run_best.append(cur)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, "o-", label="validation primary", color="#1f77b4")
        ax.plot(xs, run_best, "--", label="best so far", color="#2ca02c")
        ax.axhline(base["primary"], color="#d62728", lw=1, label=f"baseline {base['primary']:.4f}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("validation primary")
        ax.set_title(f"{run_id}: validation primary per iteration")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory.png", dpi=120)
        plt.close(fig)

    print(f"report written to {out_dir} "
          f"({'with' if scored else 'without'} trajectory.png)")
    return 0
