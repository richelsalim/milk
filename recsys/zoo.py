"""Model zoo CLI: `python -m recsys.zoo list` and `python -m recsys.zoo bench`.

bench trains each model on the training split with the given wall-clock budget,
scores validation through prepare.evaluate, appends a row to reports/zoo_baselines.md
and saves checkpoints/zoo/<name>/val_scores.npy.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare  # noqa: E402
from recsys.features import build  # noqa: E402
from recsys.models import MODELS  # noqa: E402

AUX_COLS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
            "long_view", "play_time_ms", "duration_ms", "date"]
MD = Path(prepare.REPO) / "reports" / "zoo_baselines.md"
HEADER = ("| model | primary | gauc | ndcg5 | train_sec | peak_rss_mb | spec | notes |\n"
          "|---|---|---|---|---|---|---|---|\n")


class _RssSampler:
    def __init__(self):
        self.proc = psutil.Process()
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.thread.join(timeout=2)


def make_aux(train_frame) -> dict:
    return {c: train_frame[c].to_numpy().astype(np.float64) for c in AUX_COLS
            if c in train_frame.columns}


def bench_one(name: str, budget: int, seeds: list[int], dataset: str = "pure",
              history_end: int = prepare.HISTORY_END, notes: str = "") -> dict:
    cls, cfg, spec = MODELS[name]
    Xtr, meta, gtr = build(spec, "train", history_end, dataset)
    Xva, _, gva = build(spec, "val", history_end, dataset)
    train_frame = prepare.load("train", dataset=dataset, history_end=history_end)
    ytr = train_frame["long_view"].to_numpy().astype(np.float32)
    aux = make_aux(train_frame)
    val_frame = prepare.load("val", dataset=dataset)
    yva = val_frame["long_view"].to_numpy().astype(np.float32)

    per_seed, scores0, train_sec, peak_mb = [], None, 0.0, 0.0
    for seed in seeds:
        model = cls(meta=meta, **cfg)
        with _RssSampler() as rss:
            t0 = time.time()
            model.fit(Xtr, ytr, gtr, aux_train=aux, X_val=Xva, y_val=yva, groups_val=gva,
                      time_budget=budget, seed=seed)
            train_sec = time.time() - t0
            peak_mb = rss.peak / 1e6
        scores = model.predict(Xva, gva)
        if seed == seeds[0]:
            scores0 = scores
        per_seed.append(prepare.evaluate("val", scores, dataset=dataset))
    res = {k: float(np.mean([s[k] for s in per_seed])) for k in ("primary", "gauc", "ndcg5")}
    res.update(train_sec=train_sec, peak_rss_mb=peak_mb, spec=spec, per_seed=per_seed,
               info=model.info)

    ckpt = Path(prepare.REPO) / "checkpoints" / "zoo" / name
    ckpt.mkdir(parents=True, exist_ok=True)
    np.save(ckpt / "val_scores.npy", scores0.astype(np.float32))

    if not MD.exists():
        MD.write_text("# Zoo baselines (validation)\n\n" + HEADER, encoding="utf-8")
    note = notes or (f"seeds {seeds}" if len(seeds) > 1 else "")
    sub = model.info.get("subsampled_rows")
    if sub:
        note = (note + f" subsample={sub}").strip()
    with open(MD, "a", encoding="utf-8") as fh:
        fh.write(f"| {name} | {res['primary']:.4f} | {res['gauc']:.4f} | {res['ndcg5']:.4f} "
                 f"| {train_sec:.0f} | {peak_mb:.0f} | {spec} | {note} |\n")
    print(f"{name}: primary {res['primary']:.4f} gauc {res['gauc']:.4f} "
          f"ndcg5 {res['ndcg5']:.4f} ({train_sec:.0f}s, {peak_mb:.0f} MB)")
    return res


def main():
    ap = argparse.ArgumentParser(prog="python -m recsys.zoo")
    ap.add_argument("cmd", choices=["list", "bench"])
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--models", default="")
    ap.add_argument("--dataset", default="pure")
    a = ap.parse_args()
    if a.cmd == "list":
        for name, (cls, cfg, spec) in MODELS.items():
            print(f"{name:16s} {cls.__name__:14s} spec={spec} {cfg or ''}")
        return
    seeds = [int(s) for s in a.seeds.split(",")]
    names = [m for m in a.models.split(",") if m] or list(MODELS)
    for name in names:
        bench_one(name, a.budget, seeds, a.dataset)


if __name__ == "__main__":
    main()
