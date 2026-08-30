"""train.py — MUTABLE: composes features + model + loss + training (see CLAUDE.md).

Contract: python train.py --out <dir> [--seed S] [--time-budget SEC] [--model NAME]
[--features NAME]. Writes val_scores.npy, test_scores.npy, config.json to <dir>,
exits 0. Defaults reproduce the official FM baseline (model fm, spec fm5).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402
from recsys.features import build  # noqa: E402
from recsys.models import MODELS  # noqa: E402
from recsys.zoo import make_aux  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget", type=int, default=300)
    ap.add_argument("--model", default="fm")
    ap.add_argument("--features", default=None)
    ap.add_argument("--config-json", default=None,
                    help="JSON dict merged over the model's default config (used by refit)")
    ap.add_argument("--history-end", type=int, default=prepare.HISTORY_END,
                    help=">20220421 = refit mode (train+val window, fixed rounds, no val eval)")
    a = ap.parse_args()

    cls, cfg, default_spec = MODELS[a.model]
    spec = a.features or default_spec
    cfg = dict(cfg)
    if a.config_json:
        raw = a.config_json
        cfg.update(json.loads(Path(raw).read_text(encoding="utf-8") if Path(raw).exists() else raw))

    # the harness invokes train.py without a dataset flag; a 1k/27k run exports
    # KUAIRAND_DATASET in the shell that calls `harness iterate` (default: pure)
    ds = os.environ.get("KUAIRAND_DATASET", "pure")
    refit = a.history_end > prepare.SPLITS["train"][1]
    t0 = time.time()
    Xtr, meta, gtr = build(spec, "train", history_end=a.history_end, dataset=ds)
    Xte, _, gte = build(spec, "test", history_end=a.history_end, dataset=ds)
    train_frame = prepare.load("train", dataset=ds, history_end=a.history_end)
    ytr = train_frame["long_view"].to_numpy().astype(np.float32)
    aux = make_aux(train_frame)
    if refit:
        Xva = gva = yva = None
    else:
        Xva, _, gva = build(spec, "val", dataset=ds)
        yva = prepare.load("val", dataset=ds)["long_view"].to_numpy().astype(np.float32)

    model = cls(meta=meta, **cfg)
    model.fit(Xtr, ytr, gtr, aux_train=aux, X_val=Xva, y_val=yva, groups_val=gva,
              time_budget=a.time_budget, seed=a.seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test_scores = model.predict(Xte, gte).astype(np.float32)
    np.save(out / "test_scores.npy", test_scores)
    val_scores = test_scores
    if not refit:
        val_scores = model.predict(Xva, gva).astype(np.float32)
        np.save(out / "val_scores.npy", val_scores)
    model.save(out)
    config = {
        "model": a.model, "features": spec, "dataset": ds,
        "seed": a.seed, "time_budget": a.time_budget,
        "config": cfg, "info": model.info, "history_end": a.history_end,
        "n_val": None if refit else len(val_scores), "n_test": len(test_scores),
        "total_sec": round(time.time() - t0, 1),
    }
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str), encoding="utf-8")
    print(f"train.py done: model={a.model} spec={spec} seed={a.seed} "
          f"rounds={model.info.get('rounds_used')} in {config['total_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
