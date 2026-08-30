"""Blend rung: weighted rank-average of registered models, or a linear stacker fitted
on out-of-fold training-window predictions built by date folds. Budget shares are
fixed configuration; the rank weights may optionally be picked from a small
`weight_grid` on validation (v2.2 — same precedent as the multitask head grid:
a handful of metric evaluations on already-computed predictions, cost recorded
in info)."""

from __future__ import annotations

import time

import numpy as np

from recsys.models.base import Recommender


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a))
    return r / max(1, len(a) - 1)


class Blend(Recommender):
    """config:
    - bases: [{model, cfg, share, weight, seed_offset}] (share = fraction of the time
      budget, weight = rank-average weight, seed_offset for bagging same-model bases),
      or legacy models: [name, ...] with equal shares.
    - mode: "rank_avg" (default) or "stack" (date-fold OOF linear stacker).
    - folds: date folds for stack mode (default 3).
    - weight_grid: optional list of weight tuples (one per base); the best on
      validation replaces the configured weights (rank_avg mode only).
    """

    def _bases(self):
        from recsys.models import MODELS
        cfg_bases = self.config.get("bases")
        if not cfg_bases:
            names = self.config.get("models", ["lgbm_lambdarank", "lgbm_pointwise"])
            cfg_bases = [{"model": n} for n in names]
        out = []
        for b in cfg_bases:
            cls, default_cfg, _spec = MODELS[b["model"]]
            out.append({
                "name": b["model"], "cls": cls,
                "cfg": {**default_cfg, **b.get("cfg", {})},
                "share": b.get("share", 1.0 / len(cfg_bases)),
                "weight": b.get("weight", 1.0 / len(cfg_bases)),
                "seed_offset": b.get("seed_offset", 0),
            })
        return out

    def fit(self, X_train, y_train, groups_train, aux_train=None,
            X_val=None, y_val=None, groups_val=None, time_budget=300, seed=0):
        mode = self.config.get("mode", "rank_avg")
        folds = int(self.config.get("folds", 3))
        bases = self._bases()
        self.mode = mode
        self.weights = np.array([b["weight"] for b in bases], dtype=np.float64)

        if mode == "stack":
            dates = (aux_train or {}).get("date")
            if dates is None:
                raise ValueError("blend stack mode needs aux_train['date'] (train row dates)")
            uniq = np.unique(dates)
            chunks = np.array_split(uniq, folds + 1)
            share = time_budget / max(1, (folds + 1) * len(bases))
            oof = {b["name"]: [] for b in bases}
            oof_y = []
            for k in range(1, folds + 1):
                fit_mask = np.isin(dates, np.concatenate(chunks[:k]))
                pred_mask = np.isin(dates, chunks[k])
                oof_y.append(y_train[pred_mask])
                for b in bases:
                    m = b["cls"](meta=self.meta,
                                 **{**b["cfg"], "rounds": self.config.get("fold_rounds", 150)})
                    m.fit(X_train[fit_mask], y_train[fit_mask], groups_train[fit_mask],
                          aux_train={k2: v[fit_mask] for k2, v in (aux_train or {}).items()},
                          time_budget=share, seed=seed)
                    p = m.predict(X_train[pred_mask], groups_train[pred_mask]).astype(np.float64)
                    oof[b["name"]].append((p - p.mean()) / (p.std() + 1e-9))
            Z = np.stack([np.concatenate(oof[b["name"]]) for b in bases], axis=1)
            yy = np.concatenate(oof_y)
            w, *_ = np.linalg.lstsq(Z, yy - yy.mean(), rcond=None)
            w = np.clip(w, 0, None)
            self.weights = (w / w.sum()) if w.sum() > 0 else np.full(len(bases), 1 / len(bases))
            self.info["stack_weights"] = {b["name"]: float(x) for b, x in zip(bases, self.weights)}
            share_final = time_budget / max(1, (folds + 1) * len(bases))
        else:
            share_final = None  # per-base share of the whole budget

        self.models = []
        for b in bases:
            budget = share_final if share_final else time_budget * b["share"]
            m = b["cls"](meta=self.meta, **b["cfg"])
            m.fit(X_train, y_train, groups_train, aux_train=aux_train,
                  X_val=X_val, y_val=y_val, groups_val=groups_val,
                  time_budget=budget, seed=seed + b["seed_offset"])
            self.models.append((b["name"], m))
        grid = self.config.get("weight_grid")
        if mode == "rank_avg" and grid and X_val is not None:
            t0 = time.time()
            ranks = [_ranks(m.predict(X_val, groups_val).astype(np.float64))
                     for _, m in self.models]
            best = (-1.0, tuple(self.weights))
            for wts in grid:
                p = self._val_primary(np.sum([w * r for w, r in zip(wts, ranks)], axis=0))
                if p > best[0] + 1e-5:
                    best = (p, tuple(wts))
            self.weights = np.asarray(best[1], dtype=np.float64)
            self.info["weight_grid"] = {"combos": len(grid), "chosen": list(best[1]),
                                        "primary": round(best[0], 6),
                                        "sec": round(time.time() - t0, 1)}
        self.info["rounds_used"] = {n: m.info.get("rounds_used") for n, m in self.models}
        self.info["bases"] = [{k: v for k, v in b.items() if k not in ("cls",)} for b in bases]
        return self

    def predict(self, X, groups):
        preds = [m.predict(X, groups).astype(np.float64) for _, m in self.models]
        if self.mode == "stack":
            z = [(p - p.mean()) / (p.std() + 1e-9) for p in preds]
            return np.sum([w * p for w, p in zip(self.weights, z)], axis=0).astype(np.float32)
        return np.sum([w * _ranks(p) for w, p in zip(self.weights, preds)],
                      axis=0).astype(np.float32)
