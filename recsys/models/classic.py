"""random, popularity and the numpy FM port of the official baseline."""

from __future__ import annotations

import time

import numpy as np

from recsys.models.base import Recommender


class RandomRec(Recommender):
    def fit(self, X_train, y_train, groups_train, aux_train=None,
            X_val=None, y_val=None, groups_val=None, time_budget=300, seed=0):
        self.seed = seed
        self.info["rounds_used"] = 0
        return self

    def predict(self, X, groups):
        return np.random.default_rng(self.seed).random(len(X)).astype(np.float32)


class PopularityRec(Recommender):
    """Smoothed training-window long_view rate per video — exactly the hi_vid_lv_rate
    feature (prior=20), which equals the phase-1 popularity rung on validation."""

    def fit(self, X_train, y_train, groups_train, aux_train=None,
            X_val=None, y_val=None, groups_val=None, time_budget=300, seed=0):
        self.col = self._col("hi_vid_lv_rate")
        self.info["rounds_used"] = 0
        return self

    def predict(self, X, groups):
        return X[:, self.col].astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FMRec(Recommender):
    """Numpy port of starter_kit/baseline.py FM (k=16, lr=0.001, Adam, batch 8192,
    <=40 epochs, patience 4). Consumes the fm5 spec's train-vocab codes; internally
    offsets fields into one table exactly like the kit's encode()."""

    def fit(self, X_train, y_train, groups_train, aux_train=None,
            X_val=None, y_val=None, groups_val=None, time_budget=300, seed=0):
        cfg = {"k": 16, "lr": 0.001, "l2": 1e-6, "epochs": 40, "batch": 8192,
               "patience": 4, **self.config}
        dims = [self.meta["field_dims"][c] for c in self.meta["columns"]]
        self.offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)
        dim = int(sum(dims))
        Xtr = X_train.astype(np.int64) + self.offsets
        rng = np.random.default_rng(seed)
        k, lr, l2 = cfg["k"], cfg["lr"], cfg["l2"]
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        mV, vV = np.zeros_like(self.V), np.zeros_like(self.V)
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        t = 0
        deadline = self._deadline(time_budget)
        max_rounds = cfg.get("rounds")  # refit mode: fixed epoch count, no early stop
        best, best_state, bad, used = -1.0, None, 0, 0
        for ep in range(1, cfg["epochs"] + 1):
            idx = rng.permutation(len(y_train))
            for i in range(0, len(idx), cfg["batch"]):
                bidx = idx[i:i + cfg["batch"]]
                Xb, yb = Xtr[bidx], y_train[bidx]
                z, E, S = self._logits(Xb)
                g = ((_sigmoid(z) - yb) / len(yb)).astype(np.float32)
                gV = np.zeros_like(self.V)
                gW = np.zeros_like(self.W)
                np.add.at(gW, Xb, g[:, None])
                np.add.at(gV, Xb, g[:, None, None] * (S[:, None, :] - E))
                gV += l2 * self.V
                gW += l2 * self.W
                t += 1
                b1, b2, eps = 0.9, 0.999, 1e-8
                for P, G, M, Vv in ((self.V, gV, mV, vV), (self.W, gW, mW, vW)):
                    M *= b1
                    M += (1 - b1) * G
                    Vv *= b2
                    Vv += (1 - b2) * (G * G)
                    P -= lr * (M / (1 - b1 ** t)) / (np.sqrt(Vv / (1 - b2 ** t)) + eps)
                self.b -= lr * g.sum()
            used = ep
            if max_rounds or X_val is None:
                if (max_rounds and ep >= max_rounds) or time.time() > deadline:
                    break
                continue
            primary = self._val_primary(self.predict(X_val, groups_val))
            if primary > best + 1e-5:
                best, bad = primary, 0
                best_state = (self.V.copy(), self.W.copy(), np.float32(self.b))
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
            if time.time() > deadline:
                break
        if best_state is not None:
            self.V, self.W, self.b = best_state
        self.info.update({"rounds_used": used, "best_val_primary": best if best > 0 else None,
                          "config": cfg})
        return self

    def _logits(self, Xoff):
        E = self.V[Xoff]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[Xoff].sum(1) + inter, E, S

    def predict(self, X, groups):
        Xoff = X.astype(np.int64) + self.offsets
        out = [self._logits(Xoff[i:i + 200_000])[0] for i in range(0, len(Xoff), 200_000)]
        return np.concatenate(out).astype(np.float32)
