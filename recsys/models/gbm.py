"""LightGBM rungs: pointwise binary and lambdarank, early-stopped on the organizer
metric (validation primary through prepare.evaluate) via a periodic callback."""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

from recsys.models.base import Recommender

EVAL_EVERY = 25


class _LGBMBase(Recommender):
    objective = "binary"

    def _params(self, seed):
        p = {"objective": self.objective, "learning_rate": 0.05, "num_leaves": 127,
             "min_data_in_leaf": 100, "feature_fraction": 0.8, "verbosity": -1,
             "deterministic": True, "num_threads": 4, "seed": seed, "force_row_wise": True}
        if self.objective == "lambdarank":
            p["eval_at"] = [5]  # default label_gain already equals 2^rel - 1
        p.update({k: v for k, v in self.config.items()
                  if k not in ("rounds", "max_rounds", "patience_evals",
                               "exclude_id_cats", "min_day")})
        return p

    def _cat_idx(self):
        cats = self.meta["categorical_idx"]
        if self.config.get("exclude_id_cats"):
            cols = self.meta["columns"]
            cats = [i for i in cats if not cols[i].startswith(("id_", "fm5_user", "fm5_video"))]
        return cats

    def _train_mask(self, X):
        """Optional alignment fix: drop early train rows whose strict-past history is
        much thinner than the full window val/test rows see (config min_day)."""
        min_day = self.config.get("min_day")
        if not min_day:
            return None
        col = self._col("ctx_days_since_start")
        return X[:, col] >= min_day

    def _dataset(self, X, y, groups, seed):
        mask = self._train_mask(X)
        if mask is not None:
            X, y, groups = X[mask], y[mask], groups[mask]
            self.info["train_rows_used"] = int(mask.sum())
        if self.objective == "lambdarank":
            perm = np.argsort(groups, kind="stable")
            _, counts = np.unique(groups[perm], return_counts=True)
            # LightGBM caps a query at 10k rows; split heavier users (1k/27k datasets)
            # into adjacent sub-lists — within-user ordering still trains per chunk.
            cap = 10_000
            chunked = []
            for c in counts:
                while c > cap:
                    chunked.append(cap)
                    c -= cap
                chunked.append(c)
            data = lgb.Dataset(X[perm], label=y[perm].astype(np.int32),
                               group=np.asarray(chunked),
                               categorical_feature=self._cat_idx(),
                               free_raw_data=True)
        else:
            data = lgb.Dataset(X, label=y, categorical_feature=self._cat_idx(),
                               free_raw_data=True)
        return data

    def fit(self, X_train, y_train, groups_train, aux_train=None,
            X_val=None, y_val=None, groups_val=None, time_budget=300, seed=0):
        params = self._params(seed)
        data = self._dataset(X_train, y_train, groups_train, seed)
        max_rounds = int(self.config.get("max_rounds", 2000))
        fixed = self.config.get("rounds")  # refit mode: exact round count, no callback
        deadline = self._deadline(time_budget)
        state = {"best": -1.0, "best_iter": 0, "bad": 0}
        patience = int(self.config.get("patience_evals", 4))

        def cb(env):
            it = env.iteration + 1
            if it % EVAL_EVERY and time.time() < deadline:
                return
            scores = env.model.predict(X_val, num_iteration=it)
            primary = self._val_primary(scores)
            if primary > state["best"] + 1e-5:
                state.update(best=primary, best_iter=it, bad=0)
            else:
                state["bad"] += 1
            if state["bad"] >= patience or time.time() > deadline:
                raise lgb.callback.EarlyStopException(it, [("cb", "primary", state["best"], True)])

        callbacks = [] if (fixed or X_val is None) else [cb]
        self.booster = lgb.train(params, data, num_boost_round=int(fixed or max_rounds),
                                 callbacks=callbacks)
        best_iter = state["best_iter"] or self.booster.current_iteration()
        self.best_iter = int(fixed or best_iter)
        self.info.update({"rounds_used": self.best_iter,
                          "best_val_primary": state["best"] if state["best"] > 0 else None,
                          "params": {k: v for k, v in params.items() if k != "verbosity"}})
        return self

    def predict(self, X, groups):
        return self.booster.predict(X, num_iteration=self.best_iter).astype(np.float32)


class LGBMPointwise(_LGBMBase):
    objective = "binary"


class LGBMLambdarank(_LGBMBase):
    objective = "lambdarank"
