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
