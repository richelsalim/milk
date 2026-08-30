"""Recommender base class (contract from IMPLEMENTATION.md phase 3)."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prepare  # noqa: E402


class Recommender:
    """fit/predict/save/load. y is long_view; aux carries the other feedback columns
    plus play_time_ms and duration_ms; groups is user_id per row; validation arrays are
    for early stopping only and are always scored through prepare.evaluate."""

    def __init__(self, meta=None, **config):
        self.meta = meta or {}
        self.config = config
        self.info = {}  # rounds/epochs actually used, subsampling, ... -> config.json

    def fit(self, X_train, y_train, groups_train, aux_train=None,
            X_val=None, y_val=None, groups_val=None, time_budget=300, seed=0):
        raise NotImplementedError

    def predict(self, X, groups) -> np.ndarray:
        raise NotImplementedError

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "model.pkl", "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, out_dir):
        with open(Path(out_dir) / "model.pkl", "rb") as fh:
            return pickle.load(fh)

    # helpers shared by subclasses -------------------------------------------
    def _val_primary(self, scores) -> float:
        """Score candidate validation predictions through the organizer metric."""
        return prepare.evaluate("val", np.asarray(scores, dtype=np.float64),
                                dataset=self.meta.get("dataset", "pure"))["primary"]

    @staticmethod
    def _deadline(time_budget):
        return time.time() + time_budget

    def _col(self, name: str) -> int:
        return self.meta["columns"].index(name)

    def _recency_weights(self, X):
        """Per-row 0.5 ** (age_days / half_life) sample weights, or None when the
        `recency_half_life_days` knob is off. Age is measured back from the newest
        training row; callers normalize to mean 1 after any row selection."""
        hl = (getattr(self, "cfg", None) or self.config).get("recency_half_life_days")
        if not hl:
            return None
        try:
            a = X[:, self._col("ctx_days_since_start")].astype(np.float64)
        except ValueError:
            raise ValueError("recency_half_life_days needs the ctx block "
                             "(ctx_days_since_start) in the feature spec") from None
        return 0.5 ** ((a.max() - a) / float(hl))
