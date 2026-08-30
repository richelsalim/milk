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
