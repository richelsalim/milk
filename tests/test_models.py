"""Every registered model on the 20k fixture: fits within budget, predicts finite
float32 of the right length, save/load round-trips to identical predictions."""

import time

import numpy as np
import pytest

import prepare
from recsys.features import build
from recsys.models import MODELS
from recsys.zoo import make_aux

FIXTURE = str(prepare.REPO / "data" / "cache" / "fixture_small")


@pytest.fixture(autouse=True)
def _fixture_root(monkeypatch):
    monkeypatch.setenv("KUAIRAND_DATA_ROOT", FIXTURE)


def _inputs(spec):
    Xtr, meta, gtr = build(spec, "train")
    Xva, _, gva = build(spec, "val")
    train_frame = prepare.load("train")
    ytr = train_frame["long_view"].to_numpy().astype(np.float32)
    yva = prepare.load("val")["long_view"].to_numpy().astype(np.float32)
    return Xtr, meta, gtr, Xva, gva, ytr, yva, make_aux(train_frame)


@pytest.mark.parametrize("name", list(MODELS))
def test_model_contract(name, tmp_path):
    cls, cfg, spec = MODELS[name]
    Xtr, meta, gtr, Xva, gva, ytr, yva, aux = _inputs(spec)
    model = cls(meta=meta, **cfg)
    t0 = time.time()
    model.fit(Xtr, ytr, gtr, aux_train=aux, X_val=Xva, y_val=yva, groups_val=gva,
              time_budget=10, seed=0)
    took = time.time() - t0
    assert took < 20, f"{name} fit took {took:.1f}s with time_budget=10"

    scores = model.predict(Xva, gva)
    assert scores.dtype == np.float32 and len(scores) == len(gva)
    assert np.isfinite(scores).all()

    model.save(tmp_path)
    loaded = type(model).load(tmp_path)
    np.testing.assert_array_equal(loaded.predict(Xva, gva), scores)


@pytest.mark.parametrize("loss", ["listwise", "bpr", "mixed"])
def test_ranking_losses_train(loss):
    cls, cfg, spec = MODELS["dcnv2"]
    Xtr, meta, gtr, Xva, gva, ytr, yva, aux = _inputs(spec)
    model = cls(meta=meta, **{**cfg, "loss": loss, "epochs": 2})
    model.fit(Xtr, ytr, gtr, aux_train=aux, X_val=Xva, y_val=yva, groups_val=gva,
              time_budget=8, seed=0)
    assert np.isfinite(model.predict(Xva, gva)).all()
