"""V2.0 checks: snapshot selection reuses cached per-epoch val predictions (no extra
forward passes), recency sample weights actually alter training (torch + lgbm), and
determinism holds with both knobs on."""

import numpy as np
import pytest

import prepare
from recsys.features import build
from recsys.models import MODELS
from recsys.models.torch_models import _TorchRec
from recsys.zoo import make_aux

FIXTURE = str(prepare.REPO / "data" / "cache" / "fixture_small")


@pytest.fixture(autouse=True)
def _fixture_root(monkeypatch):
    monkeypatch.setenv("KUAIRAND_DATA_ROOT", FIXTURE)


@pytest.fixture
def inputs(_fixture_root):
    Xtr, meta, gtr = build("full", "train")
    Xva, _, gva = build("full", "val")
    train_frame = prepare.load("train")
    ytr = train_frame["long_view"].to_numpy().astype(np.float32)
    yva = prepare.load("val")["long_view"].to_numpy().astype(np.float32)
    return Xtr, meta, gtr, Xva, gva, ytr, yva, make_aux(train_frame)


def _fit_ple(inputs, monkeypatch=None, **cfg):
    Xtr, meta, gtr, Xva, gva, ytr, yva, aux = inputs
    cls, base_cfg, _ = MODELS["ple"]
    model = cls(meta=meta, **{**base_cfg, "epochs": 3, "patience": 99, **cfg})
    model.fit(Xtr, ytr, gtr, aux_train=aux, X_val=Xva, y_val=yva, groups_val=gva,
              time_budget=30, seed=0)
    return model, (Xva, gva)


def test_snapshot_selection_uses_cache_no_extra_forward(inputs, monkeypatch):
    calls = {"n": 0}
    orig = _TorchRec._predict_raw

    def counting(self, X):
        calls["n"] += 1
        return orig(self, X)

    monkeypatch.setattr(_TorchRec, "_predict_raw", counting)
    model, _ = _fit_ple(inputs, snapshot_k=3)
    used = model.info["rounds_used"]
    # exactly one forward pass per epoch: snapshot selection and the multitask head
    # grid both run off the cached vectors
    assert calls["n"] == used, f"{calls['n']} forward passes for {used} epochs"
    ens = model.info["snapshot_ensemble"]
    assert ens["k"] == min(3, used) and isinstance(ens["beat_single"], bool)
    assert ens["primary"] >= ens["single_best"] or not ens["beat_single"]


def test_recency_weights_change_torch_training(inputs):
    base, (Xva, gva) = _fit_ple(inputs, snapshot_k=0, head_grid=False)
    weighted, _ = _fit_ple(inputs, snapshot_k=0, head_grid=False, recency_half_life_days=3)
    assert weighted.info["recency_half_life_days"] == 3
    assert not np.allclose(base.predict(Xva, gva), weighted.predict(Xva, gva))


def test_recency_weight_math(inputs):
    Xtr, meta = inputs[0], inputs[1]
    cls, cfg, _ = MODELS["ple"]
    m = cls(meta=meta, **{**cfg, "recency_half_life_days": 7})
    m.cfg = {**m.defaults, **m.config}
    w = m._recency_weights(Xtr)
    days = Xtr[:, meta["columns"].index("ctx_days_since_start")].astype(np.float64)
    assert w is not None and w.max() == 1.0
    np.testing.assert_allclose(w, 0.5 ** ((days.max() - days) / 7.0))


def test_recency_weights_change_lgbm_training(inputs):
    Xtr, meta, gtr, Xva, gva, ytr, yva, aux = inputs
    cls, _, _ = MODELS["lgbm_pointwise"]
    preds = []
    for hl in (None, 3):
        m = cls(meta=meta, rounds=30, recency_half_life_days=hl)
        m.fit(Xtr, ytr, gtr, aux_train=aux, X_val=None, time_budget=30, seed=0)
        preds.append(m.predict(Xva, gva))
    assert not np.allclose(preds[0], preds[1])


def test_blend_weight_grid_fields_seed_offset(inputs):
    from recsys.models import SNAP
    from recsys.models.blend import Blend
    Xtr, meta, gtr, Xva, gva, ytr, yva, aux = inputs
    grid = [(0.2, 0.4, 0.4), (0.0, 0.5, 0.5), (0.34, 0.33, 0.33)]
    m = Blend(meta=meta, mode="rank_avg", bases=[
        {"model": "fm", "cfg": {"fields": ["id_user", "id_video", "id_author",
                                           "id_tag", "id_music"]},
         "share": 0.2, "weight": 0.2},
        {"model": "ple", "cfg": {**SNAP, "epochs": 2}, "share": 0.4, "weight": 0.4},
        {"model": "ple", "cfg": {**SNAP, "epochs": 2}, "share": 0.4, "weight": 0.4,
         "seed_offset": 1},
    ], weight_grid=grid)
    m.fit(Xtr, ytr, gtr, aux_train=aux, X_val=Xva, y_val=yva, groups_val=gva,
          time_budget=20, seed=0)
    assert tuple(m.info["weight_grid"]["chosen"]) in grid
    np.testing.assert_allclose(m.weights, m.info["weight_grid"]["chosen"])
    # the two ple bases really differ (seed offset) and fm ran on the id subset
    p1 = m.models[1][1].predict(Xva, gva)
    p2 = m.models[2][1].predict(Xva, gva)
    assert not np.allclose(p1, p2)
    assert len(m.models[0][1].cols_) == 5
    assert np.isfinite(m.predict(Xva, gva)).all()


def test_determinism_with_v2_knobs(inputs):
    a, (Xva, gva) = _fit_ple(inputs, snapshot_k=3, recency_half_life_days=3)
    b, _ = _fit_ple(inputs, snapshot_k=3, recency_half_life_days=3)
    np.testing.assert_array_equal(a.predict(Xva, gva), b.predict(Xva, gva))
