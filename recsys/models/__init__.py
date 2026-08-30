"""Model registry: name -> (class, default config, default feature spec).

Torch defaults train on the full 1.14M-row window (subsample=None): the initial
500k default capped every torch rung ~0.004 below its full-data score
(reports/decisions.md, phase 3). blend's default is the budget-feasible diverse
pair that clears the zoo gate: deepfm (converges ~120 s) + ple, weighted
rank-average, both on the `full` spec.
"""

from recsys.models.base import Recommender  # noqa: F401
from recsys.models.blend import Blend
from recsys.models.classic import FMRec, PopularityRec, RandomRec
from recsys.models.gbm import LGBMLambdarank, LGBMPointwise
from recsys.models.torch_models import CWM, PLE, DCNv2, DeepFM, DINLite, MMoE

FULL_DATA = {"patience": 3, "epochs": 60, "subsample": None}
SNAP = {**FULL_DATA, "snapshot_k": 3}  # v2.0: budget-free snapshot ensembling default

MODELS = {
    "random": (RandomRec, {}, "fm5"),
    "popularity": (PopularityRec, {}, "full"),
    "fm": (FMRec, {}, "fm5"),
    "lgbm_pointwise": (LGBMPointwise, {}, "full"),
    "lgbm_lambdarank": (LGBMLambdarank, {}, "full"),
    "deepfm": (DeepFM, dict(SNAP), "full"),
    "dcnv2": (DCNv2, dict(SNAP), "full"),
    "mmoe": (MMoE, dict(SNAP), "full"),
    "ple": (PLE, dict(SNAP), "full"),
    "cwm": (CWM, dict(FULL_DATA), "full"),
    "din_lite": (DINLite, dict(SNAP), "full_seq"),
    "blend": (Blend, {"mode": "rank_avg", "bases": [
        {"model": "deepfm", "cfg": dict(SNAP), "share": 0.45, "weight": 0.35},
        {"model": "ple", "cfg": dict(SNAP), "share": 0.55, "weight": 0.65},
    ]}, "full"),
}
