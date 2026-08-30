"""Model registry: name -> (class, default config, default feature spec)."""

from recsys.models.base import Recommender  # noqa: F401
from recsys.models.blend import Blend
from recsys.models.classic import FMRec, PopularityRec, RandomRec
from recsys.models.gbm import LGBMLambdarank, LGBMPointwise
from recsys.models.torch_models import CWM, PLE, DCNv2, DeepFM, DINLite, MMoE

MODELS = {
    "random": (RandomRec, {}, "fm5"),
    "popularity": (PopularityRec, {}, "full"),
    "fm": (FMRec, {}, "fm5"),
    "lgbm_pointwise": (LGBMPointwise, {}, "full"),
    "lgbm_lambdarank": (LGBMLambdarank, {}, "full"),
    "deepfm": (DeepFM, {}, "full"),
    "dcnv2": (DCNv2, {}, "full"),
    "mmoe": (MMoE, {}, "full"),
    "ple": (PLE, {}, "full"),
    "cwm": (CWM, {}, "full"),
    "din_lite": (DINLite, {}, "full_seq"),
    "blend": (Blend, {"models": ["lgbm_lambdarank", "lgbm_pointwise"],
                      "mode": "stack", "folds": 3}, "full"),
}
