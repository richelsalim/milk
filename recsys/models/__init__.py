"""Model registry: name -> (class, default config, default feature spec)."""

from recsys.models.base import Recommender  # noqa: F401
from recsys.models.classic import FMRec, PopularityRec, RandomRec
from recsys.models.gbm import LGBMLambdarank, LGBMPointwise
from recsys.models.torch_models import PLE, DCNv2, DeepFM, MMoE

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
                      "mode": "stack", "folds": 3}, "full"),
}
