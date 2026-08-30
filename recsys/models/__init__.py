"""Model registry: name -> (class, default config, default feature spec)."""

from recsys.models.base import Recommender  # noqa: F401
from recsys.models.classic import FMRec, PopularityRec, RandomRec

MODELS = {
    "random": (RandomRec, {}, "fm5"),
    "popularity": (PopularityRec, {}, "full"),
    "fm": (FMRec, {}, "fm5"),
}
