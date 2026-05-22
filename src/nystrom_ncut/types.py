"""Shared ``Literal`` type aliases for the public API."""
from typing import Literal


DistanceOptions = Literal[
    "cosine",
    "euclidean",
]

AffinityOptions = Literal[
    "cosine",
    "rbf",
]

EigSolverOptions = Literal[
    "svd_lowrank",
    "lobpcg",
    "svd",
    "eigh",
]

SampleOptions = Literal[
    "full",
    "random",
    "fps",
    "fps_recursive",
]

AxisAlignSortOptions = Literal[
    "count",
    "norm",
    "marginal_norm",
]
