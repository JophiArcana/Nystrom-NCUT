"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    """Make every test deterministic."""
    torch.manual_seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def two_clusters() -> torch.Tensor:
    """Two well-separated Gaussian clusters in R^4."""
    n, d = 256, 4
    M = torch.randn(n, d)
    M[: n // 2] += 5.0
    M[n // 2 :] -= 5.0
    return M


@pytest.fixture
def small_features() -> torch.Tensor:
    """Small feature matrix used for spot-checks."""
    return torch.randn(80, 6)
