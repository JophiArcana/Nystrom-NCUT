"""Tests for :func:`subsample_features` and :class:`SampleConfig`."""
from __future__ import annotations

import pytest
import torch

from nystrom_ncut import SampleConfig, subsample_features


@pytest.mark.parametrize("method", ["full", "random", "fps"])
def test_subsample_basic(method: str) -> None:
    M = torch.randn(200, 8)
    cfg = SampleConfig(method=method, num_sample=32)
    indices = subsample_features(M, distance_type="euclidean", config=cfg)
    expected_n = 200 if method == "full" else 32
    assert indices.shape[-1] == expected_n
    assert indices.unique().numel() == expected_n


def test_subsample_indices_sorted() -> None:
    M = torch.randn(200, 8)
    indices = subsample_features(
        M, distance_type="euclidean", config=SampleConfig(method="fps", num_sample=16)
    )
    assert torch.all(indices[:-1] <= indices[1:])


def test_fps_recursive_requires_recursive_obj() -> None:
    M = torch.randn(200, 8)
    cfg = SampleConfig(method="fps_recursive", num_sample=16, n_iter=2, fps_dim=4)
    with pytest.raises(ValueError, match="fps_recursive"):
        subsample_features(M, distance_type="euclidean", config=cfg)


def test_unknown_method_raises() -> None:
    M = torch.randn(200, 8)
    cfg = SampleConfig(method="not_a_method", num_sample=16)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        subsample_features(M, distance_type="euclidean", config=cfg)
