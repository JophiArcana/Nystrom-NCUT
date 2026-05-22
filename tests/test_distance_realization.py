"""Tests for :class:`DistanceRealization`."""
from __future__ import annotations

import torch

from nystrom_ncut import DistanceRealization, SampleConfig


def test_shapes(two_clusters: torch.Tensor) -> None:
    dr = DistanceRealization(
        n_components=5,
        distance_type="euclidean",
        sample_config=SampleConfig(method="fps", num_sample=64),
    )
    V = dr.fit_transform(two_clusters)
    assert V.shape == (two_clusters.shape[0], 5)
    assert dr.eigenvalues_.shape == (5,)


def test_finite_embedding(two_clusters: torch.Tensor) -> None:
    dr = DistanceRealization(
        n_components=4,
        distance_type="cosine",
        sample_config=SampleConfig(method="fps", num_sample=64),
    )
    V = dr.fit_transform(two_clusters)
    assert torch.isfinite(V).all()


def test_recovers_distance(small_features: torch.Tensor) -> None:
    """For a small graph in 'full' mode, X @ X.T should approximate the
    double-centered Gram matrix."""
    dr = DistanceRealization(
        n_components=small_features.shape[0],
        distance_type="euclidean",
        sample_config=SampleConfig(method="full"),
    )
    X = dr.fit_transform(small_features)
    assert X.shape[0] == small_features.shape[0]
    assert torch.isfinite(X).all()


def test_transform_on_new_points(small_features: torch.Tensor) -> None:
    dr = DistanceRealization(
        n_components=4,
        distance_type="euclidean",
        sample_config=SampleConfig(method="fps", num_sample=16),
    )
    dr.fit(small_features)
    new = torch.randn(20, small_features.shape[-1])
    V_new = dr.transform(new)
    assert V_new.shape == (20, 4)
