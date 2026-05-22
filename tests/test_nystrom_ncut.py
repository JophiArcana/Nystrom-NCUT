"""Tests for :class:`NystromNCut`."""
from __future__ import annotations

import pytest
import torch

from nystrom_ncut import NystromNCut, SampleConfig


@pytest.mark.parametrize("affinity_type", ["cosine", "rbf"])
def test_shapes(two_clusters: torch.Tensor, affinity_type: str) -> None:
    nc = NystromNCut(
        n_components=5,
        affinity_type=affinity_type,
        sample_config=SampleConfig(method="fps", num_sample=64),
    )
    V = nc.fit_transform(two_clusters)
    assert V.shape == (two_clusters.shape[0], 5)
    assert nc.eigenvalues_.shape == (5,)


@pytest.mark.parametrize("method", ["full", "random", "fps"])
def test_sample_methods(two_clusters: torch.Tensor, method: str) -> None:
    nc = NystromNCut(
        n_components=4,
        affinity_type="cosine",
        sample_config=SampleConfig(method=method, num_sample=64),
    )
    V = nc.fit_transform(two_clusters)
    assert torch.isfinite(V).all()


def test_fps_recursive(two_clusters: torch.Tensor) -> None:
    nc = NystromNCut(
        n_components=4,
        affinity_type="cosine",
        sample_config=SampleConfig(method="fps_recursive", num_sample=64, n_iter=2, fps_dim=4),
    )
    V = nc.fit_transform(two_clusters)
    assert V.shape == (two_clusters.shape[0], 4)


def test_eigenvalues_descending(two_clusters: torch.Tensor) -> None:
    nc = NystromNCut(
        n_components=6,
        affinity_type="cosine",
        sample_config=SampleConfig(method="fps", num_sample=64),
    )
    nc.fit(two_clusters)
    eigs = nc.eigenvalues_.abs()
    assert torch.all(eigs[:-1] >= eigs[1:] - 1e-6)


def test_full_method_matches_kernel_full(two_clusters: torch.Tensor) -> None:
    """With method='full' all points are anchors so embedding has finite values."""
    nc = NystromNCut(
        n_components=5,
        affinity_type="cosine",
        sample_config=SampleConfig(method="full"),
    )
    V = nc.fit_transform(two_clusters)
    assert torch.isfinite(V).all()
    assert V.shape == (two_clusters.shape[0], 5)


def test_clusters_separable(two_clusters: torch.Tensor) -> None:
    """The first non-trivial eigenvector should split the two Gaussians."""
    nc = NystromNCut(
        n_components=3,
        affinity_type="rbf",
        sample_config=SampleConfig(method="fps", num_sample=64),
    )
    V = nc.fit_transform(two_clusters)
    half = two_clusters.shape[0] // 2
    labels = (V[:, 1] > V[:, 1].median()).int()
    cluster_a = labels[:half].float().mean().item()
    cluster_b = labels[half:].float().mean().item()
    assert abs(cluster_a - cluster_b) > 0.5


def test_sample_config_not_mutated(two_clusters: torch.Tensor) -> None:
    cfg = SampleConfig(method="fps", num_sample=10_000_000)
    nc = NystromNCut(n_components=3, sample_config=cfg)
    nc.fit_transform(two_clusters)
    assert cfg.num_sample == 10_000_000


def test_precomputed_indices(small_features: torch.Tensor) -> None:
    nc = NystromNCut(
        n_components=3,
        affinity_type="cosine",
        sample_config=SampleConfig(method="fps", num_sample=16),
    )
    indices = torch.arange(16)
    V = nc.fit_transform(small_features, precomputed_sampled_indices=indices)
    assert V.shape == (small_features.shape[0], 3)
