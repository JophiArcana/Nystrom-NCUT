"""Tests for :class:`KernelNCut`."""
from __future__ import annotations

import pytest
import torch

from nystrom_ncut import KernelNCut, SampleConfig


@pytest.mark.parametrize("affinity_type", ["cosine", "rbf"])
def test_shapes(two_clusters: torch.Tensor, affinity_type: str) -> None:
    kn = KernelNCut(
        n_components=5,
        kernel_dim=128,
        affinity_type=affinity_type,
    )
    V = kn.fit_transform(two_clusters)
    assert V.shape == (two_clusters.shape[0], 5)
    assert kn.eigenvalues_.shape == (5,)


def test_finite_embedding(two_clusters: torch.Tensor) -> None:
    kn = KernelNCut(n_components=4, kernel_dim=64, affinity_type="cosine")
    V = kn.fit_transform(two_clusters)
    assert torch.isfinite(V).all()


def test_clusters_separable(two_clusters: torch.Tensor) -> None:
    kn = KernelNCut(n_components=3, kernel_dim=256, affinity_type="rbf")
    V = kn.fit_transform(two_clusters)
    half = two_clusters.shape[0] // 2
    labels = (V[:, 1] > V[:, 1].median()).int()
    cluster_a = labels[:half].float().mean().item()
    cluster_b = labels[half:].float().mean().item()
    assert abs(cluster_a - cluster_b) > 0.4


def test_eigenvalues_descending(two_clusters: torch.Tensor) -> None:
    kn = KernelNCut(n_components=6, kernel_dim=128, affinity_type="cosine")
    kn.fit(two_clusters)
    eigs = kn.eigenvalues_.abs()
    assert torch.all(eigs[:-1] >= eigs[1:] - 1e-6)


def test_subsampled_consistency(small_features: torch.Tensor) -> None:
    """With method='full' and same seed we should get reproducible output."""
    torch.manual_seed(42)
    kn1 = KernelNCut(n_components=3, kernel_dim=128, affinity_type="cosine",
                     sample_config=SampleConfig(method="full"))
    V1 = kn1.fit_transform(small_features)

    torch.manual_seed(42)
    kn2 = KernelNCut(n_components=3, kernel_dim=128, affinity_type="cosine",
                     sample_config=SampleConfig(method="full"))
    V2 = kn2.fit_transform(small_features)

    assert torch.allclose(V1, V2)
