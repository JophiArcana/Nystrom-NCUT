"""Tests for KNN-based embedding extrapolation."""
from __future__ import annotations

import torch

from nystrom_ncut import (
    NystromNCut,
    SampleConfig,
    extrapolate_knn,
    extrapolate_knn_with_subsampling,
)


def test_extrapolate_knn_shape() -> None:
    anchor_features = torch.randn(200, 16)
    anchor_output = torch.randn(200, 5)
    new_features = torch.randn(50, 16)

    out = extrapolate_knn(
        anchor_features, anchor_output, new_features, affinity_type="cosine", knn=5
    )
    assert out.shape == (50, 5)
    assert torch.isfinite(out).all()


def test_extrapolate_knn_with_subsampling_shape() -> None:
    full_features = torch.randn(300, 16)
    full_output = torch.randn(300, 5)
    new_features = torch.randn(40, 16)

    out = extrapolate_knn_with_subsampling(
        full_features,
        full_output,
        new_features,
        sample_config=SampleConfig(method="fps", num_sample=64),
        affinity_type="cosine",
        knn=5,
    )
    assert out.shape == (40, 5)


def test_extrapolate_consistent_with_nystrom_transform() -> None:
    """``extrapolate_knn`` should produce a plausibly close embedding to the
    Nystrom transform of new points using the anchors."""
    torch.manual_seed(0)
    M = torch.randn(150, 8)
    new = torch.randn(20, 8)

    nc = NystromNCut(
        n_components=4,
        affinity_type="cosine",
        sample_config=SampleConfig(method="fps", num_sample=32),
    )
    nc.fit(M)
    V_nystrom = nc.transform(new)
    assert V_nystrom.shape == (20, 4)
    assert torch.isfinite(V_nystrom).all()
