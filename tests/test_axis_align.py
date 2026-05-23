"""Tests for :class:`AxisAlign`."""
from __future__ import annotations

import pytest
import torch

from nystrom_ncut import AxisAlign


@pytest.mark.parametrize("sort_method", ["count", "norm", "marginal_norm"])
def test_soft_output_shape(sort_method: str) -> None:
    eigvecs = torch.randn(100, 5)
    aa = AxisAlign(sort_method=sort_method)
    rotated = aa.fit_transform(eigvecs, hard=False)
    assert rotated.shape == eigvecs.shape


@pytest.mark.parametrize("sort_method", ["count", "norm", "marginal_norm"])
def test_hard_output_shape(sort_method: str) -> None:
    eigvecs = torch.randn(100, 5)
    aa = AxisAlign(sort_method=sort_method)
    labels = aa.fit_transform(eigvecs, hard=True)
    assert labels.shape == (100,)
    assert labels.dtype == torch.long
    assert labels.min() >= 0
    assert labels.max() < 5


def test_rotation_is_orthogonal() -> None:
    """The recovered rotation matrix ``R`` should be (approximately) orthogonal."""
    torch.manual_seed(0)
    eigvecs = torch.randn(200, 4)
    aa = AxisAlign(sort_method="norm")
    aa.fit(eigvecs)
    RR = aa.R @ aa.R.mT
    eye = torch.eye(4)
    assert torch.allclose(RR, eye, atol=1e-4)


def test_clusters_in_basis() -> None:
    """A near-one-hot embedding should yield the natural labels."""
    n_per, k = 30, 4
    base = torch.eye(k)
    eigvecs = torch.cat([base[i].expand(n_per, k) for i in range(k)], dim=0)
    eigvecs = eigvecs + 0.01 * torch.randn_like(eigvecs)

    aa = AxisAlign(sort_method="count")
    labels = aa.fit_transform(eigvecs, hard=True)
    # Every cluster should map to a unique label internally consistent.
    for i in range(k):
        block = labels[i * n_per : (i + 1) * n_per]
        assert (block == block[0]).all()


def test_axis_align_init_unique_anchor_rows() -> None:
    """The init loop in :class:`AxisAlign.fit` must pick distinct row indices
    even when the data sits in a lower-dimensional subspace (causing the
    orthogonal-direction argmin to tie). The previous implementation would
    re-pick the smallest already-picked index on a tie, leaving ``R`` with
    duplicate rows. We skip the iterative SVD loop (``max_iter=0``) so the
    test is checking the init pass directly.
    """
    # Pathological 4-row input on a 2D plane: with d=4 the orthogonal-direction
    # search must exhaust fresh directions on the third pick.
    X = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
    ])
    aa = AxisAlign(sort_method="norm", max_iter=0)
    aa.fit(X)
    d = aa.R.shape[-2]
    for i in range(d):
        for j in range(i + 1, d):
            assert not torch.equal(aa.R[i], aa.R[j]), (
                f"Init picked the same row twice: R[{i}] == R[{j}]"
            )
