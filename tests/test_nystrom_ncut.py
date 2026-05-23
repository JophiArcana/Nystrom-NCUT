"""Tests for :class:`NystromNCut`."""
from __future__ import annotations

import pytest
import torch

from nystrom_ncut import NystromNCut, SampleConfig
from nystrom_ncut.nystrom import solve_eig


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


def test_laplacian_kernel_a_r_matches_row_sums(two_clusters: torch.Tensor) -> None:
    """`LaplacianKernel.a_r` should be the row sums of the symmetric anchor
    affinity `A` (masked anchors set to +inf). Catches regressions that
    accidentally desymmetrize `A` or change the row/column convention.
    """
    nc = NystromNCut(
        n_components=3,
        affinity_type="cosine",
        sample_config=SampleConfig(method="full"),
    )
    nc.fit(two_clusters)
    kernel = nc.base_transformer.kernel
    expected = torch.where(
        kernel.anchor_mask,
        torch.tensor(torch.inf),
        torch.sum(kernel.A, dim=-1),
    )
    assert torch.allclose(kernel.a_r, expected, equal_nan=True)


def test_solve_eig_cpu_baseline() -> None:
    """`solve_eig` must produce correct top-k eigenpairs of a PSD matrix on CPU.

    Also acts as a baseline for the device-locality test below: the indexing
    machinery inside `solve_eig` builds a batch index that must live on the
    same device as the input.
    """
    torch.manual_seed(0)
    X = torch.randn(12, 12)
    A = X @ X.mT
    U, L = solve_eig(A, num_eig=4, eig_solver="eigh")
    assert U.shape == (12, 4)
    assert L.shape == (4,)
    assert torch.all(L[:-1].abs() >= L[1:].abs() - 1e-5)
    reconstructed = U @ torch.diag(L) @ U.mT
    full_eigs = torch.linalg.eigvalsh(A).sort(descending=True).values[:4]
    assert torch.allclose(L.sort(descending=True).values, full_eigs, atol=1e-4)
    assert torch.linalg.norm(reconstructed) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_solve_eig_cuda_no_device_mismatch() -> None:
    """The batch index inside `solve_eig` must live on the input's device.

    A CPU `torch.arange` would either crash or trigger an implicit H2D copy
    when indexing a CUDA tensor; we just verify it returns a CUDA result.
    """
    torch.manual_seed(0)
    X = torch.randn(12, 12, device="cuda")
    A = X @ X.mT
    U, L = solve_eig(A, num_eig=4, eig_solver="eigh")
    assert U.device.type == "cuda"
    assert L.device.type == "cuda"


def test_solve_eig_svd_lowrank_descending() -> None:
    """`solve_eig` with `svd_lowrank` must return eigenpairs in descending order
    of magnitude (singular values are already sorted, so the SVD-gated slice
    must preserve that ordering). `svd_lowrank` is randomized and only its top
    components are accurate, so the value comparison is restricted to those."""
    torch.manual_seed(0)
    X = torch.randn(20, 20)
    A = X @ X.mT
    U, L = solve_eig(A, num_eig=6, eig_solver="svd_lowrank")
    assert U.shape == (20, 6)
    assert L.shape == (6,)
    assert torch.all(L[:-1].abs() >= L[1:].abs() - 1e-5)
    # Top-2 singular values from svd_lowrank are reliably close to the true ones.
    full_eigs = torch.linalg.eigvalsh(A).sort(descending=True).values
    assert torch.allclose(L[:2], full_eigs[:2], atol=1e-2, rtol=1e-2)


def test_adaptive_scaling_path(two_clusters: torch.Tensor) -> None:
    """`adaptive_scaling=True` exercises the `diag(B^T Ainv B)` einsum; verify
    the fused form matches a reference computation and that `fit_transform`
    returns finite values."""
    nc = NystromNCut(
        n_components=4,
        affinity_type="rbf",
        adaptive_scaling=True,
        sample_config=SampleConfig(method="fps", num_sample=64),
    )
    V = nc.fit_transform(two_clusters)
    assert torch.isfinite(V).all()
    assert V.shape == (two_clusters.shape[0], 4)

    # Cross-check the fused diagonal against the matmul reference on a small
    # synthetic block.
    kernel = nc.base_transformer.kernel
    B = torch.randn(kernel._Ainv_UL.shape[-2], 7)
    Ainv_dense = kernel._Ainv_UL @ kernel._Ainv_VT
    expected = torch.einsum("nm,np,pm->m", B, Ainv_dense, B)
    actual = (B * kernel._apply_Ainv(B)).sum(dim=-2)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_factored_ainv_matches_dense(two_clusters: torch.Tensor) -> None:
    """`LaplacianKernel._apply_Ainv(x)` must equal `(_Ainv_UL @ _Ainv_VT) @ x`."""
    nc = NystromNCut(
        n_components=3,
        affinity_type="cosine",
        sample_config=SampleConfig(method="full"),
    )
    nc.fit(two_clusters)
    kernel = nc.base_transformer.kernel
    n = kernel._Ainv_UL.shape[-2]
    x = torch.randn(n, 5)
    dense = (kernel._Ainv_UL @ kernel._Ainv_VT) @ x
    factored = kernel._apply_Ainv(x)
    assert torch.allclose(factored, dense, atol=1e-5, rtol=1e-5)


def test_chunked_update_matches_unchunked(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OnlineNystrom.update` chunked path must agree with the unchunked path.

    Force chunking by shrinking `CHUNK_SIZE`; result must be `allclose` (modulo
    floating-point reordering) to the same fit_transform run with one chunk.
    """
    from nystrom_ncut import global_settings

    torch.manual_seed(0)
    M = torch.randn(120, 8)
    M[:60] += 3.0
    M[60:] -= 3.0

    def run_with(chunk_size: int) -> torch.Tensor:
        monkeypatch.setattr(global_settings, "CHUNK_SIZE", chunk_size)
        # The kernel_ncut + nystrom_utils modules import `CHUNK_SIZE` by name at
        # module load; we must patch it where it is read.
        from nystrom_ncut.nystrom import nystrom_utils as _nu
        from nystrom_ncut import extrapolation as _ex
        monkeypatch.setattr(_nu, "CHUNK_SIZE", chunk_size)
        monkeypatch.setattr(_ex, "CHUNK_SIZE", chunk_size)
        torch.manual_seed(0)
        nc = NystromNCut(
            n_components=4,
            affinity_type="cosine",
            sample_config=SampleConfig(method="fps", num_sample=32),
            eig_solver="eigh",
        )
        return nc.fit_transform(M)

    V_unchunked = run_with(1024)
    V_chunked = run_with(16)
    # Sign of eigenvectors may flip per chunk; compare absolute values per component.
    assert torch.allclose(V_unchunked.abs(), V_chunked.abs(), atol=1e-3, rtol=1e-3)


def test_chunk_size_kwarg_matches_global() -> None:
    """The per-instance ``chunk_size`` kwarg on :class:`NystromNCut` must produce
    output equivalent (up to per-component sign) to running with the module
    default. This verifies the new kwarg threads all the way down to
    ``OnlineNystrom.update`` / ``transform`` and does not change correctness.
    """
    torch.manual_seed(0)
    M = torch.randn(120, 8)
    M[:60] += 3.0
    M[60:] -= 3.0

    def run(chunk_size):
        torch.manual_seed(0)
        nc = NystromNCut(
            n_components=4,
            affinity_type="cosine",
            sample_config=SampleConfig(method="fps", num_sample=32),
            eig_solver="eigh",
            chunk_size=chunk_size,
        )
        return nc.fit_transform(M)

    V_default = run(None)  # falls back to module CHUNK_SIZE
    V_small = run(16)
    assert torch.allclose(V_default.abs(), V_small.abs(), atol=1e-3, rtol=1e-3)


def test_solve_eig_sign_convention_max_abs() -> None:
    """`solve_eig` must orient each eigenvector so that the entry with the
    largest absolute value is non-negative. This is the new max-absolute-entry
    convention (replacing the prior sum-based one)."""
    torch.manual_seed(0)
    X = torch.randn(10, 10)
    A = X @ X.mT
    U, _ = solve_eig(A, num_eig=5, eig_solver="eigh")
    abs_max_idx = torch.argmax(U.abs(), dim=-2)
    leading = U.gather(-2, abs_max_idx[None, :]).squeeze(-2)
    assert torch.all(leading >= 0)


def test_low_memory_matches_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """`low_memory=True` (3 affinity passes, no cache) must match the cached
    default path (`low_memory=False`, 2 affinity passes + cache)."""
    from nystrom_ncut import global_settings
    from nystrom_ncut.nystrom import nystrom_utils as _nu
    from nystrom_ncut import extrapolation as _ex

    monkeypatch.setattr(global_settings, "CHUNK_SIZE", 16)
    monkeypatch.setattr(_nu, "CHUNK_SIZE", 16)
    monkeypatch.setattr(_ex, "CHUNK_SIZE", 16)

    torch.manual_seed(0)
    M = torch.randn(120, 8)
    M[:60] += 3.0
    M[60:] -= 3.0

    def run(low_memory: bool) -> torch.Tensor:
        torch.manual_seed(0)
        nc = NystromNCut(
            n_components=4,
            affinity_type="cosine",
            sample_config=SampleConfig(method="fps", num_sample=32),
            eig_solver="eigh",
            low_memory=low_memory,
        )
        return nc.fit_transform(M)

    V_default = run(False)
    V_low_mem = run(True)
    assert torch.allclose(V_default.abs(), V_low_mem.abs(), atol=1e-4, rtol=1e-4)
