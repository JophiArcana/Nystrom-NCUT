from abc import abstractmethod
from typing import Optional, Tuple

import torch

from ..common import (
    ceildiv,
)
from ..global_settings import (
    CHUNK_SIZE,
)
from ..transformer import (
    OnlineTorchTransformerMixin,
)
from ..types import EigSolverOptions

__all__ = [
    "EigSolverOptions",
    "OnlineKernel",
    "OnlineNystrom",
    "solve_eig",
]


class OnlineKernel:
    @abstractmethod
    def fit(self, features: torch.Tensor) -> "OnlineKernel":                # [... x n x d]
        """Fit the kernel to anchor features and return ``self``."""

    def accumulate(self, features: torch.Tensor) -> None:                   # [... x m x d]
        """Update internal running statistics from ``features`` without
        materializing the normalized cross-affinity block.

        Default implementation calls :meth:`update` and discards the return value;
        subclasses should override for a cheaper stats-only path.
        """
        self.update(features)

    @abstractmethod
    def update(self, features: torch.Tensor) -> torch.Tensor:               # [... x m x d] -> [... x m x n]
        """Incrementally update with new features and return the cross-affinity block."""

    @abstractmethod
    def transform(self, features: torch.Tensor = None) -> torch.Tensor:     # [... x m x d] -> [... x m x n]
        """Compute the cross-affinity block between anchors and ``features`` (or anchors)."""


class OnlineNystrom(OnlineTorchTransformerMixin):
    def __init__(
        self,
        n_components: int,
        kernel: OnlineKernel,
        eig_solver: EigSolverOptions,
        low_memory: bool = False,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return
            kernel (OnlineKernel): Online kernel that computes pairwise matrix entries from input features and allows updates
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
            low_memory (bool): if True, :meth:`update` walks chunks an extra
                time at the end instead of caching per-chunk projections. Saves
                ``O(total_m * (d+1))`` memory at the cost of one more pass of
                cross-affinity computation.
            chunk_size (int): per-instance override for the chunk size used in
                :meth:`update` and :meth:`transform`. Defaults to the module-level
                ``CHUNK_SIZE`` constant.
        """
        self.n_components: int = n_components
        self.kernel: OnlineKernel = kernel
        self.eig_solver: EigSolverOptions = eig_solver
        self.low_memory: bool = low_memory
        self.chunk_size: int = chunk_size if chunk_size is not None else CHUNK_SIZE
        self.shape: Optional[torch.Size] = None                 # ...

        # Anchor matrices
        self.anchor_features: Optional[torch.Tensor] = None     # [... x n x d]
        self.A: Optional[torch.Tensor] = None                   # [... x n x n]
        self.Ahinv: Optional[torch.Tensor] = None               # [... x n x n] -- lazy; see _Ahinv()
        self.Ahinv_UL: Optional[torch.Tensor] = None            # [... x n x indirect_pca_dim]
        self.Ahinv_VT: Optional[torch.Tensor] = None            # [... x indirect_pca_dim x n]
        self._kernel_L: Optional[torch.Tensor] = None           # [... x (d + 1)] eigvals from _update_to_kernel

        # Updated matrices
        self.S: Optional[torch.Tensor] = None                   # [... x n x n]
        self.transform_matrix: Optional[torch.Tensor] = None    # [... x n x n_components]
        self.eigenvalues_: Optional[torch.Tensor] = None        # [... x n_components]

    def _update_to_kernel(self, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.A = self.kernel.transform()
        self.S = torch.nan_to_num(self.A, nan=0.0)
        U, L = solve_eig(
            self.S,
            num_eig=d + 1,  # truncated pseudo-inverse rank for the anchor block
            eig_solver=self.eig_solver,
        )                                                                                           # [... x n x (? + 1)], [... x (? + 1)]
        self.Ahinv_UL = U * (L[..., None, :] ** -0.5)                                               # [... x n x (? + 1)]
        self.Ahinv_VT = U.mT                                                                        # [... x (? + 1) x n]
        self._kernel_L = L                                                                          # [... x (? + 1)]
        self.Ahinv = None                                                                           # invalidate any prior materialization
        return U, L

    def fit(self, features: torch.Tensor) -> "OnlineNystrom":
        self.anchor_features = features

        self.kernel.fit(self.anchor_features)
        U, L = self._update_to_kernel(features.shape[-1])                                           # [... x n x (d + 1)], [... x (d + 1)]

        self.transform_matrix = (U / L[..., None, :])[..., :, :self.n_components]                   # [... x n x n_components]
        self.eigenvalues_ = L[..., :self.n_components]                                              # [... x n_components]
        return self

    def update(self, features: torch.Tensor) -> torch.Tensor:
        """Incrementally update the eigendecomposition with new ``features``.

        Two affinity-block passes by default: one cheap ``accumulate`` pass for
        running stats, and one fused projection pass that caches the
        ``Ahinv_VT @ B`` projection so the final embedding only needs a small
        matmul. When ``self.low_memory`` is ``True``, the cache is dropped and
        a third affinity pass produces the final embedding.

        Returns:
            torch.Tensor: spectral embedding of ``features``, shape
            ``(..., m, n_components)``.
        """
        d = features.shape[-1]
        n_chunks = ceildiv(features.shape[-2], self.chunk_size)
        chunks = torch.chunk(features, n_chunks, dim=-2)

        # Pass 1: stats-only accumulation (no normalization, no kernel.transform output materialized).
        for chunk in chunks:
            self.kernel.accumulate(chunk)
        self._update_to_kernel(d)

        # Pass 2: fused projection + (optional) cache of compressed_B per chunk.
        compressed_BBT = 0.0                                                                    # [... x (? + 1) x (? + 1)]
        compressed_B_cache = [] if not self.low_memory else None
        for chunk in chunks:
            _B = self.kernel.transform(chunk).mT                                                # [... x n x _m]
            _compressed_B = torch.nan_to_num(self.Ahinv_VT @ _B, nan=0.0)                       # [... x (? + 1) x _m]
            compressed_BBT = compressed_BBT + _compressed_B @ _compressed_B.mT                  # [... x (? + 1) x (? + 1)]
            if compressed_B_cache is not None:
                compressed_B_cache.append(_compressed_B)

        self.S = self.S + self.Ahinv_UL @ compressed_BBT @ self.Ahinv_UL.mT                     # [... x n x n]
        US, self.eigenvalues_ = solve_eig(self.S, self.n_components, self.eig_solver)           # [... x n x n_components], [... x n_components]
        # transform_matrix = Ahinv @ US * eigval_scale, in factored form to avoid the n*n matmul.
        eigval_scale = self.eigenvalues_[..., None, :] ** -0.5                                  # [... x 1 x n_components]
        self.transform_matrix = self.Ahinv_UL @ (self.Ahinv_VT @ US) * eigval_scale             # [... x n x n_components]

        if compressed_B_cache is not None:
            # B_normalized.mT @ Ahinv_UL = (Ahinv_VT @ B_normalized).mT * L^{-1/2}
            # (since Ahinv_UL = U * L^{-1/2}, Ahinv_VT = U.mT). The cached _compressed_B is
            # exactly Ahinv_VT @ B_normalized, so the per-chunk VS is cheap.
            L_inv_sqrt = (self._kernel_L ** -0.5)[..., :, None]                                 # [... x (?+1) x 1]
            inner = (self.Ahinv_VT @ US) * eigval_scale                                         # [... x (?+1) x n_components]
            VS = [(cB * L_inv_sqrt).mT @ inner for cB in compressed_B_cache]                    # list[... x _m x n_components]
        else:
            VS = [self.kernel.transform(chunk) @ self.transform_matrix for chunk in chunks]     # list[... x _m x n_components]
        return torch.cat(VS, dim=-2)                                                            # [... x m x n_components]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        if features is None:
            VS = self.A @ self.transform_matrix                                                     # [... x n x n_components]
        else:
            n_chunks = ceildiv(features.shape[-2], self.chunk_size)
            if n_chunks > 1:
                """ Chunked version """
                chunks = torch.chunk(features, n_chunks, dim=-2)
                VS = []
                for chunk in chunks:
                    VS.append(self.kernel.transform(chunk) @ self.transform_matrix)                 # [... x _m x n_components]
                VS = torch.cat(VS, dim=-2)
            else:
                """ Unchunked version """
                VS = self.kernel.transform(features) @ self.transform_matrix                        # [... x m x n_components]
        return VS                                                                                   # [... x m x n_components]


def solve_eig(
    A: torch.Tensor,
    num_eig: int,
    eig_solver: EigSolverOptions,
    eig_value_buffer: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-``num_eig`` eigendecomposition with sign-corrected eigenvectors.

    Sorted by ``|eigenvalue|`` in descending order; each eigenvector is sign-
    flipped so that the entry with the largest absolute value is non-negative
    (this resolves the sign ambiguity of eigenvectors so that results are
    reproducible). All supported solvers return real eigenpairs.

    Args:
        A (torch.Tensor): input matrix, shape ``(..., n, n)``.
        num_eig (int): number of eigenvectors to return.
        eig_solver (str): one of ``'svd_lowrank'``, ``'lobpcg'``, ``'svd'``, ``'eigh'``.
        eig_value_buffer (float): a ridge added to the diagonal before decomposition
            and subtracted from the returned eigenvalues afterwards. Mathematically
            sound only when ``A`` is symmetric; intended use is to make a symmetric
            but slightly non-PSD matrix solvable by ``lobpcg``. For SVD solvers
            (``'svd'``, ``'svd_lowrank'``) the returned values are singular values,
            so the subtraction additionally requires ``A + eig_value_buffer * I`` to
            be PSD (i.e. the buffer must dominate any negative eigenvalues of ``A``);
            otherwise the returned values do not correspond to the original
            eigenvalues of ``A``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(eigenvectors, eigenvalues)``,
        real tensors with shapes ``(..., n, num_eig)`` and ``(..., num_eig)``.
    """
    shape: torch.Size = A.shape[:-2]
    A = A.view((-1, *A.shape[-2:]))
    bsz: int = A.shape[0]

    A = A + eig_value_buffer * torch.eye(A.shape[-1], device=A.device)
    num_eig = min(A.shape[-1], num_eig)
    # compute eigenvectors
    if eig_solver == "svd_lowrank":  # default
        # only top q eigenvectors, fastest
        eigen_vector, eigen_value, _ = torch.svd_lowrank(A, q=num_eig)              # real: [(...) x N x D], [(...) x D]
    elif eig_solver == "lobpcg":
        # only top k eigenvectors, fast
        eigen_value, eigen_vector = torch.lobpcg(A, k=num_eig)
    elif eig_solver == "svd":
        # all eigenvectors, slow
        eigen_vector, eigen_value, _ = torch.svd(A)
    elif eig_solver == "eigh":
        # all eigenvectors, slow
        eigen_value, eigen_vector = torch.linalg.eigh(A)
    else:
        raise ValueError(
            "eigen_solver should be 'lobpcg', 'svd_lowrank', 'svd' or 'eigh'"
        )
    eigen_value = eigen_value - eig_value_buffer

    # SVD-family solvers return singular values already sorted descending, so a
    # straight slice is equivalent to the topk-and-gather path used below.
    if eig_solver in ("svd", "svd_lowrank"):
        eigen_value = eigen_value[..., :num_eig]                                    # real: [(...) x S]
        eigen_vector = eigen_vector[..., :, :num_eig]                               # real: [(...) x N x S]
    else:
        # sort eigenvectors by eigenvalues, take top (descending order)
        indices = torch.topk(eigen_value.abs(), k=num_eig, dim=-1).indices          # int: [(...) x S]
        batch_idx = torch.arange(bsz, device=A.device)[:, None]                     # int: [(...) x 1]
        eigen_value = eigen_value[batch_idx, indices]                               # real: [(...) x S]
        eigen_vector = eigen_vector[batch_idx, :, indices].mT                       # real: [(...) x N x S]

    # Resolve the eigenvector sign ambiguity by forcing the entry with the
    # largest absolute value to be non-negative. More stable than the prior
    # sum-of-entries convention, which could flip near zero on small precision
    # differences across platforms.
    abs_max_idx = torch.argmax(eigen_vector.real.abs(), dim=-2, keepdim=True)       # int: [(...) x 1 x S]
    sign = torch.sign(torch.gather(eigen_vector.real, -2, abs_max_idx))             # float: [(...) x 1 x S]
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    eigen_vector = eigen_vector * sign

    eigen_value = eigen_value.view((*shape, *eigen_value.shape[-1:]))               # real: [... x S]
    eigen_vector = eigen_vector.view((*shape, *eigen_vector.shape[-2:]))            # real: [... x N x S]
    return eigen_vector, eigen_value
