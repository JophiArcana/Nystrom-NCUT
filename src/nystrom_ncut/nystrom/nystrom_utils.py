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
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return
            kernel (OnlineKernel): Online kernel that computes pairwise matrix entries from input features and allows updates
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
        """
        self.n_components: int = n_components
        self.kernel: OnlineKernel = kernel
        self.eig_solver: EigSolverOptions = eig_solver
        self.shape: Optional[torch.Size] = None                 # ...

        # Anchor matrices
        self.anchor_features: Optional[torch.Tensor] = None     # [... x n x d]
        self.A: Optional[torch.Tensor] = None                   # [... x n x n]
        self.Ahinv: Optional[torch.Tensor] = None               # [... x n x n]
        self.Ahinv_UL: Optional[torch.Tensor] = None            # [... x n x indirect_pca_dim]
        self.Ahinv_VT: Optional[torch.Tensor] = None            # [... x indirect_pca_dim x n]

        # Updated matrices
        self.S: Optional[torch.Tensor] = None                   # [... x n x n]
        self.transform_matrix: Optional[torch.Tensor] = None    # [... x n x n_components]
        self.eigenvalues_: Optional[torch.Tensor] = None        # [... x n_components]

    def _update_to_kernel(self, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.A = self.kernel.transform()
        self.S = torch.nan_to_num(self.A, nan=0.0)
        U, L = solve_eig(
            self.S,
            num_eig=d + 1,  # d * (d + 3) // 2 + 1,
            eig_solver=self.eig_solver,
        )                                                                                           # [... x n x (? + 1)], [... x (? + 1)]
        self.Ahinv_UL = U * (L[..., None, :] ** -0.5)                                               # [... x n x (? + 1)]
        self.Ahinv_VT = U.mT                                                                        # [... x (? + 1) x n]
        self.Ahinv = self.Ahinv_UL @ self.Ahinv_VT                                                  # [... x n x n]
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

        The chunked accumulation of ``B B^T`` is mathematically identical to the
        unchunked path; chunking only bounds peak memory at the cost of an extra
        pass over ``features`` to compute the cross-affinity transforms.

        Returns:
            torch.Tensor: spectral embedding of ``features``, shape
            ``(..., m, n_components)``.
        """
        d = features.shape[-1]
        n_chunks = ceildiv(features.shape[-2], CHUNK_SIZE)
        chunks = torch.chunk(features, n_chunks, dim=-2)

        for chunk in chunks:
            self.kernel.update(chunk)
        self._update_to_kernel(d)

        compressed_BBT = 0.0                                                                    # [... x (? + 1) x (? + 1)]
        for chunk in chunks:
            _B = self.kernel.transform(chunk).mT                                                # [... x n x _m]
            _compressed_B = torch.nan_to_num(self.Ahinv_VT @ _B, nan=0.0)                       # [... x (? + 1) x _m]
            compressed_BBT = compressed_BBT + _compressed_B @ _compressed_B.mT                  # [... x (? + 1) x (? + 1)]
        self.S = self.S + self.Ahinv_UL @ compressed_BBT @ self.Ahinv_UL.mT                     # [... x n x n]
        US, self.eigenvalues_ = solve_eig(self.S, self.n_components, self.eig_solver)           # [... x n x n_components], [... x n_components]
        self.transform_matrix = self.Ahinv @ US * (self.eigenvalues_[..., None, :] ** -0.5)     # [... x n x n_components]

        VS = [self.kernel.transform(chunk) @ self.transform_matrix for chunk in chunks]         # list[... x _m x n_components]
        return torch.cat(VS, dim=-2)                                                            # [... x m x n_components]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        if features is None:
            VS = self.A @ self.transform_matrix                                                     # [... x n x n_components]
        else:
            n_chunks = ceildiv(features.shape[-2], CHUNK_SIZE)
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
    flipped so that the sum of its real entries is non-negative (this resolves
    the sign ambiguity of eigenvectors so that results are reproducible).

    Args:
        A (torch.Tensor): input matrix, shape ``(..., n, n)``.
        num_eig (int): number of eigenvectors to return.
        eig_solver (str): one of ``'svd_lowrank'``, ``'lobpcg'``, ``'svd'``, ``'eigh'``.
        eig_value_buffer (float): a ridge added to the diagonal before decomposition
            and subtracted from the returned eigenvalues afterwards. This is only
            mathematically sound when ``A`` is symmetric; intended use is to make
            a symmetric but slightly non-PSD matrix solvable by ``lobpcg``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(eigenvectors, eigenvalues)``,
        shapes ``(..., n, num_eig)`` and ``(..., num_eig)``.
    """
    shape: torch.Size = A.shape[:-2]
    A = A.view((-1, *A.shape[-2:]))
    bsz: int = A.shape[0]

    A = A + eig_value_buffer * torch.eye(A.shape[-1], device=A.device)
    num_eig = min(A.shape[-1], num_eig)
    # compute eigenvectors
    if eig_solver == "svd_lowrank":  # default
        # only top q eigenvectors, fastest
        eigen_vector, eigen_value, _ = torch.svd_lowrank(A, q=num_eig)              # complex: [(...) x N x D], [(...) x D]
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

    # sort eigenvectors by eigenvalues, take top (descending order)
    indices = torch.topk(eigen_value.abs(), k=num_eig, dim=-1).indices              # int: [(...) x S]
    eigen_value = eigen_value[torch.arange(bsz)[:, None], indices]                  # complex: [(...) x S]
    eigen_vector = eigen_vector[torch.arange(bsz)[:, None], :, indices].mT          # complex: [(...) x N x S]

    # correct the random rotation (flipping sign) of eigenvectors
    sign = torch.sign(torch.sum(eigen_vector.real, dim=-2, keepdim=True))           # float: [(...) x 1 x S]
    sign[sign == 0] = 1.0
    eigen_vector = eigen_vector * sign

    eigen_value = eigen_value.view((*shape, *eigen_value.shape[-1:]))               # complex: [... x S]
    eigen_vector = eigen_vector.view((*shape, *eigen_vector.shape[-2:]))            # complex: [... x N x S]
    return eigen_vector, eigen_value
