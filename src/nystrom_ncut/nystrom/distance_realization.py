"""Distance Realization (classical MDS) via Nystrom approximation.

Given a distance matrix ``D``, this computes an embedding ``X`` such that
``X X^T`` approximates the double-centered Gram matrix ``G = -0.5 J D^2 J``
(Borg & Groenen, *Modern Multidimensional Scaling*, 2005). The Nystrom trick
avoids ever forming ``D`` in full: only the anchor block ``A`` and cross
block ``B`` are materialized.
"""
from typing import Optional

import torch

from .nystrom_utils import (
    EigSolverOptions,
    OnlineKernel,
    OnlineNystrom,
    solve_eig,
)
from ..distance_utils import (
    DistanceOptions,
    distance_from_features,
)
from ..sampling_utils import (
    SampleConfig,
)
from ..transformer import (
    OnlineTransformerSubsampleFit,
)


class GramKernel(OnlineKernel):
    """Online double-centered Gram kernel used by :class:`DistanceRealization`."""

    def __init__(
        self,
        distance_type: DistanceOptions,
        eig_solver: EigSolverOptions,
    ):
        self.distance_type: DistanceOptions = distance_type
        self.eig_solver: EigSolverOptions = eig_solver

        # Anchor matrices
        self.anchor_features: Optional[torch.Tensor] = None             # [... x n x d]
        self.A: Optional[torch.Tensor] = None                           # [... x n x n]
        self.Ainv: Optional[torch.Tensor] = None                        # [... x n x n]

        # Running statistics
        self.a_r: Optional[torch.Tensor] = None                         # [... x n]
        self.b_r: Optional[torch.Tensor] = None                         # [... x n]
        self.matrix_sum: Optional[torch.Tensor] = None                  # [...]
        self.n_features: int = 0                                        # N

    def fit(self, features: torch.Tensor) -> "GramKernel":
        self.anchor_features = features                                 # [... x n x d]
        self.A = -0.5 * distance_from_features(
            self.anchor_features,
            self.anchor_features,
            distance_type=self.distance_type,
        )                                                               # [... x n x n]
        d = features.shape[-1]
        U, L = solve_eig(
            self.A,
            num_eig=d + 1,
            eig_solver=self.eig_solver,
        )                                                               # [... x n x (d + 1)], [... x (d + 1)]
        self.Ainv = U @ torch.nan_to_num(torch.diag_embed(1 / L), posinf=0.0, neginf=0.0) @ U.mT    # [... x n x n]
        self.a_r = torch.sum(self.A, dim=-1)                            # [... x n]
        self.b_r = torch.zeros_like(self.a_r)                           # [... x n]
        self.matrix_sum = torch.sum(self.a_r, dim=-1)                   # [...]
        self.n_features = features.shape[-2]                            # n
        return self

    def _refresh_matrix_sum(self) -> None:
        # matrix_sum = sum(A) + 2 sum(b_r) + b_r^T Ainv b_r
        Ainv_br = (self.Ainv @ self.b_r[..., None])[..., 0]             # [... x n]
        self.matrix_sum = (
            torch.sum(self.a_r, dim=-1)                                 # [...]
            + 2 * torch.sum(self.b_r, dim=-1)                           # [...]
            + torch.sum(Ainv_br * self.b_r, dim=-1)                     # [...]
        )

    def accumulate(self, features: torch.Tensor) -> None:
        """Cheap stats-only path: accumulate ``b_r``, ``n_features``, ``matrix_sum``."""
        B = -0.5 * distance_from_features(
            self.anchor_features,
            features,
            distance_type=self.distance_type,
        )                                                               # [... x n x m]
        self.b_r = self.b_r + torch.sum(B, dim=-1)                      # [... x n]
        self.n_features += features.shape[-2]                           # N
        self._refresh_matrix_sum()

    def update(self, features: torch.Tensor) -> torch.Tensor:
        B = -0.5 * distance_from_features(
            self.anchor_features,
            features,
            distance_type=self.distance_type,
        )                                                               # [... x n x m]
        b_r = torch.sum(B, dim=-1)                                      # [... x n]
        b_c = torch.sum(B, dim=-2)                                      # [... x m]
        self.b_r = self.b_r + b_r                                       # [... x n]
        self.n_features += features.shape[-2]                           # N
        self._refresh_matrix_sum()

        row_sum = self.a_r + self.b_r                                   # [... x n]
        col_sum = b_c + (B.mT @ (self.Ainv @ self.b_r[..., None]))[..., 0]      # [... x m]
        shift = (
            -(row_sum[..., :, None] + col_sum[..., None, :]) / self.n_features
            + self.matrix_sum[..., None, None] / (self.n_features ** 2)
        )                                                                       # [... x n x m]
        return (B + shift).mT                                                   # [... x m x n]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        row_sum = self.a_r + self.b_r                                           # [... x n]
        if features is None:
            B = self.A                                                          # [... x n x n]
            col_sum = row_sum                                                   # [... x n]
        else:
            B = -0.5 * distance_from_features(
                self.anchor_features,
                features,
                distance_type=self.distance_type,
            )                                                                   # [... x n x m]
            b_c = torch.sum(B, dim=-2)                                          # [... x m]
            col_sum = b_c + (B.mT @ (self.Ainv @ self.b_r[..., None]))[..., 0]  # [... x m]
        shift = (
            -(row_sum[..., :, None] + col_sum[..., None, :]) / self.n_features
            + self.matrix_sum[..., None, None] / (self.n_features ** 2)
        )                                                                       # [... x n x m]
        return (B + shift).mT                                                   # [... x m x n]


class DistanceRealization(OnlineTransformerSubsampleFit):
    """Nystrom Distance Realization for large scale graphs.

    Produces an embedding ``X`` whose pairwise inner products approximate the
    double-centered Gram of the input distance matrix, up to a global rotation.
    """

    def __init__(
        self,
        n_components: int = 100,
        distance_type: DistanceOptions = "cosine",
        sample_config: SampleConfig = None,
        eig_solver: EigSolverOptions = "svd_lowrank",
        low_memory: bool = False,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return.
            distance_type (str): distance metric for the Gram kernel, ['cosine', 'euclidean'].
            sample_config (SampleConfig): subgraph sampling configuration.
            eig_solver (str): eigen decomposition solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
            low_memory (bool): if True, the chunked update path trades an extra
                pass of cross-affinity computation for ``O(total_m * (d+1))``
                less memory. Defaults to False (faster, higher memory).
            chunk_size (int): per-instance override for the chunk size used in
                the update/transform loops. Defaults to the module-level
                ``CHUNK_SIZE`` constant.
        """
        OnlineTransformerSubsampleFit.__init__(
            self,
            base_transformer=OnlineNystrom(
                n_components=n_components,
                kernel=GramKernel(distance_type, eig_solver),
                eig_solver=eig_solver,
                low_memory=low_memory,
                chunk_size=chunk_size,
            ),
            distance_type=distance_type,
            sample_config=SampleConfig() if sample_config is None else sample_config,
        )

    def fit_transform(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        V = OnlineTransformerSubsampleFit.fit_transform(self, features, precomputed_sampled_indices)
        return V * (self.eigenvalues_[..., None, :] ** 0.5)

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        V = OnlineTransformerSubsampleFit.transform(self, features)
        return V * (self.eigenvalues_[..., None, :] ** 0.5)
