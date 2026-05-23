"""Nystrom-approximated Normalized Cut.

Implements the symmetric Laplacian spectral clustering of
Shi & Malik 2000 (*Normalized Cuts and Image Segmentation*) using the Nystrom
extension of Fowlkes, Belongie, Chung, Malik 2004
(https://people.eecs.berkeley.edu/~malik/papers/FBCM-nystrom.pdf):
only the anchor-anchor block ``A`` and cross-anchor block ``B`` are
materialized, and the full affinity ``W`` is never formed.
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
    AffinityOptions,
    AFFINITY_TO_DISTANCE,
    affinity_from_features,
)
from ..sampling_utils import (
    SampleConfig,
)
from ..transformer import (
    OnlineTransformerSubsampleFit,
)


class LaplacianKernel(OnlineKernel):
    def __init__(
        self,
        affinity_type: AffinityOptions,
        affinity_focal_gamma: float,
        adaptive_scaling: bool,
        eig_solver: EigSolverOptions,
    ):
        self.affinity_type: AffinityOptions = affinity_type
        self.affinity_focal_gamma = affinity_focal_gamma
        self.adaptive_scaling: bool = adaptive_scaling
        self.eig_solver: EigSolverOptions = eig_solver

        # Anchor matrices
        self.anchor_features: Optional[torch.Tensor] = None                         # [... x n x d]
        self.anchor_mask: Optional[torch.Tensor] = None
        self.A: Optional[torch.Tensor] = None                                       # [... x n x n]
        # `Ainv` is kept as a public attribute for back-compat introspection, but
        # is not materialized in `fit`; consumers use `_apply_Ainv` instead, which
        # applies the factored low-rank inverse `_Ainv_UL @ _Ainv_VT @ x`.
        self.Ainv: Optional[torch.Tensor] = None                                    # [... x n x n] (unused)
        self._Ainv_UL: Optional[torch.Tensor] = None                                # [... x n x (d + 1)]
        self._Ainv_VT: Optional[torch.Tensor] = None                                # [... x (d + 1) x n]

        # Updated matrices
        self.a_r: Optional[torch.Tensor] = None                                     # [... x n]
        self.b_r: Optional[torch.Tensor] = None                                     # [... x n]

    def _apply_Ainv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the factored low-rank inverse ``Ainv @ x`` without materializing
        the dense ``n x n`` matrix. Equivalent to ``self.Ainv @ x`` when
        ``self.Ainv = self._Ainv_UL @ self._Ainv_VT`` is the truncated
        pseudo-inverse from the top ``(d + 1)`` eigendirections of ``A``.
        """
        return self._Ainv_UL @ (self._Ainv_VT @ x)

    def fit(self, features: torch.Tensor) -> "LaplacianKernel":
        self.anchor_features = features                                             # [... x n x d]
        self.anchor_mask = torch.all(torch.isnan(self.anchor_features), dim=-1)     # [... x n]

        self.A = torch.where(self.anchor_mask[..., None], 0.0, affinity_from_features(
            features_A=self.anchor_features,                                        # [... x n x d]
            features_B=self.anchor_features,                                        # [... x n x d]
            affinity_type=self.affinity_type,
            affinity_focal_gamma=self.affinity_focal_gamma,
        ))                                                                          # [... x n x n]
        d = features.shape[-1]
        U, L = solve_eig(
            torch.nan_to_num(self.A, nan=0.0),
            num_eig=d + 1,  # truncated pseudo-inverse rank for the anchor block
            eig_solver=self.eig_solver,
        )                                                                                           # [... x n x (d + 1)], [... x (d + 1)]
        L_inv = torch.nan_to_num(1 / L, posinf=0.0, neginf=0.0)                                     # [... x (d + 1)]
        self._Ainv_UL = U * L_inv[..., None, :]                                                     # [... x n x (d + 1)]
        self._Ainv_VT = U.mT                                                                        # [... x (d + 1) x n]
        self.Ainv = None                                                                             # see attribute docstring
        self.a_r = torch.where(self.anchor_mask, torch.inf, torch.sum(self.A, dim=-1))             # [... x n]
        self.b_r = torch.zeros_like(self.a_r)                                                       # [... x n]
        return self

    def _affinity(self, features: torch.Tensor) -> torch.Tensor:
        B = torch.where(self.anchor_mask[..., None], 0.0, affinity_from_features(
            features_A=self.anchor_features,                                        # [... x n x d]
            features_B=features,                                                    # [... x m x d]
            affinity_type=self.affinity_type,
            affinity_focal_gamma=self.affinity_focal_gamma,
        ))                                                                          # [... x n x m]
        if self.adaptive_scaling:
            # diag(B^T Ainv B) computed via a single fused matmul + elementwise sum.
            diagonal = (B * self._apply_Ainv(B)).sum(dim=-2)                        # [... x m]
            adaptive_scale = diagonal ** -0.5                                       # [... x m]
            B = B * adaptive_scale[..., None, :]
        return B                                                                    # [... x n x m]

    def accumulate(self, features: torch.Tensor) -> None:
        """Cheap stats-only path: compute B, add its row sums into ``b_r``."""
        B = self._affinity(features)                                                # [... x n x m]
        self.b_r = self.b_r + torch.sum(torch.nan_to_num(B, nan=0.0), dim=-1)       # [... x n]

    def update(self, features: torch.Tensor) -> torch.Tensor:
        B = self._affinity(features)                                                # [... x n x m]
        b_r = torch.sum(torch.nan_to_num(B, nan=0.0), dim=-1)                       # [... x n]
        b_c = torch.sum(B, dim=-2)                                                  # [... x m]
        self.b_r = self.b_r + b_r                                                   # [... x n]

        row_sum = self.a_r + self.b_r                                               # [... x n]
        col_sum = b_c + (B.mT @ self._apply_Ainv(self.b_r[..., None]))[..., 0]      # [... x m]
        scale = (row_sum[..., :, None] * col_sum[..., None, :]) ** -0.5             # [... x n x m]
        return (B * scale).mT                                                       # [... x m x n]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        row_sum = self.a_r + self.b_r                                               # [... x n]
        if features is None:
            B = self.A                                                              # [... x n x n]
            col_sum = row_sum                                                       # [... x n]
        else:
            B = self._affinity(features)                                            # [... x n x m]
            b_c = torch.sum(B, dim=-2)                                              # [... x m]
            col_sum = b_c + (B.mT @ self._apply_Ainv(self.b_r[..., None]))[..., 0]  # [... x m]
        scale = (row_sum[..., :, None] * col_sum[..., None, :]) ** -0.5             # [... x n x m]
        return (B * scale).mT                                                       # [... x m x n]


class NystromNCut(OnlineTransformerSubsampleFit):
    """Nystrom Normalized Cut for large scale graph."""

    def __init__(
        self,
        n_components: int,
        affinity_type: AffinityOptions = "cosine",
        affinity_focal_gamma: float = 1.0,
        adaptive_scaling: bool = False,
        sample_config: SampleConfig = None,
        eig_solver: EigSolverOptions = "svd_lowrank",
        low_memory: bool = False,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return
            affinity_type (str): distance metric for affinity matrix, ['cosine', 'euclidean', 'rbf'].
            affinity_focal_gamma (float): affinity matrix temperature, lower t reduce the not-so-connected edge weights,
                smaller t result in more sharp eigenvectors.
            adaptive_scaling (bool): whether to scale off-diagonal affinity vectors so extended diagonal equals 1
            sample_config (SampleConfig): subgraph sampling configuration. ``method`` is one of
                ['full', 'random', 'fps', 'fps_recursive']; ``'fps'`` (farthest point sampling) is
                recommended for better Nystrom-approximation accuracy.
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
                kernel=LaplacianKernel(affinity_type, affinity_focal_gamma, adaptive_scaling, eig_solver),
                eig_solver=eig_solver,
                low_memory=low_memory,
                chunk_size=chunk_size,
            ),
            distance_type=AFFINITY_TO_DISTANCE[affinity_type],
            sample_config=SampleConfig() if sample_config is None else sample_config,
        )
