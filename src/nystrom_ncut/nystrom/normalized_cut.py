import einops
import torch

from .nystrom_utils import (
    EigSolverOptions,
    OnlineKernel,
    OnlineNystromSubsampleFit,
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


class LaplacianKernel(OnlineKernel):
    def __init__(
        self,
        affinity_focal_gamma: float,
        affinity_type: AffinityOptions,
        adaptive_scaling: bool,
        eig_solver: EigSolverOptions,
    ):
        self.affinity_focal_gamma = affinity_focal_gamma
        self.affinity_type: AffinityOptions = affinity_type
        self.adaptive_scaling: bool = adaptive_scaling
        self.eig_solver: EigSolverOptions = eig_solver

        # Anchor matrices
        self.anchor_features: torch.Tensor = None                                   # [... x n x d]
        self.anchor_mask: torch.Tensor = None
        self.A: torch.Tensor = None                                                 # [... x n x n]
        self.Ainv: torch.Tensor = None                                              # [... x n x n]

        # Updated matrices
        self.a_r: torch.Tensor = None                                               # [... x n]
        self.b_r: torch.Tensor = None                                               # [... x n]

    def fit(self, features: torch.Tensor) -> None:
        self.anchor_features = features                                             # [... x n x d]
        self.anchor_mask = torch.all(torch.isnan(self.anchor_features), dim=-1)     # [... x n]

        self.A = torch.nan_to_num(affinity_from_features(
            self.anchor_features,                                                   # [... x n x d]
            affinity_focal_gamma=self.affinity_focal_gamma,
            affinity_type=self.affinity_type,
        ), nan=0.0)                                                                 # [... x n x n]
        d = features.shape[-1]
        U, L = solve_eig(
            self.A,
            num_eig=d + 1,  # d * (d + 3) // 2 + 1,
            eig_solver=self.eig_solver,
        )                                                                           # [... x n x (d + 1)], [... x (d + 1)]
        self.Ainv = U @ torch.diag_embed(1 / L) @ U.mT                              # [... x n x n]
        self.a_r = torch.where(self.anchor_mask, torch.inf, torch.sum(self.A, dim=-1))  # [... x n]
        self.b_r = torch.zeros_like(self.a_r)                                       # [... x n]

    def _affinity(self, features: torch.Tensor) -> torch.Tensor:
        B = torch.where(self.anchor_mask[..., None], 0.0, affinity_from_features(
            self.anchor_features,                                                   # [... x n x d]
            features,                                                               # [... x m x d]
            affinity_focal_gamma=self.affinity_focal_gamma,
            affinity_type=self.affinity_type,
        ))                                                                          # [... x n x m]
        if self.adaptive_scaling:
            diagonal = (
                einops.rearrange(B, "... n m -> ... m 1 n")                         # [... x m x 1 x n]
                @ self.Ainv                                                         # [... x n x n]
                @ einops.rearrange(B, "... n m -> ... m n 1")                       # [... x m x n x 1]
            ).squeeze(-2, -1)                                                       # [... x m]
            adaptive_scale = diagonal ** -0.5                                       # [... x m]
            B = B * adaptive_scale[..., None, :]
        return B                                                                    # [... x n x m]

    def update(self, features: torch.Tensor) -> torch.Tensor:
        B = self._affinity(features)                                                # [... x n x m]
        b_r = torch.sum(torch.nan_to_num(B, nan=0.0), dim=-1)                       # [... x n]
        b_c = torch.sum(B, dim=-2)                                                  # [... x m]
        self.b_r = self.b_r + b_r                                                   # [... x n]

        row_sum = self.a_r + self.b_r                                               # [... x n]
        col_sum = b_c + (B.mT @ (self.Ainv @ self.b_r[..., None]))[..., 0]          # [... x m]
        scale = (row_sum[..., :, None] * col_sum[..., None, :]) ** -0.5             # [... x n x m]
        return (B * scale).mT                                                       # [... x m x n]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        row_sum = self.a_r + self.b_r                                               # [... x n]
        if features is None:
            B = self.A                                                              # [... x n x n]
            col_sum = row_sum                                                       # [... x n]
        else:
            B = self._affinity(features)
            b_c = torch.sum(B, dim=-2)                                              # [... x m]
            col_sum = b_c + (B.mT @ (self.Ainv @ self.b_r[..., None]))[..., 0]      # [... x m]
        scale = (row_sum[..., :, None] * col_sum[..., None, :]) ** -0.5             # [... x n x m]
        return (B * scale).mT                                                       # [... x m x n]


class NCut(OnlineNystromSubsampleFit):
    """Nystrom Normalized Cut for large scale graph."""

    def __init__(
        self,
        n_components: int = 100,
        affinity_focal_gamma: float = 1.0,
        affinity_type: AffinityOptions = "cosine",
        adaptive_scaling: bool = False,
        sample_config: SampleConfig = SampleConfig(),
        eig_solver: EigSolverOptions = "svd_lowrank",
        chunk_size: int = 8192,
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return
            affinity_focal_gamma (float): affinity matrix temperature, lower t reduce the not-so-connected edge weights,
                smaller t result in more sharp eigenvectors.
            distance (str): distance metric for affinity matrix, ['cosine', 'euclidean', 'rbf'].
            adaptive_scaling (bool): whether to scale off-diagonal affinity vectors so extended diagonal equals 1
            sample_config (str): subgraph sampling, ['farthest', 'random'].
                farthest point sampling is recommended for better Nystrom-approximation accuracy
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
            chunk_size (int): chunk size for large-scale matrix multiplication
        """
        OnlineNystromSubsampleFit.__init__(
            self,
            n_components=n_components,
            kernel=LaplacianKernel(affinity_focal_gamma, affinity_type, adaptive_scaling, eig_solver),
            distance_type=AFFINITY_TO_DISTANCE[affinity_type],
            sample_config=sample_config,
            eig_solver=eig_solver,
            chunk_size=chunk_size,
        )
