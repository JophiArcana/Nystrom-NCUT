import torch

from .nystrom_utils import (
    EigSolverOptions,
    OnlineKernel,
    OnlineNystromSubsampleFit,
    solve_eig,
)
from ..distance_utils import (
    DistanceOptions,
    distance_from_features,
)
from ..sampling_utils import (
    SampleConfig,
)


class GramKernel(OnlineKernel):
    def __init__(
        self,
        distance: DistanceOptions,
        eig_solver: EigSolverOptions,
    ):
        self.distance: DistanceOptions = distance
        self.eig_solver: EigSolverOptions = eig_solver

        # Anchor matrices
        self.anchor_features: torch.Tensor = None               # [n x d]
        self.A: torch.Tensor = None                             # [n x n]
        self.Ainv: torch.Tensor = None                          # [n x n]

        # Updated matrices
        self.a_r: torch.Tensor = None                           # [n]
        self.b_r: torch.Tensor = None                           # [n]
        self.matrix_sum: torch.Tensor = torch.zeros(())         # []
        self.n_features: int = None                             # N

    def fit(self, features: torch.Tensor) -> None:
        self.anchor_features = features                         # [n x d]
        self.A = -0.5 * distance_from_features(
            self.anchor_features,                               # [n x d]
            self.anchor_features,
            distance=self.distance,
        )                                                       # [n x n]
        d = features.shape[-1]
        U, L = solve_eig(
            self.A,
            num_eig=d + 1,  # d * (d + 3) // 2 + 1,
            eig_solver=self.eig_solver,
        )                                                       # [n x (d + 1)], [d + 1]
        self.Ainv = U @ torch.diag(1 / L) @ U.mT                # [n x n]
        self.a_r = torch.sum(self.A, dim=-1)                    # [n]
        self.b_r = torch.zeros_like(self.a_r)                   # [n]
        self.matrix_sum = torch.sum(self.a_r)                   # []
        self.n_features = features.shape[0]                     # n

    def update(self, features: torch.Tensor) -> torch.Tensor:
        B = -0.5 * distance_from_features(
            self.anchor_features,                               # [n x d]
            features,                                           # [m x d]
            distance=self.distance,
        )                                                       # [n x m]
        b_r = torch.sum(B, dim=-1)                              # [n]
        b_c = torch.sum(B, dim=-2)                              # [m]
        self.b_r = self.b_r + b_r                               # [n]
        self.matrix_sum = (
            torch.sum(self.a_r)
            + 2 * torch.sum(self.b_r)
            + self.Ainv @ self.b_r @ self.b_r
        )                                                       # []
        self.n_features += features.shape[0]                    # N

        row_sum = self.a_r + self.b_r                           # [n]
        col_sum = b_c + B.mT @ self.Ainv @ self.b_r             # [m]
        shift = -(row_sum[:, None] + col_sum) / self.n_features + self.matrix_sum / (self.n_features ** 2)  # [n x m]
        return (B + shift).mT                                   # [m x n]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        row_sum = self.a_r + self.b_r
        if features is None:
            B = self.A                                          # [n x n]
            col_sum = row_sum                                   # [n]
        else:
            B = -0.5 * distance_from_features(
                self.anchor_features,
                features,
                distance=self.distance,
            )
            b_c = torch.sum(B, dim=-2)                          # [m]
            col_sum = b_c + B.mT @ self.Ainv @ self.b_r         # [m]
        shift = -(row_sum[:, None] + col_sum) / self.n_features + self.matrix_sum / (self.n_features ** 2)  # [n x m]
        return (B + shift).mT                                   # [m x n]


class DistanceRealization(OnlineNystromSubsampleFit):
    """Nystrom Distance Realization for large scale graph."""

    def __init__(
        self,
        n_components: int = 100,
        distance: DistanceOptions = "cosine",
        sample_config: SampleConfig = SampleConfig(),
        eig_solver: EigSolverOptions = "svd_lowrank",
        chunk_size: int = 8192,
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return
            sample_config (str): subgraph sampling, ['farthest', 'random'].
                farthest point sampling is recommended for better Nystrom-approximation accuracy
            distance (str): distance metric for affinity matrix, ['cosine', 'euclidean', 'rbf'].
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
            chunk_size (int): chunk size for large-scale matrix multiplication
        """
        OnlineNystromSubsampleFit.__init__(
            self,
            n_components=n_components,
            kernel=GramKernel(distance, eig_solver),
            distance=distance,
            sample_config=sample_config,
            eig_solver=eig_solver,
            chunk_size=chunk_size,
        )
        self.distance: DistanceOptions = distance

    def fit_transform(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        V, L = OnlineNystromSubsampleFit.fit_transform(self, features, precomputed_sampled_indices)
        return V * (L ** 0.5)

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        V, L = OnlineNystromSubsampleFit.transform(self, features)
        return V * (L ** 0.5)
