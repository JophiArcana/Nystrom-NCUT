from typing import Literal

import torch
import torch.nn.functional as Fn

from .transformer_mixin import (
    TorchTransformerMixin,
)


class AxisAlign(TorchTransformerMixin):
    """Multiclass Spectral Clustering, SX Yu, J Shi, 2003
    Args:
        max_iter (int, optional): Maximum number of iterations.
    """
    SortOptions = Literal["count", "norm", "marginal_norm"]

    def __init__(
        self,
        sort_method: SortOptions = "norm",
        max_iter: int = 100,
    ):
        self.sort_method: AxisAlign.SortOptions = sort_method
        self.max_iter: int = max_iter

        self.R: torch.Tensor = None

    def fit(self, X: torch.Tensor) -> "AxisAlign":
        # Normalize eigenvectors
        n, d = X.shape
        normalized_X = Fn.normalize(X, p=2, dim=-1)

        # Initialize R matrix with the first column from a random row of EigenVectors
        self.R = torch.empty((d, d), device=X.device)
        self.R[0] = normalized_X[torch.randint(0, n, (), device=X.device)]

        # Loop to populate R with k orthogonal directions
        c = torch.zeros((n,), device=X.device)
        for i in range(1, d):
            c += torch.abs(normalized_X @ self.R[i - 1])
            self.R[i] = normalized_X[torch.argmin(c, dim=0)]

        # Iterative optimization loop
        idx, prev_objective = None, torch.inf
        for _ in range(self.max_iter):
            # Discretize the projected eigenvectors
            idx = torch.argmax(normalized_X @ self.R.mT, dim=-1)
            M = torch.zeros((d, d), device=X.device).index_add_(0, idx, normalized_X)

            # Check for convergence
            objective = torch.norm(M)
            if torch.abs(objective - prev_objective) < torch.finfo(torch.float32).eps:
                break
            prev_objective = objective

            # SVD decomposition to compute the next R
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            self.R = U @ Vh

        # Permute the rotation matrix so the dimensions are sorted in descending cluster significance
        if self.sort_method == "count":
            sort_metric = torch.bincount(idx, minlength=d)
        elif self.sort_method == "norm":
            rotated_X = X @ self.R.mT
            sort_metric = torch.linalg.norm(rotated_X, dim=0)
        elif self.sort_method == "marginal_norm":
            rotated_X = X @ self.R.mT
            sort_metric = torch.zeros((d,), device=X.device).index_add_(0, idx, rotated_X[range(n), idx] ** 2)
        else:
            raise ValueError(f"Invalid sort method {self.sort_method}.")

        self.R = self.R[torch.argsort(sort_metric, dim=0, descending=True)]
        return self

    def transform(self, X: torch.Tensor, normalize: bool = True, hard: bool = False) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): continuous eigenvectors from NCUT, shape (n, k)
            normalize (bool): whether to normalize input features before rotating
            hard (bool): whether to return cluster indices of input features or just the rotated features
        Returns:
            torch.Tensor: Discretized eigenvectors, shape (n, k), each row is a one-hot vector.
        """
        if normalize:
            X = Fn.normalize(X, p=2, dim=1)
        rotated_X = X @ self.R.mT
        return torch.argmax(rotated_X, dim=1) if hard else rotated_X

    def fit_transform(self, X: torch.Tensor, normalize: bool = True, hard: bool = False) -> torch.Tensor:
        return self.fit(X).transform(X, normalize=normalize, hard=hard)
