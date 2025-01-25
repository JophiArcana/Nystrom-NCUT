import random

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
    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter
        self.R: torch.Tensor = None

    def fit(self, X: torch.Tensor) -> "AxisAlign":
        # Normalize eigenvectors
        n, d = X.shape
        X = Fn.normalize(X, p=2, dim=-1)

        # Initialize R matrix with the first column from a random row of EigenVectors
        self.R = torch.empty((d, d), device=X.device)
        self.R[0] = X[random.randint(0, n - 1)]

        # Loop to populate R with k orthogonal directions
        c = torch.zeros((n,), device=X.device)
        for i in range(1, d):
            c += torch.abs(X @ self.R[i - 1])
            self.R[i] = X[torch.argmin(c, dim=0)]

        # Iterative optimization loop
        idx, prev_objective = None, torch.inf
        for _ in range(self.max_iter):
            # Discretize the projected eigenvectors
            idx = torch.argmax(X @ self.R.mT, dim=-1)
            M = torch.zeros((d, d)).index_add_(0, idx, X)

            # Check for convergence
            objective = torch.norm(M)
            if torch.abs(objective - prev_objective) < torch.finfo(torch.float32).eps:
                break
            prev_objective = objective

            # SVD decomposition to compute the next R
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            self.R = U @ Vh

        # Permute the rotation matrix so the dimensions are sorted in descending cluster counts
        self.R = self.R[torch.argsort(torch.bincount(idx, minlength=d), dim=0, descending=True)]
        return self

    def transform(self, X: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): continuous eigenvectors from NCUT, shape (n, k)
            hard (bool): whether to return cluster indices of input features or just the rotated features
        Returns:
            torch.Tensor: Discretized eigenvectors, shape (n, k), each row is a one-hot vector.
        """
        rotated_X = X @ self.R.mT
        return torch.argmax(rotated_X, dim=1) if hard else rotated_X

    def fit_transform(self, X: torch.Tensor, hard: bool = False) -> torch.Tensor:
        return self.fit(X).transform(X, hard=hard)
