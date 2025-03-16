from typing import Literal

import torch
import torch.nn.functional as Fn

from ..common import (
    default_device,
)
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
        with default_device(X.device):
            d = X.shape[-1]
            normalized_X = Fn.normalize(X, p=2, dim=-1)                                                         # float: [... x n x d]

            # Initialize R matrix with the first column from a random row of EigenVectors
            def get_idx(idx: torch.Tensor) -> torch.Tensor:
                return torch.gather(normalized_X, -2, idx[..., None, None].expand([-1] * (X.ndim - 2) + [1, d]))[..., 0, :]

            self.R = torch.empty((*X.shape[:-2], d, d))                                                         # float: [... x d x d]
            mask = torch.all(torch.isfinite(normalized_X), dim=-1)                                              # bool: [... x n]
            start_idx = torch.argmax(mask.to(torch.float) + torch.rand(mask.shape), dim=-1)                     # int: [...]
            self.R[..., 0, :] = get_idx(start_idx)

            # Loop to populate R with k orthogonal directions
            c = torch.zeros(X.shape[:-1])                                                                       # float: [... x n]
            for i in range(1, d):
                c += torch.abs(normalized_X @ self.R[..., i - 1, :, None])[..., 0]
                self.R[..., i, :] = get_idx(torch.argmin(c.nan_to_num(nan=torch.inf), dim=-1))

            # Iterative optimization loop
            normalized_X = torch.nan_to_num(normalized_X, nan=0.0)
            idx, prev_objective = None, torch.inf
            for _ in range(self.max_iter):
                # Discretize the projected eigenvectors
                idx = torch.argmax(normalized_X @ self.R.mT, dim=-1)                                                    # int: [... x n]
                M = torch.sum((idx[..., None] == torch.arange(d))[..., None] * normalized_X[..., :, None, :], dim=-3)   # float: [... x d x d]

                # Check for convergence
                objective = torch.norm(M)
                if torch.abs(objective - prev_objective) < torch.finfo(torch.float32).eps:
                    break
                prev_objective = objective

                # SVD decomposition to compute the next R
                U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                self.R = U @ Vh

            # Permute the rotation matrix so the dimensions are sorted in descending cluster significance
            match self.sort_method:
                case "count":
                    sort_metric = torch.sum((idx[..., None] == torch.arange(d)), dim=-2)
                case "norm":
                    rotated_X = torch.nan_to_num(X @ self.R.mT, nan=0.0)
                    sort_metric = torch.linalg.norm(rotated_X, dim=-2)
                case "marginal_norm":
                    rotated_X = torch.nan_to_num(X @ self.R.mT, nan=0.0)
                    sort_metric = torch.sum((idx[..., None] == torch.arange(d)) * (torch.gather(rotated_X, -1, idx[..., None]) ** 2), dim=-2)
                case _:
                    raise ValueError(f"Invalid sort method {self.sort_method}.")

            order = torch.argsort(sort_metric, dim=-1, descending=True)
            self.R = torch.gather(self.R, -2, order[..., None].expand([-1] * order.ndim + [d]))
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
            X = Fn.normalize(X, p=2, dim=-1)
        rotated_X = X @ self.R.mT
        return torch.argmax(rotated_X, dim=-1) if hard else rotated_X

    def fit_transform(self, X: torch.Tensor, normalize: bool = True, hard: bool = False) -> torch.Tensor:
        return self.fit(X).transform(X, normalize=normalize, hard=hard)
