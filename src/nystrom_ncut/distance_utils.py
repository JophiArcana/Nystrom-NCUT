from typing import Literal

import torch

from .common import lazy_normalize


DistanceOptions = Literal["cosine", "euclidean", "rbf"]


def to_euclidean(x: torch.Tensor, disttype: DistanceOptions) -> torch.Tensor:
    if disttype == "cosine":
        return lazy_normalize(x, p=2, dim=-1)
    elif disttype == "rbf":
        return x
    else:
        raise ValueError(f"to_euclidean not implemented for disttype {disttype}.")


def distance_from_features(
    features: torch.Tensor,
    features_B: torch.Tensor,
    distance: DistanceOptions,
):
    """Compute affinity matrix from input features.
    Args:
        features (torch.Tensor): input features, shape (n_samples, n_features)
        features_B (torch.Tensor, optional): optional, if not None, compute affinity between two features
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'.
    Returns:
        (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
    """
    # compute distance matrix from input features
    if distance == "cosine":
        features = lazy_normalize(features, dim=-1)
        features_B = lazy_normalize(features_B, dim=-1)
        D = 1 - features @ features_B.T
    elif distance == "euclidean":
        D = torch.cdist(features, features_B, p=2)
    elif distance == "rbf":
        D = torch.cdist(features, features_B, p=2) ** 2

        # Outlier-robust scale invariance using quantiles to estimate standard deviation
        c = 2.0
        p = torch.erf(torch.tensor((-c, c), device=features.device) * (2 ** -0.5))
        stds = torch.quantile(features, q=(p + 1) / 2, dim=0)
        stds = (stds[1] - stds[0]) / 2
        D = D / (2 * torch.linalg.norm(stds) ** 2)
    else:
        raise ValueError("distance should be 'cosine' or 'euclidean', 'rbf'")
    return D


def affinity_from_features(
    features: torch.Tensor,
    features_B: torch.Tensor = None,
    affinity_focal_gamma: float = 1.0,
    distance: DistanceOptions = "cosine",
):
    """Compute affinity matrix from input features.

    Args:
        features (torch.Tensor): input features, shape (n_samples, n_features)
        features_B (torch.Tensor, optional): optional, if not None, compute affinity between two features
        affinity_focal_gamma (float): affinity matrix parameter, lower t reduce the edge weights
            on weak connections, default 1.0
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'.
    Returns:
        (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
    """
    # compute affinity matrix from input features

    # if feature_B is not provided, compute affinity matrix on features x features
    # if feature_B is provided, compute affinity matrix on features x feature_B
    features_B = features if features_B is None else features_B

    # compute distance matrix from input features
    D = distance_from_features(features, features_B, distance)

    # torch.exp make affinity matrix positive definite,
    # lower affinity_focal_gamma reduce the weak edge weights
    A = torch.exp(-D / affinity_focal_gamma)
    return A
