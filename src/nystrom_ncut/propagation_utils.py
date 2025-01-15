import logging

import numpy as np
import torch
import torch.nn.functional as Fn

from .common import (
    DistanceOptions,
    SampleOptions,
    ceildiv,
    lazy_normalize,
)


@torch.no_grad()
def run_subgraph_sampling(
    features: torch.Tensor,
    num_sample: int,
    max_draw: int = 1000000,
    sample_method: SampleOptions = "farthest",
):
    if num_sample >= features.shape[0]:
        # if too many samples, use all samples and bypass Nystrom-like approximation
        logging.info(
            "num_sample is larger than total, bypass Nystrom-like approximation"
        )
        sampled_indices = torch.arange(features.shape[0])
    else:
        # sample subgraph
        if sample_method == "farthest":  # default
            if num_sample > max_draw:
                logging.warning(
                    f"num_sample is larger than max_draw, apply farthest point sampling on random sampled {max_draw} samples"
                )
                draw_indices = torch.randperm(features.shape[0])[:max_draw]
                sampled_indices = farthest_point_sampling(
                    features[draw_indices].detach(),
                    num_sample=num_sample,
                )
                sampled_indices = draw_indices[sampled_indices]
            else:
                sampled_indices = farthest_point_sampling(
                    features.detach(),
                    num_sample=num_sample,
                )
        elif sample_method == "random":  # not recommended
            sampled_indices = torch.randperm(features.shape[0])[:num_sample]
        else:
            raise ValueError("sample_method should be 'farthest' or 'random'")
    return sampled_indices.to(features.device)


def farthest_point_sampling(
    features: torch.Tensor,
    num_sample: int = 300,
    h: int = 9,
):
    try:
        import fpsample
    except ImportError:
        raise ImportError(
            "fpsample import failed, please install `pip install fpsample`"
        )

    # PCA to reduce the dimension
    if features.shape[1] > 8:
        u, s, v = torch.pca_lowrank(features, q=8)
        features = u @ torch.diag(s)

    h = min(h, int(np.log2(features.shape[0])))

    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(
        features.numpy(force=True), num_sample, h
    ).astype(np.int64)
    return torch.from_numpy(kdline_fps_samples_idx)


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
        stds = torch.quantile(features, q=torch.tensor((0.158655, 0.841345), device=features.device), dim=0)
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


def extrapolate_knn(
    anchor_features: torch.Tensor,          # [n x d]
    anchor_output: torch.Tensor,            # [n x d']
    extrapolation_features: torch.Tensor,   # [m x d]
    knn: int = 10,                          # k
    distance: DistanceOptions = "cosine",
    affinity_focal_gamma: float = 1.0,
    chunk_size: int = 8192,
    device: str = None,
    move_output_to_cpu: bool = False
) -> torch.Tensor:                          # [m x d']
    """A generic function to propagate new nodes using KNN.

    Args:
        anchor_features (torch.Tensor): features from subgraph, shape (num_sample, n_features)
        anchor_output (torch.Tensor): output from subgraph, shape (num_sample, D)
        extrapolation_features (torch.Tensor): features from existing nodes, shape (new_num_samples, n_features)
        knn (int): number of KNN to propagate eige nvectors
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'
        chunk_size (int): chunk size for matrix multiplication
        device (str): device to use for computation, if None, will not change device
    Returns:
        torch.Tensor: propagated eigenvectors, shape (new_num_samples, D)

    Examples:
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> old_features = torch.randn(3000, 100)
        >>> new_features = torch.randn(200, 100)
        >>> new_eigenvectors = extrapolate_knn(old_features,old_eigenvectors,new_features,knn=3)
        >>> # new_eigenvectors.shape = (200, 20)

    """
    device = anchor_output.device if device is None else device

    # used in nystrom_ncut
    # propagate eigen_vector from subgraph to full graph
    anchor_output = anchor_output.to(device)

    n_chunks = ceildiv(extrapolation_features.shape[0], chunk_size)
    V_list = []
    for _v in torch.chunk(extrapolation_features, n_chunks, dim=0):
        _v = _v.to(device)                                                                              # [_m x d]

        _A = affinity_from_features(anchor_features, _v, affinity_focal_gamma, distance).mT             # [_m x n]
        if knn is not None:
            _A, indices = _A.topk(k=knn, dim=-1, largest=True)                                          # [_m x k], [_m x k]
            _anchor_output = anchor_output[indices]                                                     # [_m x k x d]
        else:
            _anchor_output = anchor_output[None]                                                        # [1 x n x d]

        _A = Fn.normalize(_A, p=1, dim=-1)                                                              # [_m x k]
        _V = (_A[:, None, :] @ _anchor_output).squeeze(1)                                               # [_m x d]

        if move_output_to_cpu:
            _V = _V.cpu()
        V_list.append(_V)

    anchor_output = torch.cat(V_list, dim=0)
    return anchor_output


# wrapper functions for adding new nodes to existing graph
def extrapolate_knn_with_subsampling(
    full_features: torch.Tensor,
    full_output: torch.Tensor,
    extrapolation_features: torch.Tensor,
    knn: int,
    num_sample: int,
    sample_method: SampleOptions,
    chunk_size: int,
    device: str
):
    """Propagate eigenvectors to new nodes using KNN. Note: this is equivalent to the class API `NCUT.tranform(new_features)`, expect for the sampling is re-done in this function.
    Args:
        full_output (torch.Tensor): eigenvectors from existing nodes, shape (num_sample, num_eig)
        full_features (torch.Tensor): features from existing nodes, shape (n_samples, n_features)
        extrapolation_features (torch.Tensor): features from new nodes, shape (n_new_samples, n_features)
        knn (int): number of KNN to propagate eigenvectors, default 3
        num_sample (int): number of samples for subgraph sampling, default 50000
        sample_method (str): sample method, 'farthest' (default) or 'random'
        chunk_size (int): chunk size for matrix multiplication, default 8192
        device (str): device to use for computation, if None, will not change device
    Returns:
        torch.Tensor: propagated eigenvectors, shape (n_new_samples, num_eig)

    Examples:
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> old_features = torch.randn(3000, 100)
        >>> new_features = torch.randn(200, 100)
        >>> new_eigenvectors = extrapolate_knn_with_subsampling(extrapolation_features,old_eigenvectors,old_features,knn=3,num_sample=,sample_method=,chunk_size=,device=)
        >>> # new_eigenvectors.shape = (200, 20)
    """

    device = full_output.device if device is None else device

    # sample subgraph
    anchor_indices = run_subgraph_sampling(
        full_features,
        num_sample,
        sample_method=sample_method,
    )

    anchor_output = full_output[anchor_indices].to(device)
    anchor_features = full_features[anchor_indices].to(device)
    extrapolation_features = extrapolation_features.to(device)

    # propagate eigenvectors from subgraph to new nodes
    new_eigenvectors = extrapolate_knn(
        anchor_features,
        anchor_output,
        extrapolation_features,
        knn=knn,
        chunk_size=chunk_size,
        device=device
    )
    return new_eigenvectors
