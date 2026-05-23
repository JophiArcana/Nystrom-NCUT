"""KNN-based extrapolation of spectral embeddings to new points."""
from typing import Optional

import torch
import torch.nn.functional as Fn

from .common import (
    ceildiv,
)
from .distance_utils import (
    AFFINITY_TO_DISTANCE,
    affinity_from_features,
)
from .global_settings import (
    CHUNK_SIZE,
)
from .sampling_utils import (
    SampleConfig,
    subsample_features,
)
from .types import AffinityOptions


__all__ = [
    "extrapolate_knn",
    "extrapolate_knn_with_subsampling",
]


def extrapolate_knn(
    anchor_features: torch.Tensor,                  # [n x d]
    anchor_output: torch.Tensor,                    # [n x d']
    extrapolation_features: torch.Tensor,           # [m x d]
    affinity_type: AffinityOptions,
    knn: Optional[int] = 10,
    affinity_focal_gamma: float = 1.0,
    device: Optional[str] = None,
    move_output_to_cpu: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:                                  # [m x d']
    """Propagate ``anchor_output`` to ``extrapolation_features`` via KNN.

    Args:
        anchor_features (torch.Tensor): features from the subgraph, shape ``(n, d)``.
        anchor_output (torch.Tensor): output on the subgraph, shape ``(n, d')``.
        extrapolation_features (torch.Tensor): features for the new nodes, shape ``(m, d)``.
        affinity_type (str): one of ``'cosine'`` or ``'rbf'``.
        knn (int): number of nearest anchors used for each new node. If ``None``,
            uses a soft (un-truncated) interpolation across all anchors.
        affinity_focal_gamma (float): affinity temperature.
        device (str): device to use; defaults to ``anchor_output.device``.
        move_output_to_cpu (bool): if ``True``, moves each chunk back to CPU.
        chunk_size (int): per-call override for the chunk size used to iterate
            over ``extrapolation_features``. Defaults to the module-level
            ``CHUNK_SIZE`` constant.

    Returns:
        torch.Tensor: propagated output, shape ``(m, d')``.

    Examples:
        >>> anchor_features = torch.randn(3000, 100)
        >>> anchor_output = torch.randn(3000, 20)
        >>> new_features = torch.randn(200, 100)
        >>> new_output = extrapolate_knn(anchor_features, anchor_output, new_features, "cosine", knn=3)
    """
    device = anchor_output.device if device is None else device
    anchor_output = anchor_output.to(device)

    effective_chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    n_chunks = ceildiv(extrapolation_features.shape[0], effective_chunk_size)
    V_list = []
    for _v in torch.chunk(extrapolation_features, n_chunks, dim=0):
        _v = _v.to(device)                                                                              # [_m x d]

        _A = affinity_from_features(
            features_A=anchor_features,
            features_B=_v,
            affinity_type=affinity_type,
            affinity_focal_gamma=affinity_focal_gamma,
        ).mT                                                                                            # [_m x n]
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

    return torch.cat(V_list, dim=0)


def extrapolate_knn_with_subsampling(
    full_features: torch.Tensor,                    # [n x d]
    full_output: torch.Tensor,                      # [n x d']
    extrapolation_features: torch.Tensor,           # [m x d]
    sample_config: SampleConfig,
    affinity_type: AffinityOptions,
    knn: Optional[int] = 10,
    affinity_focal_gamma: float = 1.0,
    device: Optional[str] = None,
    move_output_to_cpu: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:                                  # [m x d']
    """Subsample anchors from ``full_features`` then call :func:`extrapolate_knn`.

    This is equivalent to ``NCUT.transform(new_features)`` except that the sampling
    is redone here.

    Args:
        full_features (torch.Tensor): features from existing nodes, shape ``(n, d)``.
        full_output (torch.Tensor): output on existing nodes, shape ``(n, d')``.
        extrapolation_features (torch.Tensor): features for new nodes, shape ``(m, d)``.
        sample_config (SampleConfig): anchor sampling configuration; ``method`` is one of
            ``'full'``, ``'random'``, ``'fps'``, ``'fps_recursive'``.
        affinity_type (str): one of ``'cosine'`` or ``'rbf'``.
        knn (int): number of nearest anchors used for each new node.
        affinity_focal_gamma (float): affinity temperature.
        device (str): device to use; defaults to ``full_output.device``.
        move_output_to_cpu (bool): if ``True``, moves each chunk back to CPU.
        chunk_size (int): per-call override for the chunk size; forwarded to
            :func:`extrapolate_knn`. Defaults to the module-level ``CHUNK_SIZE``.

    Returns:
        torch.Tensor: propagated output, shape ``(m, d')``.

    Examples:
        >>> from nystrom_ncut import SampleConfig
        >>> old_features = torch.randn(3000, 100)
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> new_features = torch.randn(200, 100)
        >>> new_eigenvectors = extrapolate_knn_with_subsampling(
        ...     old_features, old_eigenvectors, new_features,
        ...     SampleConfig(method="fps", num_sample=512), "cosine", knn=3,
        ... )
    """
    device = full_output.device if device is None else device

    anchor_indices = subsample_features(
        features=full_features,
        distance_type=AFFINITY_TO_DISTANCE[affinity_type],
        config=sample_config,
    )

    anchor_output = full_output[anchor_indices].to(device)
    anchor_features = full_features[anchor_indices].to(device)
    extrapolation_features = extrapolation_features.to(device)

    return extrapolate_knn(
        anchor_features,
        anchor_output,
        extrapolation_features,
        affinity_type,
        knn=knn,
        affinity_focal_gamma=affinity_focal_gamma,
        device=device,
        move_output_to_cpu=move_output_to_cpu,
        chunk_size=chunk_size,
    )
