"""Anchor-point sampling for Nystrom-style approximation.

Provides farthest-point sampling (FPS) and uniform random sampling utilities
used to pick a subset of anchor points before fitting an
:class:`~nystrom_ncut.transformer.OnlineTransformerSubsampleFit` instance.
"""
from dataclasses import dataclass
from typing import Any, Optional

import torch
from pytorch3d.ops import sample_farthest_points

from .common import (
    default_device,
)
from .distance_utils import (
    DistanceOptions,
    to_euclidean,
)
from .types import SampleOptions


__all__ = [
    "SampleOptions",
    "SampleConfig",
    "subsample_features",
    "fpsample",
]


@dataclass
class SampleConfig:
    """Configuration for anchor sampling.

    Attributes:
        method: One of ``"full"``, ``"random"``, ``"fps"``, or ``"fps_recursive"``.
        num_sample: Maximum number of anchors. Clamped to ``features.shape[-2]``.
        fps_dim: Dimensionality of the low-rank PCA used for FPS.
        n_iter: Number of refinement iterations for ``"fps_recursive"``.
    """
    method: SampleOptions = "full"
    num_sample: int = 10000
    fps_dim: int = 12
    n_iter: Optional[int] = None


@torch.no_grad()
def subsample_features(
    features: torch.Tensor,
    distance_type: DistanceOptions,
    config: SampleConfig,
    recursive_obj: Optional[Any] = None,
) -> torch.Tensor:
    """Pick anchor indices from ``features``.

    Args:
        features (torch.Tensor): input features, shape ``(..., n, d)``.
        distance_type (str): ``'cosine'`` or ``'euclidean'``.
        config (SampleConfig): sampling configuration.
        recursive_obj: only required when ``config.method == 'fps_recursive'``; an
            :class:`OnlineTransformerSubsampleFit` instance whose
            ``fit_transform`` is called between rounds to refine the anchors.

    Returns:
        torch.Tensor: sampled indices, shape ``(..., num_sample)``.
    """
    features = features.detach()                                                                        # float: [... x n x d]
    with default_device(features.device):
        if config.method == "full" or config.num_sample >= features.shape[0]:
            sampled_indices = torch.arange(features.shape[-2]).expand(features.shape[:-1])              # int: [... x n]
        else:
            match config.method:
                case "fps":
                    sampled_indices = fpsample(to_euclidean(features, distance_type), config)

                case "random":
                    mask = torch.all(torch.isfinite(features), dim=-1)                                  # bool: [... x n]
                    weights = mask.to(torch.float) + torch.rand(mask.shape)                             # float: [... x n]
                    sampled_indices = torch.topk(weights, k=config.num_sample, dim=-1).indices          # int: [... x num_sample]

                case "fps_recursive":
                    if recursive_obj is None:
                        raise ValueError(
                            "'fps_recursive' requires a recursive_obj argument; "
                            "this is set automatically when called via OnlineTransformerSubsampleFit."
                        )
                    features = to_euclidean(features, distance_type)                                    # float: [... x n x d]
                    sampled_indices = subsample_features(
                        features=features,
                        distance_type=distance_type,
                        config=SampleConfig(method="fps", num_sample=config.num_sample, fps_dim=config.fps_dim),
                    )                                                                                   # int: [... x num_sample]
                    for _ in range(config.n_iter):
                        fps_features = recursive_obj.fit_transform(
                            features, precomputed_sampled_indices=sampled_indices,
                        )
                        fps_features = to_euclidean(fps_features[:, :config.fps_dim], "cosine")
                        sampled_indices = torch.sort(fpsample(fps_features, config), dim=-1).values

                case _:
                    raise ValueError(
                        f"sample_method should be one of 'full', 'random', 'fps', 'fps_recursive', got {config.method!r}"
                    )
            sampled_indices = torch.sort(sampled_indices, dim=-1).values
        return sampled_indices


def fpsample(
    features: torch.Tensor,
    config: SampleConfig,
) -> torch.Tensor:
    shape = features.shape[:-2]                                                         # ...
    features = features.view((-1, *features.shape[-2:]))                                # [(...) x n x d]
    bsz = features.shape[0]

    mask = torch.all(torch.isfinite(features), dim=-1)                                  # bool: [(...) x n]
    count = torch.sum(mask, dim=-1)                                                     # int: [(...)]
    order = torch.topk(mask.to(torch.int), k=torch.max(count).item(), dim=-1).indices   # int: [(...) x max_count]

    features = torch.nan_to_num(features[torch.arange(bsz)[:, None], order], nan=0.0)   # float: [(...) x max_count x d]
    if features.shape[-1] > config.fps_dim:
        U, S, V = torch.pca_lowrank(features, q=config.fps_dim)                         # float: [(...) x max_count x fps_dim]
        features = U * S[..., None, :]                                                  # float: [(...) x max_count x fps_dim]

    try:
        sample_indices = sample_farthest_points(
            features, lengths=count, K=config.num_sample
        )[1]                                                                            # int: [(...) x num_sample]
    except RuntimeError:
        original_device = features.device
        alternative_device = "cuda" if original_device == "cpu" else "cpu"
        sample_indices = sample_farthest_points(
            features.to(alternative_device), lengths=count.to(alternative_device), K=config.num_sample,
        )[1].to(original_device)                                                        # int: [(...) x num_sample]
    sample_indices = torch.gather(order, 1, sample_indices)                             # int: [(...) x num_sample]

    return sample_indices.view((*shape, *sample_indices.shape[-1:]))                    # int: [... x num_sample]


def __getattr__(name: str):
    """Backward-compat: ``OnlineTransformerSubsampleFit`` moved to ``nystrom_ncut.transformer``."""
    if name == "OnlineTransformerSubsampleFit":
        from .transformer import OnlineTransformerSubsampleFit
        return OnlineTransformerSubsampleFit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
