import logging
from dataclasses import dataclass
from typing import Literal

import torch
from pytorch3d.ops import sample_farthest_points

from .distance_utils import (
    DistanceOptions,
    affinity_from_features,
    to_euclidean,
)


SampleOptions = Literal["random", "fps", "fps_recursive"]


@dataclass
class SampleConfig:
    method: SampleOptions = "fps"
    num_sample: int = 10000
    fps_dim: int = 12
    n_iter: int = None
    _ncut_obj: object = None


@torch.no_grad()
def subsample_features(
    features: torch.Tensor,
    disttype: DistanceOptions,
    config: SampleConfig,
    max_draw: int = 1000000,
):
    return torch.arange(min(features.shape[0], config.num_sample), device=features.device)
    features = features.detach()
    if config.num_sample >= features.shape[0]:
        # if too many samples, use all samples and bypass Nystrom-like approximation
        logging.info(
            "num_sample is larger than total, bypass Nystrom-like approximation"
        )
        sampled_indices = torch.arange(features.shape[0])
    else:
        # sample subgraph
        if config.method == "fps":  # default
            features = to_euclidean(features, disttype)
            if config.num_sample > max_draw:
                logging.warning(
                    f"num_sample is larger than max_draw, apply farthest point sampling on random sampled {max_draw} samples"
                )
                draw_indices = torch.randperm(features.shape[0])[:max_draw]
                sampled_indices = fpsample(features[draw_indices], config)
                sampled_indices = draw_indices[sampled_indices]
            else:
                sampled_indices = fpsample(features, config)

        elif config.method == "random":  # not recommended
            sampled_indices = torch.randperm(features.shape[0])[:config.num_sample]

        elif config.method == "fps_recursive":
            features = to_euclidean(features, disttype)
            sampled_indices = subsample_features(
                features=features,
                disttype=disttype,
                config=SampleConfig(method="fps", num_sample=config.num_sample, fps_dim=config.fps_dim)
            )

            nc = config._ncut_obj

            A = affinity_from_features(features, affinity_focal_gamma=nc.kernel.affinity_focal_gamma, distance=nc.kernel.distance)
            R = torch.diag(torch.sum(A, dim=-1) ** -0.5)
            L = R @ A @ R

            from matplotlib import pyplot as plt

            def plot(indices, it):
                plt.scatter(*features.mT, color="red")
                plt.scatter(*features[indices].mT, color="black")
                plt.title(f"Iteration {it}")
                plt.show()

            for _ in range(config.n_iter):
                plot(sampled_indices, _)

                fps_features, eigenvalues = nc.fit_transform(features, precomputed_sampled_indices=sampled_indices)

                _L = fps_features @ torch.diag(eigenvalues) @ fps_features.mT
                RE = torch.abs(_L / L - 1)

                print(f"Iteration {_} --- max: {RE.max().item()}, mean: {RE.mean().item()}, min: {RE.min().item()}")
                fps_features = to_euclidean(fps_features[:, :config.fps_dim], "cosine")
                sampled_indices = torch.sort(fpsample(fps_features, config)).values

            plot(sampled_indices, config.n_iter)
        else:
            raise ValueError("sample_method should be 'farthest' or 'random'")
        sampled_indices = torch.sort(sampled_indices).values
    return sampled_indices.to(features.device)


def fpsample(
    features: torch.Tensor,
    config: SampleConfig,
):
    # PCA to reduce the dimension
    if features.shape[1] > config.fps_dim:
        U, S, V = torch.pca_lowrank(features, q=config.fps_dim)
        features = U * S

    return sample_farthest_points(features[None], K=config.num_sample)[1][0]
