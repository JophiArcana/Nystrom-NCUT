import copy
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import torch
from pytorch3d.ops import sample_farthest_points

from .common import (
    default_device,
)
from .distance_utils import (
    DistanceOptions,
    to_euclidean,
)
from .transformer import (
    TorchTransformerMixin,
    OnlineTorchTransformerMixin,
)


SampleOptions = Literal["full", "random", "fps", "fps_recursive"]


@dataclass
class SampleConfig:
    method: SampleOptions = "full"
    num_sample: int = 10000
    fps_dim: int = 12
    n_iter: int = None
    _recursive_obj: TorchTransformerMixin = None


@torch.no_grad()
def subsample_features(
    features: torch.Tensor,
    distance_type: DistanceOptions,
    config: SampleConfig,
):
    features = features.detach()                                                                        # float: [... x n x d]
    with default_device(features.device):
        if config.method == "full" or config.num_sample >= features.shape[0]:
            sampled_indices = torch.arange(features.shape[-2]).expand(features.shape[:-1])              # int: [... x n]
        else:
            # sample
            match config.method:
                case "fps":  # default
                    sampled_indices = fpsample(to_euclidean(features, distance_type), config)

                case "random":  # not recommended
                    mask = torch.all(torch.isfinite(features), dim=-1)                                  # bool: [... x n]
                    weights = mask.to(torch.float) + torch.rand(mask.shape)                             # float: [... x n]
                    sampled_indices = torch.topk(weights, k=config.num_sample, dim=-1).indices          # int: [... x num_sample]

                case "fps_recursive":
                    features = to_euclidean(features, distance_type)                                    # float: [... x n x d]
                    sampled_indices = subsample_features(
                        features=features,
                        distance_type=distance_type,
                        config=SampleConfig(method="fps", num_sample=config.num_sample, fps_dim=config.fps_dim)
                    )                                                                                   # int: [... x num_sample]
                    nc = config._recursive_obj
                    for _ in range(config.n_iter):
                        fps_features, eigenvalues = nc.fit_transform(features, precomputed_sampled_indices=sampled_indices)

                        fps_features = to_euclidean(fps_features[:, :config.fps_dim], "cosine")
                        sampled_indices = torch.sort(fpsample(fps_features, config), dim=-1).values

                case _:
                    raise ValueError("sample_method should be 'farthest' or 'random'")
            sampled_indices = torch.sort(sampled_indices, dim=-1).values
        return sampled_indices


def fpsample(
    features: torch.Tensor,
    config: SampleConfig,
):
    shape = features.shape[:-2]                                                         # ...
    features = features.view((-1, *features.shape[-2:]))                                # [(...) x n x d]
    bsz = features.shape[0]

    mask = torch.all(torch.isfinite(features), dim=-1)                                  # bool: [(...) x n]
    count = torch.sum(mask, dim=-1)                                                     # int: [(...)]
    order = torch.topk(mask.to(torch.int), k=torch.max(count).item(), dim=-1).indices   # int: [(...) x max_count]

    features = torch.nan_to_num(features[torch.arange(bsz)[:, None], order], nan=0.0)   # float: [(...) x max_count x d]
    if features.shape[-1] > config.fps_dim:
        U, S, V = torch.pca_lowrank(features, q=config.fps_dim)                         # float: [(...) x max_count x fps_dim], [(...) x fps_dim], [(...) x fps_dim x fps_dim]
        features = U * S[..., None, :]                                                  # float: [(...) x max_count x fps_dim]

    try:
        sample_indices = sample_farthest_points(
            features, lengths=count, K=config.num_sample
        )[1]                                                                            # int: [(...) x num_sample]
    except RuntimeError:
        original_device = features.device
        alternative_device = "cuda" if original_device == "cpu" else "cpu"
        sample_indices = sample_farthest_points(
            features.to(alternative_device), lengths=count.to(alternative_device), K=config.num_sample
        )[1].to(original_device)                                                        # int: [(...) x num_sample]
    sample_indices = torch.gather(order, 1, sample_indices)                             # int: [(...) x num_sample]

    return sample_indices.view((*shape, *sample_indices.shape[-1:]))                    # int: [... x num_sample]


class OnlineTransformerSubsampleFit(TorchTransformerMixin, OnlineTorchTransformerMixin):
    def __init__(
        self,
        base_transformer: OnlineTorchTransformerMixin,
        distance_type: DistanceOptions,
        sample_config: SampleConfig,
    ):
        OnlineTorchTransformerMixin.__init__(self)
        self.base_transformer: OnlineTorchTransformerMixin = base_transformer
        self.distance_type: DistanceOptions = distance_type
        self.sample_config: SampleConfig = sample_config
        self.sample_config._recursive_obj = copy.deepcopy(self)
        self.anchor_indices: torch.Tensor = None

    def _fit_helper(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _n = features.shape[-2]
        self.sample_config.num_sample = min(self.sample_config.num_sample, _n)

        if precomputed_sampled_indices is not None:
            self.anchor_indices = precomputed_sampled_indices
        else:
            self.anchor_indices = subsample_features(
                features=features,
                distance_type=self.distance_type,
                config=self.sample_config,
            )
        sampled_features = torch.gather(features, -2, self.anchor_indices[..., None].expand([-1] * self.anchor_indices.ndim + [features.shape[-1]]))
        self.base_transformer.fit(sampled_features)

        _n_not_sampled = _n - self.anchor_indices.shape[-1]
        if _n_not_sampled > 0:
            unsampled_mask = torch.full(features.shape[:-1], True, device=features.device).scatter_(-1, self.anchor_indices, False)
            unsampled_indices = torch.where(unsampled_mask)[-1].view((*features.shape[:-2], -1))
            unsampled_features = torch.gather(features, -2, unsampled_indices[..., None].expand([-1] * unsampled_indices.ndim + [features.shape[-1]]))
            V_unsampled = self.base_transformer.update(unsampled_features)
        else:
            unsampled_indices = V_unsampled = None
        return unsampled_indices, V_unsampled

    def fit(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor = None,
    ) -> "OnlineTransformerSubsampleFit":
        """Fit Nystrom Normalized Cut on the input features.
        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None
        Returns:
            (NCut): self
        """
        self._fit_helper(features, precomputed_sampled_indices)
        return self

    def fit_transform(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None

        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
        unsampled_indices, V_unsampled = self._fit_helper(features, precomputed_sampled_indices)
        V_sampled = self.base_transformer.transform()

        if unsampled_indices is not None:
            V = torch.zeros((*features.shape[:-1], V_sampled.shape[-1]), device=features.device)
            for (indices, _V) in [(self.anchor_indices, V_sampled), (unsampled_indices, V_unsampled)]:
                V.scatter_(-2, indices[..., None].expand([-1] * indices.ndim + [V_sampled.shape[-1]]), _V)
        else:
            V = V_sampled
        return V

    def update(self, features: torch.Tensor) -> torch.Tensor:
        return self.base_transformer.update(features)

    def transform(self, features: torch.Tensor = None, **transform_kwargs) -> torch.Tensor:
        return self.base_transformer.transform(features)

    @property
    def eigenvalues_(self) -> torch.Tensor:
        return getattr(self.base_transformer, "eigenvalues_", None)
