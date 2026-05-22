"""Subsample-and-fit pipeline shared by :class:`NystromNCut`, :class:`KernelNCut`,
and :class:`DistanceRealization`."""
import copy
from typing import Optional, Tuple

import torch

from .transformer_mixin import (
    OnlineTorchTransformerMixin,
    TorchTransformerMixin,
)
from ..sampling_utils import (
    SampleConfig,
    subsample_features,
)
from ..types import DistanceOptions


__all__ = ["OnlineTransformerSubsampleFit"]


class OnlineTransformerSubsampleFit(TorchTransformerMixin, OnlineTorchTransformerMixin):
    """Wrap an online transformer with anchor subsampling.

    Picks anchor indices via :func:`subsample_features`, fits ``base_transformer``
    on the anchors, calls ``update`` on the remaining points (in chunks via
    :mod:`global_settings`'s ``CHUNK_SIZE``), and scatters the results back into
    original index order.
    """

    def __init__(
        self,
        base_transformer: OnlineTorchTransformerMixin,
        distance_type: DistanceOptions,
        sample_config: SampleConfig,
    ):
        OnlineTorchTransformerMixin.__init__(self)
        self.base_transformer: OnlineTorchTransformerMixin = base_transformer
        self.distance_type: DistanceOptions = distance_type
        # Deepcopy to avoid mutating a user-supplied (and often shared) ``SampleConfig``.
        self.sample_config: SampleConfig = copy.deepcopy(sample_config)
        self.anchor_indices: Optional[torch.Tensor] = None

    def _fit_helper(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        _n = features.shape[-2]
        local_config = copy.copy(self.sample_config)
        local_config.num_sample = min(local_config.num_sample, _n)

        if precomputed_sampled_indices is not None:
            self.anchor_indices = precomputed_sampled_indices
        else:
            # Recursive FPS needs a side-effect-free clone of ``self`` to call
            # ``fit_transform`` on between rounds.
            recursive_obj = copy.deepcopy(self) if local_config.method == "fps_recursive" else None
            self.anchor_indices = subsample_features(
                features=features,
                distance_type=self.distance_type,
                config=local_config,
                recursive_obj=recursive_obj,
            )
        sampled_features = torch.gather(
            features, -2,
            self.anchor_indices[..., None].expand([-1] * self.anchor_indices.ndim + [features.shape[-1]]),
        )
        self.base_transformer.fit(sampled_features)

        _n_not_sampled = _n - self.anchor_indices.shape[-1]
        if _n_not_sampled > 0:
            unsampled_mask = torch.full(features.shape[:-1], True, device=features.device).scatter_(-1, self.anchor_indices, False)
            unsampled_indices = torch.where(unsampled_mask)[-1].view((*features.shape[:-2], -1))
            unsampled_features = torch.gather(
                features, -2,
                unsampled_indices[..., None].expand([-1] * unsampled_indices.ndim + [features.shape[-1]]),
            )
            V_unsampled = self.base_transformer.update(unsampled_features)
        else:
            unsampled_indices = V_unsampled = None
        return unsampled_indices, V_unsampled

    def fit(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: Optional[torch.Tensor] = None,
    ) -> "OnlineTransformerSubsampleFit":
        """Fit on the anchor subset.

        Args:
            features (torch.Tensor): input features, shape ``(..., n_samples, n_features)``.
            precomputed_sampled_indices (torch.Tensor): if provided, overrides the
                sample method with these explicit anchor indices.

        Returns:
            self
        """
        self._fit_helper(features, precomputed_sampled_indices)
        return self

    def fit_transform(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fit and return the spectral embedding.

        Args:
            features (torch.Tensor): input features, shape ``(..., n_samples, n_features)``.
            precomputed_sampled_indices (torch.Tensor): if provided, overrides the
                sample method with these explicit anchor indices.

        Returns:
            torch.Tensor: eigenvectors, shape ``(..., n_samples, n_components)``. The
            descending-sorted eigenvalues are available via :attr:`eigenvalues_`.
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
