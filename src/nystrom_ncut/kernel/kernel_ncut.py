import torch

from ..distance_utils import (
    AffinityOptions,
    AFFINITY_TO_DISTANCE,
    get_normalization_factor,
)
from ..sampling_utils import (
    SampleConfig,
    OnlineTransformerSubsampleFit,
)
from ..transformer import (
    OnlineTorchTransformerMixin,
)


class KernelNCutBaseTransformer(OnlineTorchTransformerMixin):
    def __init__(
        self,
        n_components: int,
        kernel_dim: int,
        affinity_type: AffinityOptions,
        affinity_focal_gamma: float,
    ):
        self.n_components: int = n_components
        self.kernel_dim: int = kernel_dim
        self.affinity_type: AffinityOptions = affinity_type
        self.affinity_focal_gamma = affinity_focal_gamma

        # Anchor matrices
        self.anchor_count: int = None                   # n
        self.W: torch.Tensor = None                     # [... x d x kernel_dim]
        self.kernelized_anchor: torch.Tensor = None     # [... x n x (2 * kernel_dim)]

        # Updated matrices
        self.total_count: int = None                    # m
        self.r: torch.Tensor = None                     # [... x (2 * kernel_dim)]
        self.transform_matrix: torch.Tensor = None      # [... x (2 * kernel_dim) x n_components]
        self.eigenvalues_: torch.Tensor = None          # [... x n_components]

    def _update(self) -> None:
        row_sum = self.kernelized_anchor @ self.r[..., None]                        # [... x n x 1]
        normalized_kernelized_anchor = self.kernelized_anchor / (row_sum ** 0.5)    # [... x n x (2 * kernel_dim)]
        _, S, V = torch.svd_lowrank(torch.nan_to_num(
            normalized_kernelized_anchor, nan=0.0,
        ), q=self.n_components)                                                     # [... x n_components], [... x (2 * kernel_dim) x n_components]
        S = S * (self.total_count / self.anchor_count) ** 0.5
        self.transform_matrix = V * torch.nan_to_num(1 / S, posinf=0.0, neginf=0.0)[..., None, :]   # [... x (2 * kernel_dim) x n_components]
        self.eigenvalues_ = S ** 2

    def fit(self, features: torch.Tensor) -> "KernelNCutBaseTransformer":
        self.anchor_count = self.total_count = features.shape[-2]
        d = features.shape[-1]
        scale = get_normalization_factor(features) * (self.affinity_focal_gamma ** 0.5)             # [...]
        self.W = torch.randn((*features.shape[:-2], d, self.kernel_dim)) / scale[..., None, None]   # [... x d x kernel_dim]

        W_anchor = features @ self.W                                                                # [... x n x kernel_dim]
        self.kernelized_anchor = torch.cat((
            torch.cos(W_anchor),
            torch.sin(W_anchor),
        ), dim=-1) / (self.kernel_dim ** 0.5)                                                       # [... x n * (2 * kernel_dim)]
        self.r = torch.sum(torch.nan_to_num(self.kernelized_anchor, nan=0.0), dim=-2)               # [... x (2 * kernel_dim)]
        self._update()
        return self

    def update(self, features: torch.Tensor) -> torch.Tensor:
        self.total_count += features.shape[-2]
        W_features = features @ self.W                                              # [... x m x kernel_dim]
        kernelized_features = torch.cat((
            torch.cos(W_features),
            torch.sin(W_features),
        ), dim=-1) / (self.kernel_dim ** 0.5)                                       # [... x m x (2 * kernel_dim)]
        b_r = torch.sum(torch.nan_to_num(kernelized_features, nan=0.0), dim=-2)     # [... x (2 * kernel_dim)]
        self.r = self.r + b_r
        self._update()

        row_sum = kernelized_features @ self.r[..., None]                           # [... x m x 1]
        normalized_kernelized_features = kernelized_features / (row_sum ** 0.5)     # [... x m x (2 * kernel_dim)]
        return normalized_kernelized_features @ self.transform_matrix               # [... x m x n_components]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        if features is None:
            kernelized_features = self.kernelized_anchor                            # [... x n x (2 * kernel_dim)]
        else:
            W_features = features @ self.W
            kernelized_features = torch.cat((
                torch.cos(W_features),
                torch.sin(W_features),
            ), dim=-1) / (self.kernel_dim ** 0.5)                                   # [... x m x (2 * kernel_dim)]
        row_sum = kernelized_features @ self.r[..., None]                           # [... x m x 1]
        normalized_kernelized_features = kernelized_features / (row_sum ** 0.5)     # [... x m x (2 * kernel_dim)]
        return normalized_kernelized_features @ self.transform_matrix               # [... x m x n_components]


class KernelNCut(OnlineTransformerSubsampleFit):
    """Kernelized Normalized Cut for large scale graph."""

    def __init__(
        self,
        n_components: int,
        kernel_dim: int = 1024,
        affinity_type: AffinityOptions = "cosine",
        affinity_focal_gamma: float = 1.0,
        sample_config: SampleConfig = SampleConfig(),
    ):
        OnlineTransformerSubsampleFit.__init__(
            self,
            base_transformer=KernelNCutBaseTransformer(
                n_components=n_components,
                kernel_dim=kernel_dim,
                affinity_type=affinity_type,
                affinity_focal_gamma=affinity_focal_gamma,
            ),
            distance_type=AFFINITY_TO_DISTANCE[affinity_type],
            sample_config=sample_config,
        )


