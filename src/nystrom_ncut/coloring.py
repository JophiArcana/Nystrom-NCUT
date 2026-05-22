"""RGB visualization helpers for spectral embeddings.

Maps high-dimensional embeddings to 2D or 3D via t-SNE / UMAP and then to RGB.
Also exposes :func:`get_mask`, which derives a segmentation mask from a
prompt eigenvector.
"""
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.base import BaseEstimator, TransformerMixin

from .common import (
    lazy_normalize,
    quantile_min_max,
    quantile_normalize,
)
from .distance_utils import (
    AFFINITY_TO_DISTANCE,
    to_euclidean,
)
from .extrapolation import (
    extrapolate_knn,
)
from .sampling_utils import (
    SampleConfig,
    subsample_features,
)
from .types import AffinityOptions


__all__ = [
    "convert_to_lab_color",
    "flatten_sphere",
    "get_mask",
    "rgb_from_2d_colormap",
    "rgb_from_3d_lab_cube",
    "rgb_from_3d_rgb_cube",
    "rgb_from_euclidean_tsne_3d",
    "rgb_from_tsne_2d",
    "rgb_from_tsne_3d",
    "rgb_from_umap_2d",
    "rgb_from_umap_3d",
    "rgb_from_umap_sphere",
    "rotate_rgb_cube",
]


def _rgb_with_dimensionality_reduction(
    features: torch.Tensor,
    num_sample: int,
    affinity_type: AffinityOptions,
    rgb_func: Callable[[torch.Tensor, float], torch.Tensor],
    q: float,
    knn: int,
    reduction: Callable[..., Union[TransformerMixin, BaseEstimator]],
    reduction_dim: int,
    reduction_kwargs: Dict[str, Any],
    seed: int,
    device: Optional[str],
) -> torch.Tensor:
    _subgraph_indices = subsample_features(
        features=features,
        distance_type=AFFINITY_TO_DISTANCE[affinity_type],
        config=SampleConfig(method="fps"),
    )
    features = extrapolate_knn(
        anchor_features=features[_subgraph_indices],
        anchor_output=features[_subgraph_indices],
        extrapolation_features=features,
        affinity_type=affinity_type,
    )

    subgraph_indices = subsample_features(
        features=features,
        distance_type=AFFINITY_TO_DISTANCE[affinity_type],
        config=SampleConfig(method="fps", num_sample=num_sample),
    )

    _inp = features[subgraph_indices].numpy(force=True)
    _subgraph_embed = torch.tensor(reduction(
        n_components=reduction_dim,
        metric=AFFINITY_TO_DISTANCE[affinity_type],
        random_state=seed,
        **reduction_kwargs,
    ).fit_transform(_inp), device=features.device, dtype=features.dtype)

    rgb = rgb_func(extrapolate_knn(
        features[subgraph_indices],
        _subgraph_embed,
        features,
        affinity_type,
        knn=knn,
        device=device,
        move_output_to_cpu=True,
    ), q)
    return rgb


def rgb_from_tsne_2d(
    features: torch.Tensor,
    num_sample: int = 1000,
    affinity_type: AffinityOptions = "cosine",
    perplexity: int = 150,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Embed ``features`` to 2D via t-SNE then map to RGB.

    Returns:
        torch.Tensor: RGB color for each data sample, shape ``(n_samples, 3)``.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise ImportError("install scikit-learn for t-SNE support") from e
    num_sample = min(num_sample, features.shape[0])
    perplexity = min(perplexity, num_sample // 2)

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        affinity_type=affinity_type,
        rgb_func=rgb_from_2d_colormap,
        q=q,
        knn=knn,
        reduction=TSNE,
        reduction_dim=2,
        reduction_kwargs={"perplexity": perplexity},
        seed=seed,
        device=device,
    )


def rgb_from_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    affinity_type: AffinityOptions = "cosine",
    perplexity: int = 150,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Embed ``features`` to 3D via t-SNE then map to RGB.

    Returns:
        torch.Tensor: RGB color for each data sample, shape ``(n_samples, 3)``.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise ImportError("install scikit-learn for t-SNE support") from e
    num_sample = min(num_sample, features.shape[0])
    perplexity = min(perplexity, num_sample // 2)

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        affinity_type=affinity_type,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q,
        knn=knn,
        reduction=TSNE,
        reduction_dim=3,
        reduction_kwargs={"perplexity": perplexity},
        seed=seed,
        device=device,
    )


def rgb_from_euclidean_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    affinity_type: AffinityOptions = "cosine",
    perplexity: int = 150,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Embed ``features`` to 3D via t-SNE in Euclidean space then map to RGB."""
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise ImportError("install scikit-learn for t-SNE support") from e
    num_sample = min(num_sample, features.shape[0])
    perplexity = min(perplexity, num_sample // 2)

    def rgb_func(X_3d: torch.Tensor, q: float) -> torch.Tensor:
        return rgb_from_3d_rgb_cube(to_euclidean(X_3d, AFFINITY_TO_DISTANCE[affinity_type]), q=q)

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        affinity_type=affinity_type,
        rgb_func=rgb_func,
        q=q,
        knn=knn,
        reduction=TSNE,
        reduction_dim=3,
        reduction_kwargs={"perplexity": perplexity},
        seed=seed,
        device=device,
    )


def rgb_from_umap_2d(
    features: torch.Tensor,
    num_sample: int = 1000,
    affinity_type: AffinityOptions = "cosine",
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Embed ``features`` to 2D via UMAP then map to RGB."""
    try:
        from umap import UMAP
    except ImportError as e:
        raise ImportError("install umap-learn for UMAP support") from e

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        affinity_type=affinity_type,
        rgb_func=rgb_from_2d_colormap,
        q=q,
        knn=knn,
        reduction=UMAP,
        reduction_dim=2,
        reduction_kwargs={"n_neighbors": n_neighbors, "min_dist": min_dist},
        seed=seed,
        device=device,
    )


def rgb_from_umap_sphere(
    features: torch.Tensor,
    num_sample: int = 1000,
    affinity_type: AffinityOptions = "cosine",
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Embed ``features`` onto the unit sphere via spherical UMAP then map to RGB."""
    try:
        from umap import UMAP
    except ImportError as e:
        raise ImportError("install umap-learn for UMAP support") from e

    def rgb_func(X: torch.Tensor, q: float) -> torch.Tensor:
        return rgb_from_3d_rgb_cube(torch.stack((
            torch.sin(X[:, 0]) * torch.cos(X[:, 1]),
            torch.sin(X[:, 0]) * torch.sin(X[:, 1]),
            torch.cos(X[:, 0]),
        ), dim=1), q=q)

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        affinity_type=affinity_type,
        rgb_func=rgb_func,
        q=q,
        knn=knn,
        reduction=UMAP,
        reduction_dim=2,
        reduction_kwargs={"n_neighbors": n_neighbors, "min_dist": min_dist, "output_metric": "haversine"},
        seed=seed,
        device=device,
    )


def rgb_from_umap_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    affinity_type: AffinityOptions = "cosine",
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Embed ``features`` to 3D via UMAP then map to RGB."""
    try:
        from umap import UMAP
    except ImportError as e:
        raise ImportError("install umap-learn for UMAP support") from e

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        affinity_type=affinity_type,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q,
        knn=knn,
        reduction=UMAP,
        reduction_dim=3,
        reduction_kwargs={"n_neighbors": n_neighbors, "min_dist": min_dist},
        seed=seed,
        device=device,
    )


def flatten_sphere(X_3d: torch.Tensor) -> torch.Tensor:
    """Project unit-sphere points onto a 2D equirectangular grid."""
    x = torch.atan2(X_3d[:, 0], X_3d[:, 1])
    y = -torch.acos(X_3d[:, 2])
    return torch.stack((x, y), dim=1)


def rotate_rgb_cube(rgb: torch.Tensor, position: int = 1) -> torch.Tensor:
    """Rotate the RGB cube into one of 6 axis-aligned orientations."""
    assert position in range(0, 7), "position should be 0, 1, 2, 3, 4, 5, 6"
    rotation_matrix = torch.tensor((
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ))
    rotation_matrix = torch.matrix_power(rotation_matrix, position % 3)
    rgb = rgb @ rotation_matrix
    if position > 3:
        rgb = 1 - rgb
    return rgb


def rgb_from_3d_rgb_cube(X_3d: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """Quantile-normalize a 3D embedding into ``[0, 1]^3`` (RGB cube)."""
    assert X_3d.shape[1] == 3, "input should be (n_samples, 3)"
    assert len(X_3d.shape) == 2, "input should be (n_samples, 3)"
    return torch.stack([
        quantile_normalize(x, q=q)
        for x in torch.unbind(X_3d, dim=1)
    ], dim=-1)


def rgb_from_3d_lab_cube(X_3d: torch.Tensor, q: float = 0.95, full_range: bool = True) -> torch.Tensor:
    """Map a 3D embedding into the CIELAB cube and convert to RGB."""
    from skimage import color
    X_3d = X_3d - torch.mean(X_3d, dim=0)
    U, S, VT = torch.linalg.svd(X_3d)
    X_3d = torch.flip(U[:, :3] * S, dims=(1,))

    AB_scale = 128.0 / torch.quantile(torch.linalg.norm(X_3d[:, 1:], dim=1), q=q, dim=0)
    L_min, L_max = torch.quantile(X_3d[:, 0], q=torch.tensor(((1 - q) / 2, (1 + q) / 2)), dim=0)
    L_scale = 100.0 / (L_max - L_min)

    X_3d[:, 0] = X_3d[:, 0] - L_min
    if full_range:
        lab = X_3d * torch.tensor((L_scale, AB_scale, AB_scale))
    else:
        lab = X_3d * L_scale

    return torch.tensor(color.lab2rgb(lab.numpy(force=True)))


def convert_to_lab_color(rgb, full_range: bool = True):
    """Convert an RGB array to CIELAB-based RGB."""
    import copy

    from skimage import color

    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    _rgb = copy.deepcopy(rgb)
    _rgb[..., 0] = _rgb[..., 0] * 100
    if full_range:
        _rgb[..., 1] = _rgb[..., 1] * 255 - 128
        _rgb[..., 2] = _rgb[..., 2] * 255 - 128
    else:
        _rgb[..., 1] = _rgb[..., 1] * 100 - 50
        _rgb[..., 2] = _rgb[..., 2] * 100 - 50
    return color.lab2rgb(_rgb)


def rgb_from_2d_colormap(X_2d: torch.Tensor, q: float = 0.95):
    """Map a 2D embedding to RGB via a 2D colormap."""
    xy = X_2d.clone()
    for i in range(2):
        xy[:, i] = quantile_normalize(xy[:, i], q=q)

    try:
        from pycolormap_2d import ColorMap2DCubeDiagonal
    except ImportError as e:
        raise ImportError("install pycolormap-2d for 2D colormap support") from e

    cmap = ColorMap2DCubeDiagonal()
    xy = xy.cpu().numpy()
    len_x, len_y = cmap._cmap_data.shape[:2]
    x = (xy[:, 0] * (len_x - 1)).astype(int)
    y = (xy[:, 1] * (len_y - 1)).astype(int)
    rgb = cmap._cmap_data[x, y]
    return torch.tensor(rgb, dtype=torch.float32) / 255


def _transform_heatmap(heatmap: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Standardize, gamma-scale, and quantile-normalize a heatmap into ``[0, 1]``."""
    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
    heatmap = torch.exp(heatmap)
    heatmap = 1 / heatmap ** gamma
    vmin, vmax = quantile_min_max(heatmap.flatten(), 0.01, 0.99)
    return (heatmap - vmin) / (vmax - vmin)


def _clean_mask(mask: np.ndarray, min_area: int = 500):
    """Remove small connected components from a binary mask."""
    import cv2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_cleaned_mask = np.zeros_like(mask)

    bounding_boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            bounding_boxes.append((x, y, w, h))
            final_cleaned_mask[labels == i] = 255

    return final_cleaned_mask, bounding_boxes


def get_mask(
    all_eigvecs: torch.Tensor,
    prompt_eigvec: torch.Tensor,
    threshold: float = 0.5,
    gamma: float = 1.0,
    denoise: bool = True,
    denoise_area_th: int = 3,
):
    """Segmentation mask from one prompt eigenvector (at a clicked latent pixel).

    Steps:

    1. Compute cosine similarity between the prompt eigenvector and all eigenvectors.
    2. Convert similarity to a heatmap, normalize and apply ``gamma`` scaling.
    3. Threshold to a binary mask.
    4. Optionally remove small connected components.

    Args:
        all_eigvecs (torch.Tensor): ``(B, H, W, num_eig)``.
        prompt_eigvec (torch.Tensor): ``(num_eig,)``.
        threshold (float): mask threshold; higher means smaller mask.
        gamma (float): mask scaling factor; higher means smaller mask.
        denoise (bool): mask denoising flag.
        denoise_area_th (int): minimum component area when denoising.

    Returns:
        np.ndarray: masks ``(B, H, W)``, 1 for object, 0 for background.

    Examples:
        >>> all_eigvecs = torch.randn(10, 64, 64, 20)
        >>> prompt_eigvec = all_eigvecs[0, 32, 32]
        >>> masks = get_mask(all_eigvecs, prompt_eigvec)
    """
    all_eigvecs = lazy_normalize(all_eigvecs, p=2, dim=-1)
    prompt_eigvec = Fn.normalize(prompt_eigvec, p=2, dim=-1)

    cos_sim = (all_eigvecs @ prompt_eigvec.unsqueeze(-1)).squeeze(-1)        # (B, H, W)
    heatmap = 1 - cos_sim
    heatmap = _transform_heatmap(heatmap, gamma=gamma)

    masks = (heatmap > threshold).numpy(force=True).astype(np.uint8)

    if denoise:
        cleaned_masks = [_clean_mask(m, min_area=denoise_area_th)[0] for m in masks]
        return np.stack(cleaned_masks)

    return masks
