import logging
from typing import Any, Callable, Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator

from .common import (
    lazy_normalize,
    quantile_min_max,
    quantile_normalize,
)
from .nystrom import (
    DistanceRealization,
)
from .propagation_utils import (
    run_subgraph_sampling,
    extrapolate_knn,
)


def _identity(X: torch.Tensor) -> torch.Tensor:
    return X


def _rgb_with_dimensionality_reduction(
    features: torch.Tensor,
    num_sample: int,
    metric: Literal["cosine", "euclidean"],
    rgb_func: Callable[[torch.Tensor, float], torch.Tensor],
    q: float,
    knn: int,
    reduction: Callable[..., BaseEstimator],
    reduction_dim: int,
    reduction_kwargs: Dict[str, Any],
    transform_func: Callable[[torch.Tensor], torch.Tensor],
    seed: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if True:
        _subgraph_indices = run_subgraph_sampling(
            features,
            num_sample=10000,
            sample_method="farthest",
        )
        features = extrapolate_knn(
            features[_subgraph_indices],
            features[_subgraph_indices],
            features,
            distance="cosine",
        )

    subgraph_indices = run_subgraph_sampling(
        features,
        num_sample,
        sample_method="farthest",
    )

    _inp = features[subgraph_indices].numpy(force=True)
    _subgraph_embed = reduction(
        n_components=reduction_dim,
        metric=metric,
        random_state=seed,
        **reduction_kwargs
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    X_nd = transform_func(extrapolate_knn(
        features[subgraph_indices],
        _subgraph_embed,
        features,
        knn=knn,
        distance=metric,
        device=device,
        move_output_to_cpu=True
    ))
    rgb = rgb_func(X_nd, q)
    return X_nd, rgb


def rgb_from_tsne_2d(
    features: torch.Tensor,
    num_sample: int = 1000,
    perplexity: int = 150,
    metric: Literal["cosine", "euclidean"] = "cosine",
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "sklearn import failed, please install `pip install scikit-learn`"
        )
    num_sample = min(num_sample, features.shape[0])
    if perplexity > num_sample // 2:
        logging.warning(
            f"perplexity is larger than num_sample, set perplexity to {num_sample // 2}"
        )
        perplexity = num_sample // 2

    x2d, rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_2d_colormap,
        q=q,
        knn=knn,
        reduction=TSNE, reduction_dim=2, reduction_kwargs={
            "perplexity": perplexity,
        }, transform_func=_identity,
        seed=seed,
        device=device,
    )
    return x2d, rgb


def rgb_from_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    perplexity: int = 150,
    metric: Literal["cosine", "euclidean"] = "cosine",
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
):
    """
    Returns:
        (torch.Tensor): Embedding in 3D, shape (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "sklearn import failed, please install `pip install scikit-learn`"
        )
    num_sample = min(num_sample, features.shape[0])
    if perplexity > num_sample // 2:
        logging.warning(
            f"perplexity is larger than num_sample, set perplexity to {num_sample // 2}"
        )
        perplexity = num_sample // 2

    x3d, rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q,
        knn=knn,
        reduction=TSNE, reduction_dim=3, reduction_kwargs={
            "perplexity": perplexity,
        }, transform_func=_identity,
        seed=seed,
        device=device,
    )
    return x3d, rgb


def rgb_from_cosine_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    perplexity: int = 150,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None
):
    """
    Returns:
        (torch.Tensor): Embedding in 3D, shape (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "sklearn import failed, please install `pip install scikit-learn`"
        )
    num_sample = min(num_sample, features.shape[0])
    if perplexity > num_sample // 2:
        logging.warning(
            f"perplexity is larger than num_sample, set perplexity to {num_sample // 2}"
        )
        perplexity = num_sample // 2

    def cosine_to_rbf(X: torch.Tensor) -> torch.Tensor:
        dr = DistanceRealization(n_components=3, num_sample=20000, distance="cosine", eig_solver="svd_lowrank")
        return dr.fit_transform(X)

    def rgb_from_cosine(X_3d: torch.Tensor, q: float) -> torch.Tensor:
        return rgb_from_3d_rgb_cube(cosine_to_rbf(X_3d), q=q)

    x3d, rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric="cosine",
        rgb_func=rgb_from_cosine,
        q=q,
        knn=knn,
        reduction=TSNE, reduction_dim=3, reduction_kwargs={
            "perplexity": perplexity,
        }, transform_func=_identity,
        seed=seed,
        device=device,
    )
    return x3d, rgb


def rgb_from_umap_2d(
    features: torch.Tensor,
    num_sample: int = 1000,
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    x2d, rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_2d_colormap,
        q=q,
        knn=knn,
        reduction=UMAP, reduction_dim=2, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        }, transform_func=_identity,
        seed=seed,
        device=device,
    )
    return x2d, rgb


def rgb_from_umap_sphere(
    features: torch.Tensor,
    num_sample: int = 1000,
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    def transform_func(X: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            torch.sin(X[:, 0]) * torch.cos(X[:, 1]),
            torch.sin(X[:, 0]) * torch.sin(X[:, 1]),
            torch.cos(X[:, 0]),
        ), dim=1)

    x3d, rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q,
        knn=knn,
        reduction=UMAP, reduction_dim=2, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "output_metric": "haversine",
        }, transform_func=transform_func,
        seed=seed,
        device=device,
    )
    return x3d, rgb


def rgb_from_umap_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    x3d, rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q,
        knn=knn,
        reduction=UMAP, reduction_dim=3, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        }, transform_func=_identity,
        seed=seed,
        device=device,
    )
    return x3d, rgb


def flatten_sphere(X_3d):
    x = np.arctan2(X_3d[:, 0], X_3d[:, 1])
    y = -np.arccos(X_3d[:, 2])
    X_2d = np.stack([x, y], axis=1)
    return X_2d


def rotate_rgb_cube(rgb, position=1):
    """rotate RGB cube to different position

    Args:
        rgb (torch.Tensor): RGB color space [0, 1], shape (*, 3)
        position (int): position to rotate, 0, 1, 2, 3, 4, 5, 6

    Returns:
        torch.Tensor: RGB color space, shape (n_samples, 3)
    """
    assert position in range(0, 7), "position should be 0, 1, 2, 3, 4, 5, 6"
    rotation_matrix = torch.tensor((
        (0., 1., 0.),
        (0., 0., 1.),
        (1., 0., 0.),
    ))
    n_mul = position % 3
    rotation_matrix = torch.matrix_power(rotation_matrix, n_mul)
    rgb = rgb @ rotation_matrix
    if position > 3:
        rgb = 1 - rgb
    return rgb


def rgb_from_3d_rgb_cube(X_3d, q=0.95):
    """convert 3D t-SNE to RGB color space
    Args:
        X_3d (torch.Tensor): 3D t-SNE embedding, shape (n_samples, 3)
        q (float): quantile, default 0.95

    Returns:
        torch.Tensor: RGB color space, shape (n_samples, 3)
    """
    assert X_3d.shape[1] == 3, "input should be (n_samples, 3)"
    assert len(X_3d.shape) == 2, "input should be (n_samples, 3)"
    rgb = torch.stack([
        quantile_normalize(x, q=q)
        for x in torch.unbind(X_3d, dim=1)
    ], dim=-1)
    return rgb


def convert_to_lab_color(rgb, full_range=True):
    from skimage import color
    import copy

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
    lab_rgb = color.lab2rgb(_rgb)
    return lab_rgb


def rgb_from_2d_colormap(X_2d, q=0.95):
    xy = X_2d.clone()
    for i in range(2):
        xy[:, i] = quantile_normalize(xy[:, i], q=q)

    try:
        from pycolormap_2d import (
            ColorMap2DBremm,
            ColorMap2DZiegler,
            ColorMap2DCubeDiagonal,
            ColorMap2DSchumann,
        )
    except ImportError:
        raise ImportError(
            "pycolormap_2d import failed, please install `pip install pycolormap-2d`"
        )

    cmap = ColorMap2DCubeDiagonal()
    xy = xy.cpu().numpy()
    len_x, len_y = cmap._cmap_data.shape[:2]
    x = (xy[:, 0] * (len_x - 1)).astype(int)
    y = (xy[:, 1] * (len_y - 1)).astype(int)
    rgb = cmap._cmap_data[x, y]
    rgb = torch.tensor(rgb, dtype=torch.float32) / 255
    return rgb


# application: get segmentation mask fron a reference eigenvector (point prompt)
def _transform_heatmap(heatmap, gamma=1.0):
    """Transform the heatmap using gamma, normalize and min-max normalization.

    Args:
        heatmap (torch.Tensor): distance heatmap, shape (B, H, W)
        gamma (float, optional): scaling factor, higher means smaller mask. Defaults to 1.0.

    Returns:
        torch.Tensor: transformed heatmap, shape (B, H, W)
    """
    # normalize the heatmap
    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
    heatmap = torch.exp(heatmap)
    # transform the heatmap using gamma
    # large gamma means more focus on the high values, hence smaller mask
    heatmap = 1 / heatmap ** gamma
    # min-max normalization [0, 1]
    vmin, vmax = quantile_min_max(heatmap.flatten())
    heatmap = (heatmap - vmin) / (vmax - vmin)
    return heatmap


def _clean_mask(mask, min_area=500):
    """clean the binary mask by removing small connected components.

    Args:
    - mask: A numpy image of a binary mask with 255 for the object and 0 for the background.
    - min_area: Minimum area for a connected component to be considered valid (default 500).

    Returns:
    - bounding_boxes: List of bounding boxes for valid objects (x, y, width, height).
    - cleaned_pil_mask: A Pillow image of the cleaned mask, with small components removed.
    """

    import cv2
    # Find connected components in the cleaned mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Initialize an empty mask to store the final cleaned mask
    final_cleaned_mask = np.zeros_like(mask)

    # Collect bounding boxes for components that are larger than the threshold and update the cleaned mask
    bounding_boxes = []
    for i in range(1, num_labels):  # Skip label 0 (background)
        x, y, w, h, area = stats[i]
        if area >= min_area:
            # Add the bounding box of the valid component
            bounding_boxes.append((x, y, w, h))
            # Keep the valid components in the final cleaned mask
            final_cleaned_mask[labels == i] = 255

    return final_cleaned_mask, bounding_boxes


def get_mask(
    all_eigvecs: torch.Tensor, prompt_eigvec: torch.Tensor,
    threshold: float = 0.5, gamma: float = 1.0,
    denoise: bool = True, denoise_area_th: int = 3):
    """Segmentation mask from one prompt eigenvector (at a clicked latent pixel).
        </br> The mask is computed by measuring the cosine similarity between the clicked eigenvector and all the eigenvectors in the latent space.
        </br> 1. Compute the cosine similarity between the clicked eigenvector and all the eigenvectors in the latent space.
        </br> 2. Transform the heatmap, normalize and apply scaling (gamma).
        </br> 3. Threshold the heatmap to get the mask.
        </br> 4. Optionally denoise the mask by removing small connected components

    Args:
        all_eigvecs (torch.Tensor): (B, H, W, num_eig)
        prompt_eigvec (torch.Tensor): (num_eig,)
        threshold (float, optional): mask threshold, higher means smaller mask. Defaults to 0.5.
        gamma (float, optional): mask scaling factor, higher means smaller mask. Defaults to 1.0.
        denoise (bool, optional): mask denoising flag. Defaults to True.
        denoise_area_th (int, optional): mask denoising area threshold. higher means more aggressive denoising. Defaults to 3.

    Returns:
        np.ndarray: masks (B, H, W), 1 for object, 0 for background

    Examples:
        >>> all_eigvecs = torch.randn(10, 64, 64, 20)
        >>> prompt_eigvec = all_eigvecs[0, 32, 32]  # center pixel
        >>> masks = get_mask(all_eigvecs, prompt_eigvec, threshold=0.5, gamma=1.0, denoise=True, denoise_area_th=3)
        >>> # masks.shape = (10, 64, 64)
    """

    # normalize the eigenvectors to unit norm, to compute cosine similarity
    all_eigvecs = lazy_normalize(all_eigvecs, p=2, dim=-1)
    prompt_eigvec = F.normalize(prompt_eigvec, p=2, dim=-1)

    # compute the cosine similarity
    cos_sim = all_eigvecs @ prompt_eigvec.unsqueeze(-1)  # (B, H, W, 1)
    cos_sim = cos_sim.squeeze(-1)  # (B, H, W)

    heatmap = 1 - cos_sim

    # transform the heatmap, normalize and apply scaling (gamma)
    heatmap = _transform_heatmap(heatmap, gamma=gamma)

    masks = heatmap > threshold
    masks = masks.numpy(force=True).astype(np.uint8)

    if denoise:
        cleaned_masks = []
        for mask in masks:
            cleaned_mask, _ = _clean_mask(mask, min_area=denoise_area_th)
            cleaned_masks.append(cleaned_mask)
        cleaned_masks = np.stack(cleaned_masks)
        return cleaned_masks

    return masks
