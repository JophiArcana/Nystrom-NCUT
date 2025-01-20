import logging
from typing import Any, Callable, Dict, Literal

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.base import BaseEstimator

from .common import (
    ceildiv,
    lazy_normalize,
    quantile_min_max,
    quantile_normalize,
)
from .distance_utils import (
    DistanceOptions,
    to_euclidean,
    affinity_from_features,
)
from .sampling_utils import (
    SampleConfig,
    subsample_features,
)


def extrapolate_knn(
    anchor_features: torch.Tensor,          # [n x d]
    anchor_output: torch.Tensor,            # [n x d']
    extrapolation_features: torch.Tensor,   # [m x d]
    distance: DistanceOptions,
    knn: int = 10,                          # k
    affinity_focal_gamma: float = 1.0,
    chunk_size: int = 8192,
    device: str = None,
    move_output_to_cpu: bool = False,
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
        >>> new_eigenvectors = extrapolate_knn(old_features, old_eigenvectors, new_features, knn=3)
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

    extrapolation_output = torch.cat(V_list, dim=0)
    return extrapolation_output


# wrapper functions for adding new nodes to existing graph
def extrapolate_knn_with_subsampling(
    full_features: torch.Tensor,            # [n x d]
    full_output: torch.Tensor,              # [n x d']
    extrapolation_features: torch.Tensor,   # [m x d]
    sample_config: SampleConfig,
    distance: DistanceOptions,
    knn: int = 10,                          # k
    affinity_focal_gamma: float = 1.0,
    chunk_size: int = 8192,
    device: str = None,
    move_output_to_cpu: bool = False,
) -> torch.Tensor:                          # [m x d']
    """Propagate eigenvectors to new nodes using KNN. Note: this is equivalent to the class API `NCUT.tranform(new_features)`, expect for the sampling is re-done in this function.
    Args:
        full_output (torch.Tensor): eigenvectors from existing nodes, shape (num_sample, num_eig)
        full_features (torch.Tensor): features from existing nodes, shape (n_samples, n_features)
        extrapolation_features (torch.Tensor): features from new nodes, shape (n_new_samples, n_features)
        knn (int): number of KNN to propagate eigenvectors, default 3
        sample_config (str): sample method, 'farthest' (default) or 'random'
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
    anchor_indices = subsample_features(
        features=full_features,
        disttype=distance,
        config=sample_config,
    )

    anchor_output = full_output[anchor_indices].to(device)
    anchor_features = full_features[anchor_indices].to(device)
    extrapolation_features = extrapolation_features.to(device)

    # propagate eigenvectors from subgraph to new nodes
    extrapolation_output = extrapolate_knn(
        anchor_features,
        anchor_output,
        extrapolation_features,
        distance,
        knn=knn,
        affinity_focal_gamma=affinity_focal_gamma,
        chunk_size=chunk_size,
        device=device,
        move_output_to_cpu=move_output_to_cpu,
    )
    return extrapolation_output


def _rgb_with_dimensionality_reduction(
    features: torch.Tensor,
    num_sample: int,
    disttype: Literal["cosine", "euclidean"],
    rgb_func: Callable[[torch.Tensor, float], torch.Tensor],
    q: float,
    knn: int,
    reduction: Callable[..., BaseEstimator],
    reduction_dim: int,
    reduction_kwargs: Dict[str, Any],
    seed: int,
    device: str,
) -> torch.Tensor:

    if True:
        _subgraph_indices = subsample_features(
            features=features,
            disttype=disttype,
            config=SampleConfig(method="fps"),
        )
        features = extrapolate_knn(
            anchor_features=features[_subgraph_indices],
            anchor_output=features[_subgraph_indices],
            extrapolation_features=features,
            distance=disttype,
        )

    subgraph_indices = subsample_features(
        features=features,
        disttype=disttype,
        config=SampleConfig(method="fps", num_sample=num_sample),
    )

    _inp = features[subgraph_indices].numpy(force=True)
    _subgraph_embed = reduction(
        n_components=reduction_dim,
        metric=disttype,
        random_state=seed,
        **reduction_kwargs
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    rgb = rgb_func(extrapolate_knn(
        features[subgraph_indices],
        _subgraph_embed,
        features,
        disttype,
        knn=knn,
        device=device,
        move_output_to_cpu=True
    ), q)
    return rgb


def rgb_from_tsne_2d(
    features: torch.Tensor,
    num_sample: int = 1000,
    disttype: Literal["cosine", "euclidean"] = "cosine",
    perplexity: int = 150,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
) -> torch.Tensor:
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

    rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        disttype=disttype,
        rgb_func=rgb_from_2d_colormap,
        q=q,
        knn=knn,
        reduction=TSNE, reduction_dim=2, reduction_kwargs={
            "perplexity": perplexity,
        },
        seed=seed,
        device=device,
    )
    return rgb


def rgb_from_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    disttype: Literal["cosine", "euclidean"] = "cosine",
    perplexity: int = 150,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
) -> torch.Tensor:
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

    rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        disttype=disttype,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q,
        knn=knn,
        reduction=TSNE, reduction_dim=3, reduction_kwargs={
            "perplexity": perplexity,
        },
        seed=seed,
        device=device,
    )
    return rgb


def rgb_from_euclidean_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    disttype: Literal["cosine", "euclidean"] = "cosine",
    perplexity: int = 150,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None
) -> torch.Tensor:
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

    def rgb_func(X_3d: torch.Tensor, q: float) -> torch.Tensor:
        return rgb_from_3d_rgb_cube(to_euclidean(X_3d, disttype), q=q)

    rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        disttype="cosine",
        rgb_func=rgb_func,
        q=q,
        knn=knn,
        reduction=TSNE, reduction_dim=3, reduction_kwargs={
            "perplexity": perplexity,
        },
        seed=seed,
        device=device,
    )
    return rgb


def rgb_from_umap_2d(
    features: torch.Tensor,
    num_sample: int = 1000,
    disttype: Literal["cosine", "euclidean"] = "cosine",
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        disttype=disttype,
        rgb_func=rgb_from_2d_colormap,
        q=q,
        knn=knn,
        reduction=UMAP, reduction_dim=2, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
        seed=seed,
        device=device,
    )
    return rgb


def rgb_from_umap_sphere(
    features: torch.Tensor,
    num_sample: int = 1000,
    disttype: Literal["cosine", "euclidean"] = "cosine",
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
    device: str = None,
) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    def rgb_func(X: torch.Tensor, q: float) -> torch.Tensor:
        return rgb_from_3d_rgb_cube(torch.stack((
            torch.sin(X[:, 0]) * torch.cos(X[:, 1]),
            torch.sin(X[:, 0]) * torch.sin(X[:, 1]),
            torch.cos(X[:, 0]),
        ), dim=1), q=q)

    rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        disttype=disttype,
        rgb_func=rgb_func,
        q=q,
        knn=knn,
        reduction=UMAP, reduction_dim=2, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "output_metric": "haversine",
        },
        seed=seed,
        device=device,
    )
    return rgb


def rgb_from_umap_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    disttype: Literal["cosine", "euclidean"] = "cosine",
    n_neighbors: int = 150,
    min_dist: float = 0.1,
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

    rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        disttype=disttype,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q,
        knn=knn,
        reduction=UMAP, reduction_dim=3, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
        seed=seed,
        device=device,
    )
    return rgb


def flatten_sphere(X_3d: torch.Tensor) -> torch.Tensor:
    x = torch.atan2(X_3d[:, 0], X_3d[:, 1])
    y = -torch.acos(X_3d[:, 2])
    X_2d = torch.stack((x, y), dim=1)
    return X_2d


def rotate_rgb_cube(rgb: torch.Tensor, position: int = 1) -> torch.Tensor:
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


def rgb_from_3d_rgb_cube(X_3d: torch.Tensor, q: float = 0.95) -> torch.Tensor:
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


def rgb_from_3d_lab_cube(X_3d: torch.Tensor, q: float = 0.95, full_range: bool = True) -> torch.Tensor:
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

    rgb = torch.tensor(color.lab2rgb(lab.numpy(force=True)))
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


def rgb_from_2d_colormap(X_2d: torch.Tensor, q: float = 0.95):
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
    vmin, vmax = quantile_min_max(heatmap.flatten(), 0.01, 0.99)
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
    prompt_eigvec = Fn.normalize(prompt_eigvec, p=2, dim=-1)

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
