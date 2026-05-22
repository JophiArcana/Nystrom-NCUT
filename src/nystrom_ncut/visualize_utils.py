"""Deprecated alias module.

Use :mod:`nystrom_ncut.extrapolation` (for ``extrapolate_knn`` and
``extrapolate_knn_with_subsampling``) and :mod:`nystrom_ncut.coloring` (for the
RGB visualization helpers and ``get_mask``) directly. This module re-exports
their public names for backward compatibility and will be removed in a future
release.
"""
from .coloring import (
    convert_to_lab_color,
    flatten_sphere,
    get_mask,
    rgb_from_2d_colormap,
    rgb_from_3d_lab_cube,
    rgb_from_3d_rgb_cube,
    rgb_from_euclidean_tsne_3d,
    rgb_from_tsne_2d,
    rgb_from_tsne_3d,
    rgb_from_umap_2d,
    rgb_from_umap_3d,
    rgb_from_umap_sphere,
    rotate_rgb_cube,
)
from .extrapolation import (
    extrapolate_knn,
    extrapolate_knn_with_subsampling,
)


__all__ = [
    "convert_to_lab_color",
    "extrapolate_knn",
    "extrapolate_knn_with_subsampling",
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
