# nystrom-ncut

PyTorch implementations of **Normalized Cut** (spectral clustering) for very
large graphs, with two complementary approximations:

- **Nyström extension** — subsample anchor points, decompose the small
  anchor-anchor affinity block, and extrapolate to all remaining points
  (Fowlkes, Belongie, Chung, Malik 2004,
  [*Spectral Grouping Using the Nyström Method*](https://people.eecs.berkeley.edu/~malik/papers/FBCM-nystrom.pdf)).
- **Random Fourier features** — replace the kernel with an explicit
  finite-dimensional feature map so the affinity is never materialized
  (Rahimi & Recht 2007,
  [*Random Features for Large-Scale Kernel Machines*](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)).

Both are built on the original normalized-cut formulation
(Shi & Malik 2000, *Normalized Cuts and Image Segmentation*).

## Installation

```bash
pip install nystrom-ncut
```

Requires Python `>=3.10` and a working `torch` + `pytorch3d` installation.
For a CUDA build of PyTorch, install it explicitly before installing this
package, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install nystrom-ncut
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick start

### Nyström Normalized Cut

```python
import torch
from nystrom_ncut import NystromNCut, SampleConfig

# (B, H, W, C) features from some backbone
model_features = torch.rand(20, 64, 64, 768)
inp = model_features.reshape(-1, 768)

ncut = NystromNCut(
    n_components=100,
    affinity_type="rbf",
    sample_config=SampleConfig(method="fps", num_sample=10000),
)
eigvectors = ncut.fit_transform(inp)              # (20*64*64, 100)
eigvalues = ncut.eigenvalues_                     # (100,)
eigvectors = eigvectors.reshape(20, 64, 64, 100)
```

### Random-feature Kernel Normalized Cut

```python
import torch
from nystrom_ncut import KernelNCut

inp = torch.rand(20 * 64 * 64, 768)
kn = KernelNCut(
    n_components=100,
    kernel_dim=1024,
    affinity_type="rbf",
)
eigvectors = kn.fit_transform(inp)
```

### Distance Realization (MDS-style embedding)

```python
import torch
from nystrom_ncut import DistanceRealization

inp = torch.rand(10000, 768)
dr = DistanceRealization(n_components=64, distance_type="euclidean")
embedding = dr.fit_transform(inp)                 # (10000, 64)
```

### Discretizing eigenvectors into clusters

`AxisAlign` implements the Yu & Shi 2003 multiclass discretization on top of
the spectral embedding:

```python
from nystrom_ncut import AxisAlign

aligner = AxisAlign(sort_method="norm")
hard_labels = aligner.fit_transform(eigvectors, hard=True)  # (N,) int cluster ids
```

## Sampling

`SampleConfig` controls how anchor points are picked for the Nyström and
kernel methods. Available methods are:

- `"full"` — use every point (no subsampling).
- `"random"` — uniform random subset.
- `"fps"` — farthest-point sampling on a low-rank PCA of the features (default
  for accuracy).
- `"fps_recursive"` — iteratively refine FPS anchors using the current
  spectral embedding.

```python
from nystrom_ncut import SampleConfig

cfg = SampleConfig(method="fps", num_sample=10000, fps_dim=12)
```

## Repository layout

```
src/nystrom_ncut/
├── kernel/             # Random-feature NCut
├── nystrom/            # Nystrom NCut + distance realization
├── transformer/        # OnlineTransformerSubsampleFit, AxisAlign, mixins
├── common.py
├── distance_utils.py
├── extrapolation.py    # KNN extrapolation of anchors to new points
├── coloring.py         # RGB visualization helpers
└── sampling_utils.py
```

## Citation

If you find this useful, please cite the upstream NCUT writeup:

> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature
> Space. Huzheng Yang, James Gee, Jianbo Shi, 2024.

## License

MIT.
