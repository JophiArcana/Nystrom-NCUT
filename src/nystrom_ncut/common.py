from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as Fn


DistanceOptions = Literal["cosine", "euclidean", "rbf"]
SampleOptions = Literal["farthest", "random"]


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def lazy_normalize(x: torch.Tensor, n: int = 1000, **normalize_kwargs: Any) -> torch.Tensor:
    numel = np.prod(x.shape[:-1])
    n = min(n, numel)
    random_indices = torch.randperm(numel)[:n]
    _x = x.flatten(0, -2)[random_indices]
    if torch.allclose(torch.norm(_x, **normalize_kwargs), torch.ones(n, device=x.device)):
        return x
    else:
        return Fn.normalize(x, **normalize_kwargs)


def to_euclidean(x: torch.Tensor, disttype: DistanceOptions) -> torch.Tensor:
    if disttype == "cosine":
        return lazy_normalize(x, p=2, dim=-1)
    elif disttype == "rbf":
        return x
    else:
        raise ValueError(f"to_euclidean not implemented for disttype {disttype}.")


def quantile_min_max(x, q1=0.01, q2=0.99, n_sample=10000):
    if x.shape[0] > n_sample:
        np.random.seed(0)
        random_idx = np.random.choice(x.shape[0], n_sample, replace=False)
        vmin, vmax = x[random_idx].quantile(q1), x[random_idx].quantile(q2)
    else:
        vmin, vmax = x.quantile(q1), x.quantile(q2)
    return vmin, vmax


def quantile_normalize(x, q=0.95):
    """normalize each dimension of x to [0, 1], take 95-th percentage, this robust to outliers
        </br> 1. sort x
        </br> 2. take q-th quantile
        </br>     min_value -> (1-q)-th quantile
        </br>     max_value -> q-th quantile
        </br> 3. normalize
        </br> x = (x - min_value) / (max_value - min_value)

    Args:
        x (torch.Tensor): input tensor, shape (n_samples, n_features)
            normalize each feature to 0-1 range
        q (float): quantile, default 0.95

    Returns:
        torch.Tensor: quantile normalized tensor
    """
    # normalize x to 0-1 range, max value is q-th quantile
    # quantile makes the normalization robust to outliers
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    vmax, vmin = quantile_min_max(x, q, 1 - q)
    x = (x - vmin) / (vmax - vmin)
    x = x.clamp(0, 1)
    return x
