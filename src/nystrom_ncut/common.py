from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fn


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def lazy_normalize(x: torch.Tensor, n: int = 1000, **normalize_kwargs: Any) -> torch.Tensor:
    """L2-normalize ``x`` unless it already appears to be normalized.

    Inspects up to ``n`` rows (deterministically, evenly spaced) to decide
    whether ``x`` needs renormalization. Avoiding the renormalization is purely
    a runtime optimization; the result is equivalent to
    ``torch.nn.functional.normalize(x, **normalize_kwargs)``.
    """
    flat = x.view((-1, x.shape[-1]))
    numel = flat.shape[0]
    if numel == 0:
        return x
    n = min(n, numel)
    step = max(1, numel // n)
    _x = flat[::step][:n]
    if torch.allclose(torch.norm(_x, **normalize_kwargs), torch.ones(_x.shape[0], device=x.device)):
        return x
    return Fn.normalize(x, **normalize_kwargs)


def quantile_min_max(x: torch.Tensor, q1: float, q2: float, n_sample: int = 10000):
    x = x.flatten()
    if len(x) > n_sample:
        np.random.seed(0)
        random_idx = np.random.choice(len(x), n_sample, replace=False)
        vmin, vmax = x[random_idx].quantile(q1), x[random_idx].quantile(q2)
    else:
        vmin, vmax = x.quantile(q1), x.quantile(q2)
    return vmin.item(), vmax.item()


def quantile_normalize(x: torch.Tensor, q: float = 0.95):
    """Normalize each dimension of ``x`` to ``[0, 1]`` using a quantile range.

    Robust to outliers:

    1. Sort ``x``.
    2. Take the ``q``-th quantile and the ``(1 - q)``-th quantile.
    3. Scale: ``x = clamp((x - vmin) / (vmax - vmin), 0, 1)``.

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


class default_device:
    def __init__(self, device: torch.device):
        self._device = device

    def __enter__(self):
        self._original_device = torch.get_default_device()
        torch.set_default_device(self._device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_default_device(self._original_device)


def profile(name: str, t: torch.Tensor) -> None:
    print(f"{name} --- nan: {t.isnan().any()}, inf: {t.isinf().any()}, max: {t.abs().max()}, min: {t.abs().min()}")
