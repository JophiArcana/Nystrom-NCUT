from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fn


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
