import torch

from src.nystrom_ncut.transformer import AxisAlign


if __name__ == "__main__":
    torch.set_printoptions(linewidth=400, sci_mode=False)
    torch.set_default_dtype(torch.float64)

    M = torch.randn((7, 5))
    U, S, VT = torch.linalg.svd(M)
    X = U.repeat((3, 1))

    X = torch.randn((100, 7))

    ax = AxisAlign(sort_method="marginal_norm")
    clusters = ax.fit_transform(X, hard=True)

    print(clusters)
    print(torch.bincount(clusters))

