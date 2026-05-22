import numpy as np
import torch
import matplotlib.pyplot as plt

from src.nystrom_ncut import NystromNCut
from ncut_pytorch import NCUT, affinity_from_features, ncut, nystrom_ncut


def plot_eigvecs_dynamic(eigvec1, eigvec2, eigvec3, sample_index, x_2d, n_cols=6, tol=1e-3):
    n_eig = eigvec1.shape[1]

    n_rows = n_eig // n_cols * 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))    
    
    # vmaxvmin_fn = lambda x: max(abs(x.min()), abs(x.max()))
    
    def vmaxvmin_fn(x):
        quantile = torch.tensor([0.01, 0.99])
        quantiles = x.flatten().quantile(quantile)
        return max(abs(quantiles[0]), abs(quantiles[1]))
        
    # plot one row of eigvec1, one row of eigvec2, repeat
    for i in range(n_eig):
        ax = axs[i // n_cols * 3, i % n_cols]
        _v = vmaxvmin_fn(eigvec1[:, i])
        sc = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=eigvec1[:, i], cmap="bwr", vmin=-_v, vmax=_v)
        fig.colorbar(sc, ax=ax)
        ax.set_title(f"exact NCUT eig{i}")
            
        ax = axs[i // n_cols * 3 + 1, i % n_cols]
        _v = vmaxvmin_fn(eigvec2[:, i])
        sc = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=eigvec2[:, i], cmap="bwr", vmin=-_v, vmax=_v)
        fig.colorbar(sc, ax=ax)
        ax.set_title(f"KNN nystrom NCUT eig{i}")
        
        ax = axs[i // n_cols * 3 + 2, i % n_cols]
        _v = vmaxvmin_fn(eigvec3[:, i])
        sc = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=eigvec3[:, i], cmap="bwr", vmin=-_v, vmax=_v)
        fig.colorbar(sc, ax=ax)
        ax.set_title(f"original nystrom NCUT eig{i}")
        
    axs[0, 0].scatter(x_2d[sample_index, 0], x_2d[sample_index, 1], c="k", s=10)
    
    return fig


def non_approximation_ncut(features, num_eig, **config_kwargs):
    aff = affinity_from_features(features, **config_kwargs)
    eigvec, eigval = ncut(aff, num_eig)
    return eigvec, eigval


def original_nystrom_ncut(features, num_eig, precomputed_sampled_indices, distance="rbf"):
    eigvec, eigval = NystromNCut(n_components=num_eig, distance=distance).fit_transform(
        features, precomputed_sampled_indices=precomputed_sampled_indices)
    return eigvec, eigval


def knn_nystrom_ncut(features, num_eig, precomputed_sampled_indices, distance="rbf"):
    eigvec, eigval = NCUT(num_eig=num_eig, knn=10, distance=distance).fit_transform(
        features, precomputed_sampled_indices=precomputed_sampled_indices)
    return eigvec, eigval


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # seed = 1213
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    x_2d, y = make_blobs(n_samples=5000, centers=5, n_features=2, random_state=42)
    x_2d = torch.tensor(x_2d, dtype=torch.float32)
    # sort by y
    x_2d = x_2d[y.argsort()]
    # print(x_2d.shape)
    

    num_eig = 6

    # 1) balanced case
    # precomputed_sampled_indices = torch.arange(0, 5000, 250)
    
    # 2) imbalanced case
    precomputed_sampled_indices = torch.cat([torch.arange(0, 5000, 250), 
                                            torch.arange(0, 1000, 10)])
    precomputed_sampled_indices = torch.unique(precomputed_sampled_indices)

    distance = "rbf"
    eigvec1, eigval1 = non_approximation_ncut(x_2d, num_eig, distance=distance)

    eigvec2, eigval2 = knn_nystrom_ncut(x_2d, num_eig, precomputed_sampled_indices, distance=distance)

    eigvec3, eigval3 = original_nystrom_ncut(x_2d, num_eig, precomputed_sampled_indices, distance=distance)

    fig = plot_eigvecs_dynamic(eigvec1, eigvec2, eigvec3, precomputed_sampled_indices, x_2d, n_cols=6,)
    fig.suptitle("""Comparison of exact NCUT, KNN nystrom NCUT, and original nystrom NCUT.
                 black dots are precomputed sampled indices""")
    fig.tight_layout()
    plt.show()
