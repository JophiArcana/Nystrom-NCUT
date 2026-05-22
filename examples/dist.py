import numpy as np
import torch
from scipy.stats import kendalltau

from src.nystrom_ncut import DistanceRealization, distance_from_features


if __name__ == "__main__":
    torch.set_printoptions(precision=8, sci_mode=False, linewidth=400)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2002)
    np.random.seed(1212)

    n, d = 5, 2
    distance_type = "rbf"
    X = torch.randn((n, d))
    dr = DistanceRealization(n_components=3, num_sample=3, distance=distance_type, eig_solver="svd")

    X_ = dr.fit_transform(X)

    print(X_ @ X_.mT)

    D = distance_from_features(X, X, distance_type)
    D_ = torch.cdist(X_, X_) ** 2

    # print(D)
    # print(D_)
    # print(torch.cdist(X, X)[:5, :5])
    # print(torch.cdist(X_, X_)[:5, :5])

    # indices = torch.randperm(n ** 2)[:10000]
    # print(kendalltau(D.flatten()[indices], D_.flatten()[indices]))



