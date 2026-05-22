import time

import torch

from src.nystrom_ncut import KernelNCut, NystromNCut, affinity_from_features, SampleConfig, AxisAlign


if __name__ == "__main__":
    torch.manual_seed(1212)
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(linewidth=400, sci_mode=False, precision=6)



    n, d = 12000, 1024
    n_components = 100
    shape = ()
    X = torch.randn((*shape, n, d))
    # X = torch.where((torch.rand((*shape, n, 1)) < 0.8).expand((*shape, n, d)), X, torch.nan)

    num_sample = 10000
    nnc = NystromNCut(
        n_components=n_components,
        affinity_type="rbf",
        adaptive_scaling=False,
        sample_config=SampleConfig(method="fps", num_sample=num_sample, fps_dim=3),
        # sample_config=SampleConfig(method="random", num_sample=num_sample),
        # sample_config=SampleConfig(method="fps_recursive", num_sample=num_sample, n_iter=10),
        eig_solver="svd_lowrank",
    )
    knc = KernelNCut(
        n_components=n_components,
        kernel_dim=1000,
        affinity_type="rbf",
        sample_config=SampleConfig(method="fps", num_sample=num_sample, fps_dim=3),
    )

    # precomputed_sampled_indices = torch.arange(num_sample).expand((*shape, num_sample))

    n_trials = 10

    start_t = time.perf_counter()
    for _ in range(n_trials):
        print(_)
        Vn = nnc.fit_transform(X)
    end_t = time.perf_counter()
    print(f"NystromNCut: {(end_t - start_t) / n_trials}s")

    start_t = time.perf_counter()
    for _ in range(n_trials):
        print(_)
        Vk = knc.fit_transform(X)
    end_t = time.perf_counter()
    print(f"KernelNCut: {(end_t - start_t) / n_trials}s")


    # print(V)

    # V_ = torch.stack([
    #     nc.fit_transform(X[idx], precomputed_sampled_indices=precomputed_sampled_indices[idx])
    #     for idx in range(shape[0])
    # ], dim=0)
    #
    # aa = AxisAlign(sort_method="count")
    # print(aa.fit_transform(V))





    # X = torch.randn((10, 5))
    # X[1] = torch.nan
    # Y = X[~torch.any(torch.isnan(X), dim=1)]
    #
    #
    # Z = torch.randn((7, 5))
    # Z[torch.randn((len(Z),)) < 0] = torch.nan
    #
    # torch.manual_seed(2002)
    # print(nc.fit_transform(torch.cat((X, Z), dim=0)))   # , precomputed_sampled_indices=torch.arange(len(X))))
    # torch.manual_seed(2002)
    # print(nc.fit_transform(torch.cat((Y, Z), dim=0)))   # , precomputed_sampled_indices=torch.arange(len(Y))))

