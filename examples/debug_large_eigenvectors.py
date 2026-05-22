import numpy as np
import torch
from matplotlib import pyplot as plt

from src.nystrom_ncut import NystromNCut, affinity_from_features, SampleConfig
from src.nystrom_ncut.nystrom.nystrom_utils import solve_eig

# from ncut_pytorch.src import rgb_from_umap_sphere
# from ncut_pytorch.src.new_ncut_pytorch import NewNCUT

# from ncut_pytorch.ncut_pytorch.backbone_text import load_text_model


def truncate_inv(A, n_inv=-1):
    if n_inv > 0:
        ev, ei, _ = torch.svd_lowrank(A, n_inv)
        c = ei.sum() / torch.trace(A)
        # ei *= c
        print("eig_inv, c", c)
    else:
        ei, ev = torch.linalg.eigh(A)
        print("eig_inv, ei", ei.min(), ei.max())
    A_inv = ev @ torch.diag(1 / (ei.abs() + 1e-8)) @ ev.T
    return A_inv


if __name__ == "__main__":
    # from src.nystrom_ncut.visualize_utils import rgb_from_3d_lab_cube
    # M = torch.randn((100, 3))
    # print(rgb_from_3d_lab_cube(M))
    # raise Exception()

    # torch.manual_seed(1212)
    # M = torch.randn((7, 3))
    # W = torch.nn.functional.cosine_similarity(M[:, None], M[None, :], dim=-1)
    # A = torch.exp(W - 1)
    # D_s2 = torch.sum(A, dim=-1, keepdim=True) ** -0.5
    # # print(A)
    # print(A * D_s2 * D_s2.mT)
    #
    # ncut = NCut(n_components=7, eig_solver="svd")
    # V, L = ncut.fit_transform(M)
    # print(V @ torch.diag(L) @ V.mT)
    # raise Exception()

    # print(load_text_model("meta-llama/Meta-Llama-3.1-8B").cuda())
    # print(AutoModelForCausalLM.from_pretrained(
    #     "meta-llama/Meta-Llama-3.1-8B",
    #     token="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",
    # ))
    # # print(transformers.pipeline(
    # #     "text-generation",
    # #     model="meta-llama/Meta-Llama-3.1-8B",
    # #     model_kwargs={"torch_dtype": torch.bfloat16},
    # #     token="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",
    # #     device="cpu",
    # # ))
    # raise Exception(

    torch.set_printoptions(precision=6, sci_mode=False, linewidth=400)
    torch.set_default_dtype(torch.float64)
    # torch.manual_seed(1212)
    # np.random.seed(1212)

    n, d = 100, 12
    num_sample = 70

    X = torch.randn((n, d))

    distance = "rbf"

    A = affinity_from_features(X, distance=distance)
    R = torch.diag(torch.sum(A, dim=-1) ** -0.5)
    W = R @ A @ R

    # C = L[num_sample:, num_sample:]
    #
    # _A = L[:num_sample, :num_sample]
    # _B = L[:num_sample, num_sample:]
    # extrapolated_C = _B.mT @ torch.inverse(_A) @ _B
    #
    # RE = torch.abs(extrapolated_C / C - 1)
    # print(torch.max(RE).item(), torch.mean(RE).item(), torch.min(RE).item())

    n_components = num_sample
    eig_solver = "svd"

    def rel_error(V, L):
        _W = V @ torch.diag(L) @ V.mT
        return torch.abs(_W / W - 1)

    def print_re(re):
        print(f"max: {re.max().item()}, mean: {re.mean().item()}, min: {re.min().item()}")

    nc = NystromNCut(
        n_components=n_components,
        distance=distance,
        adaptive_scaling=False,
        # sample_config=SampleConfig(method="random", num_sample=num_sample),
        sample_config=SampleConfig(method="fps_recursive", num_sample=num_sample, n_iter=10),
        eig_solver=eig_solver,
    )
    V = nc.fit_transform(X, precomputed_sampled_indices=torch.arange(num_sample))
    L = nc.eigenvalues_
    print(V.mT @ V)
    raise Exception()
    # print(L)
    # re = rel_error(V, L)
    # plt.imshow(re)
    # plt.show()

    _A = affinity_from_features(X[:num_sample], X[:num_sample], distance=distance)
    _B = affinity_from_features(X[:num_sample], X[num_sample:], distance=distance)
    # if False:
    #     _C = _B.mT @ torch.inverse(_A) @ _B
    #     _adaptive_scale = torch.diag(_C) ** -0.5
    #     _B = _B * _adaptive_scale
    _M = torch.cat((torch.cat((_A, _B), dim=1), torch.cat((_B.mT, _B.mT @ torch.inverse(_A) @ _B), dim=1)), dim=0)
    _R = torch.diag(torch.sum(_M, dim=-1) ** -0.5)
    _W = _R @ _M @ _R
    _V, _L = solve_eig(_W, n_components, eig_solver)
    # print(_L)

    # _V = torch.nn.functional.normalize(_V, p=2, dim=-1)
    print(_V.mT @ _V)

    plt.scatter(torch.arange(n), torch.linalg.norm(_V, dim=-1))
    plt.show()

    # plt.hist(max_rel, bins=30)
    # plt.show()

    # plt.imshow(re0)
    # plt.colorbar()
    # plt.show()
    #
    # plt.scatter(torch.arange(n), torch.linalg.norm(X0, dim=-1))
    # plt.show()
    raise Exception()


    #
    # # plt.scatter(torch.arange(n), torch.linalg.norm(X0, dim=-1))
    # # plt.show()
    # # raise Exception()
    #
    # def align_to(X, eigs):
    #     sign = torch.sign(torch.sum(X0 * X, dim=0))
    #     return X * sign, eigs
    #
    # Xs = []
    # n_trials = 20
    # sum_X, sum_eigs = 0.0, 0.0
    # for _ in range(n_trials):
    #     nc = NCUT(n_components=n_components, num_sample=num_sample, distance=distance, eig_solver=eig_solver)
    #     X, eigs = align_to(*nc.fit_transform(M))
    #     Xs.append(X)
    #
    #     re = rel_error(X, eigs)
    #     print(f"max: {re.max().item()}, mean: {re.mean().item()}, min: {re.min().item()}")
    #
    #     # print(X[:3, :10])
    #     # print(eigs[:10])
    #
    #     sum_X = sum_X + X
    #     sum_eigs = sum_eigs + eigs
    #
    # # print(torch.diag(Xs[0].mT @ Xs[1]))
    # # raise Exception()
    #
    # print("=" * 120)
    # mean_X, mean_eigs = sum_X / n_trials, sum_eigs / n_trials
    # mean_re = rel_error(mean_X, mean_eigs)
    # print(f"max: {mean_re.max().item()}, mean: {mean_re.mean().item()}, min: {mean_re.min().item()}")
    #
    # raise Exception()



    ncs = [
        NCUT(n_components=n_components, num_sample=n, distance=distance, eig_solver=eig_solver),
        NCUT(n_components=n_components, num_sample=num_sample, distance=distance, eig_solver=eig_solver),
        # OldNCUT(num_eig=n_components, num_sample=num_sample, knn=10, distance=distance, eig_solver=eig_solver, make_orthogonal=True),
    ]

    for NC in ncs:
        torch.manual_seed(1212)
        np.random.seed(1212)
        V, L = NC.fit_transform(X)

        RE = rel_error(V, L)
        print(f"max: {RE.max().item()}, mean: {RE.mean().item()}, min: {RE.min().item()}")

    # torch.manual_seed(1212)
    # np.random.seed(1212)
    #
    # aX, R = axis_align(X)
    # print(aX[:3])
    # print(R)
    # print(R @ R.mT)




    # import time
    # n_trials = 10
    #
    # with torch.no_grad():
    #     start_t = time.perf_counter()
    #     for _ in range(n_trials):
    #         X, eigs = NC.fit_transform(M)
    #     end_t = time.perf_counter()
    #     print(X.min().item(), X.max().item(), eigs)
    #     print(f"{1e3 * (end_t - start_t) / n_trials}ms")
    #
    #     start_t = time.perf_counter()
    #     for _ in range(n_trials):
    #         nX, neigs = nNC.fit_transform(M)
    #     end_t = time.perf_counter()
    #     print(nX.min().item(), nX.max().item(), neigs)
    #     print(f"{1e3 * (end_t - start_t) / n_trials}ms")
    # raise Exception()

    # assert torch.all(torch.isclose(X, torch.Tensor([
    #     [0.320216, 0.144101, -0.110744, -0.560543, -0.007982],
    #     [0.297634, 0.662867, 0.146107, 0.277893, 0.553959],
    #     [0.324994, -0.057295, 0.052916, 0.391666, -0.460911],
    #     [0.301703, -0.460709, 0.528563, 0.222525, 0.325546],
    #     [0.316614, 0.043475, -0.526899, 0.100665, -0.030259],
    #     [0.325425, -0.127884, 0.294540, -0.012173, -0.303528],
    #     [0.318136, -0.288952, -0.065148, -0.470192, 0.244805],
    #     [0.309522, -0.352693, -0.473237, 0.234057, 0.276185],
    #     [0.320464, 0.229301, 0.281134, -0.308938, -0.169746],
    #     [0.326147, 0.213536, -0.112246, 0.155114, -0.341439]
    # ]), atol=1e-6)), "Failed assertion"

    # torch.manual_seed(1212)
    # np.random.seed(1212)
    # X_2d, rgb = rgb_from_umap_sphere(X)
    # # X_3d, rgb = rgb_from_cosine_tsne_3d(X)
    # print(rgb)
