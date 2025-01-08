import numpy as np
import torch
import torch.nn.functional as Fn

from src.nystrom_ncut.ncut_pytorch import NCUT, axis_align
# from ncut_pytorch.src import rgb_from_umap_sphere
# from ncut_pytorch.src.new_ncut_pytorch import NewNCUT

# from ncut_pytorch.ncut_pytorch.backbone_text import load_text_model


if __name__ == "__main__":
    # torch.manual_seed(1212)
    # M = torch.randn((7, 3))
    # W = torch.nn.functional.cosine_similarity(M[:, None], M[None, :], dim=-1)
    # A = torch.exp(W - 1)
    # D_s2 = torch.sum(A, dim=-1, keepdim=True) ** -0.5
    # # print(A)
    # print(A * D_s2 * D_s2.mT)
    #
    # ncut = NCUT(num_eig=7, knn=1, eig_solver="svd")
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

    torch.set_printoptions(precision=8, sci_mode=False, linewidth=400)
    torch.set_default_dtype(torch.float64)
    # torch.manual_seed(1212)
    # np.random.seed(1212)

    M = torch.rand((200, 12))
    NC = NCUT(n_components=12, num_sample=80, sample_method="random", chunk_size=20)

    torch.manual_seed(1212)
    np.random.seed(1212)
    X, eigs = NC.fit_transform(M)
    print(eigs)
    raise Exception()

    normalized_M = Fn.normalize(M, p=2, dim=-1)
    A = torch.exp(-(1 - normalized_M @ normalized_M.mT))
    R = torch.diag(torch.sum(A, dim=-1) ** -0.5)
    L = R @ A @ R
    # print(L)
    print(X @ torch.diag(eigs) @ X.mT)
    print(L)
    print(torch.abs(X @ torch.diag(eigs) @ X.mT / L - 1))

    # torch.manual_seed(1212)
    # np.random.seed(1212)
    #
    # aX, R = axis_align(X)
    # print(aX[:3])
    # print(R)
    # print(R @ R.mT)
    raise Exception()




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

    torch.manual_seed(1212)
    np.random.seed(1212)
    X_2d, rgb = rgb_from_umap_sphere(X)
    # X_3d, rgb = rgb_from_cosine_tsne_3d(X)
    print(rgb)
