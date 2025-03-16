import copy
import logging
from abc import abstractmethod
from typing import Literal, Tuple

import torch

from ..common import (
    ceildiv,
)
from ..distance_utils import (
    DistanceOptions,
)
from ..sampling_utils import (
    SampleConfig,
    subsample_features,
)
from ..transformer import (
    TorchTransformerMixin,
)


EigSolverOptions = Literal["svd_lowrank", "lobpcg", "svd", "eigh"]


class OnlineKernel:
    @abstractmethod
    def fit(self, features: torch.Tensor) -> "OnlineKernel":                # [... x n x d]
        """"""

    @abstractmethod
    def update(self, features: torch.Tensor) -> torch.Tensor:               # [... x m x d] -> [... x m x n]
        """"""

    @abstractmethod
    def transform(self, features: torch.Tensor = None) -> torch.Tensor:     # [... x m x d] -> [... x m x n]
        """"""


class OnlineNystrom(TorchTransformerMixin):
    def __init__(
        self,
        n_components: int,
        kernel: OnlineKernel,
        eig_solver: EigSolverOptions,
        chunk_size: int = 8192,
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return
            kernel (OnlineKernel): Online kernel that computes pairwise matrix entries from input features and allows updates
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
        """
        self.n_components: int = n_components
        self.kernel: OnlineKernel = kernel
        self.eig_solver: EigSolverOptions = eig_solver
        self.shape: torch.Size = None               # ...

        self.chunk_size = chunk_size

        # Anchor matrices
        self.anchor_features: torch.Tensor = None   # [... x n x d]
        self.A: torch.Tensor = None                 # [... x n x n]
        self.Ahinv: torch.Tensor = None             # [... x n x n]
        self.Ahinv_UL: torch.Tensor = None          # [... x n x indirect_pca_dim]
        self.Ahinv_VT: torch.Tensor = None          # [... x indirect_pca_dim x n]

        # Updated matrices
        self.S: torch.Tensor = None                 # [... x n x n]
        self.transform_matrix: torch.Tensor = None  # [... x n x n_components]
        self.eigenvalues_: torch.Tensor = None      # [... x n]

    def _update_to_kernel(self, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.A = self.S = self.kernel.transform()
        U, L = solve_eig(
            self.A,
            num_eig=d + 1,  # d * (d + 3) // 2 + 1,
            eig_solver=self.eig_solver,
        )                                                                                           # [... x n x (? + 1)], [... x (? + 1)]
        self.Ahinv_UL = U * (L[..., None, :] ** -0.5)                                               # [... x n x (? + 1)]
        self.Ahinv_VT = U.mT                                                                        # [... x (? + 1) x n]
        self.Ahinv = self.Ahinv_UL @ self.Ahinv_VT                                                  # [... x n x n]
        return U, L

    def fit(self, features: torch.Tensor) -> "OnlineNystrom":
        OnlineNystrom.fit_transform(self, features)
        return self

    def fit_transform(self, features: torch.Tensor) -> torch.Tensor:
        self.anchor_features = features

        self.kernel.fit(self.anchor_features)
        U, L = self._update_to_kernel(features.shape[-1])                                           # [... x n x (d + 1)], [... x (d + 1)]

        self.transform_matrix = (U / L[..., None, :])[..., :, :self.n_components]                   # [... x n x n_components]
        self.eigenvalues_ = L[..., :self.n_components]                                              # [... x n_components]
        return U[..., :, :self.n_components]                                                        # [... x n x n_components]

    def update(self, features: torch.Tensor) -> torch.Tensor:
        d = features.shape[-1]
        n_chunks = ceildiv(features.shape[-2], self.chunk_size)
        if n_chunks > 1:
            """ Chunked version """
            chunks = torch.chunk(features, n_chunks, dim=-2)
            for chunk in chunks:
                self.kernel.update(chunk)
            self._update_to_kernel(d)

            compressed_BBT = 0.0                                                                    # [... x (? + 1) x (? + 1))]
            for chunk in chunks:
                _B = self.kernel.transform(chunk).mT                                                # [... x n x _m]
                _compressed_B = self.Ahinv_VT @ _B                                                  # [... x (? + 1) x _m]
                _compressed_B = torch.nan_to_num(_compressed_B, nan=0.0)
                compressed_BBT = compressed_BBT + _compressed_B @ _compressed_B.mT                  # [... x (? + 1) x (? + 1)]
            self.S = self.S + self.Ahinv_UL @ compressed_BBT @ self.Ahinv_UL.mT                     # [... x n x n]
            US, self.eigenvalues_ = solve_eig(self.S, self.n_components, self.eig_solver)           # [... x n x n_components], [... x n_components]
            self.transform_matrix = self.Ahinv @ US * (self.eigenvalues_[..., None, :] ** -0.5)     # [... x n x n_components]

            VS = []
            for chunk in chunks:
                VS.append(self.kernel.transform(chunk) @ self.transform_matrix)                     # [... x _m x n_components]
            VS = torch.cat(VS, dim=-2)
            return VS                                                                               # [... x m x n_components]
        else:
            """ Unchunked version """
            B = self.kernel.update(features).mT                                                     # [... x n x m]
            self._update_to_kernel(d)
            compressed_B = self.Ahinv_VT @ B                                                        # [... x (? + 1) x m]
            compressed_B = torch.nan_to_num(compressed_B, nan=0.0)

            self.S = self.S + self.Ahinv_UL @ (compressed_B @ compressed_B.mT) @ self.Ahinv_UL.mT   # [... x n x n]
            US, self.eigenvalues_ = solve_eig(self.S, self.n_components, self.eig_solver)           # [... x n x n_components], [... x n_components]
            self.transform_matrix = self.Ahinv @ US * (self.eigenvalues_[..., None, :] ** -0.5)     # [... x n x n_components]

            return B.mT @ self.transform_matrix                                                     # [... x m x n_components]

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        n_chunks = ceildiv(features.shape[-2], self.chunk_size)
        if n_chunks > 1:
            """ Chunked version """
            chunks = torch.chunk(features, n_chunks, dim=-2)
            VS = []
            for chunk in chunks:
                VS.append(self.kernel.transform(chunk) @ self.transform_matrix)                     # [... x _m x n_components]
            VS = torch.cat(VS, dim=-2)
        else:
            """ Unchunked version """
            VS = self.kernel.transform(features) @ self.transform_matrix                            # [... x m x n_components]
        return VS                                                                                   # [... x m x n_components]


class OnlineNystromSubsampleFit(OnlineNystrom):
    def __init__(
        self,
        n_components: int,
        kernel: OnlineKernel,
        distance_type: DistanceOptions,
        sample_config: SampleConfig,
        eig_solver: EigSolverOptions = "svd_lowrank",
        chunk_size: int = 8192,
    ):
        OnlineNystrom.__init__(
            self,
            n_components=n_components,
            kernel=kernel,
            eig_solver=eig_solver,
            chunk_size=chunk_size,
        )
        self.distance_type: DistanceOptions = distance_type
        self.sample_config: SampleConfig = sample_config
        self.sample_config._ncut_obj = copy.deepcopy(self)
        self.anchor_indices: torch.Tensor = None

    def _fit_helper(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _n = features.shape[-2]
        if self.sample_config.num_sample >= _n:
            logging.info(
                f"NCUT nystrom num_sample is larger than number of input samples, nyström approximation is not needed, setting num_sample={_n}"
            )
            self.num_sample = _n

        if precomputed_sampled_indices is not None:
            self.anchor_indices = precomputed_sampled_indices
        else:
            self.anchor_indices = subsample_features(
                features=features,
                distance_type=self.distance_type,
                config=self.sample_config,
            )
        sampled_features = torch.gather(features, -2, self.anchor_indices[..., None].expand([-1] * self.anchor_indices.ndim + [features.shape[-1]]))
        OnlineNystrom.fit(self, sampled_features)

        _n_not_sampled = _n - self.anchor_indices.shape[-1]
        if _n_not_sampled > 0:
            unsampled_mask = torch.full(features.shape[:-1], True, device=features.device).scatter_(-1, self.anchor_indices, False)
            unsampled_indices = torch.where(unsampled_mask)[-1].view((*features.shape[:-2], -1))
            unsampled_features = torch.gather(features, -2, unsampled_indices[..., None].expand([-1] * unsampled_indices.ndim + [features.shape[-1]]))
            V_unsampled = OnlineNystrom.update(self, unsampled_features)
        else:
            unsampled_indices = V_unsampled = None
        return unsampled_indices, V_unsampled

    def fit(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor = None,
    ) -> "OnlineNystromSubsampleFit":
        """Fit Nystrom Normalized Cut on the input features.
        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None
        Returns:
            (NCut): self
        """
        OnlineNystromSubsampleFit._fit_helper(self, features, precomputed_sampled_indices)
        return self

    def fit_transform(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None

        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
        unsampled_indices, V_unsampled = OnlineNystromSubsampleFit._fit_helper(self, features, precomputed_sampled_indices)
        V_sampled = OnlineNystrom.transform(self, self.anchor_features)

        if unsampled_indices is not None:
            V = torch.zeros((*features.shape[:-1], self.n_components), device=features.device)
            for (indices, _V) in [(self.anchor_indices, V_sampled), (unsampled_indices, V_unsampled)]:
                V.scatter_(-2, indices[..., None].expand([-1] * indices.ndim + [self.n_components]), _V)
        else:
            V = V_sampled
        return V


def solve_eig(
    A: torch.Tensor,
    num_eig: int,
    eig_solver: EigSolverOptions,
    eig_value_buffer: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of Eigensolver cut without Nystrom-like approximation.

    Args:
        A (torch.Tensor): input matrix, shape (n_samples, n_samples)
        num_eig (int): number of eigenvectors to return
        eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh']
        eig_value_buffer (float): value added to diagonal to buffer symmetric but non-PSD matrices
    Returns:
        (torch.Tensor): eigenvectors corresponding to the eigenvalues, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues of the eigenvectors, sorted in descending order
    """
    shape: torch.Size = A.shape[:-2]
    A = A.view((-1, *A.shape[-2:]))
    bsz: int = A.shape[0]

    A = A + eig_value_buffer * torch.eye(A.shape[-1], device=A.device)
    num_eig = min(A.shape[-1], num_eig)
    # compute eigenvectors
    if eig_solver == "svd_lowrank":  # default
        # only top q eigenvectors, fastest
        eigen_vector, eigen_value, _ = torch.svd_lowrank(A, q=num_eig)              # complex: [(...) x N x D], [(...) x D]
    elif eig_solver == "lobpcg":
        # only top k eigenvectors, fast
        eigen_value, eigen_vector = torch.lobpcg(A, k=num_eig)
    elif eig_solver == "svd":
        # all eigenvectors, slow
        eigen_vector, eigen_value, _ = torch.svd(A)
    elif eig_solver == "eigh":
        # all eigenvectors, slow
        eigen_value, eigen_vector = torch.linalg.eigh(A)
    else:
        raise ValueError(
            "eigen_solver should be 'lobpcg', 'svd_lowrank', 'svd' or 'eigh'"
        )
    eigen_value = eigen_value - eig_value_buffer

    # sort eigenvectors by eigenvalues, take top (descending order)
    indices = torch.topk(eigen_value.abs(), k=num_eig, dim=-1).indices              # int: [(...) x S]
    eigen_value = eigen_value[torch.arange(bsz)[:, None], indices]                  # complex: [(...) x S]
    eigen_vector = eigen_vector[torch.arange(bsz)[:, None], :, indices].mT          # complex: [(...) x N x S]

    # correct the random rotation (flipping sign) of eigenvectors
    sign = torch.sign(torch.sum(eigen_vector.real, dim=-2, keepdim=True))           # float: [(...) x 1 x S]
    sign[sign == 0] = 1.0
    eigen_vector = eigen_vector * sign

    eigen_value = eigen_value.view((*shape, *eigen_value.shape[-1:]))               # complex: [... x S]
    eigen_vector = eigen_vector.view((*shape, *eigen_vector.shape[-2:]))            # complex: [... x N x S]
    return eigen_vector, eigen_value
