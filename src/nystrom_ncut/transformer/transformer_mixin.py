from abc import abstractmethod
from typing import Any

import torch


class TorchTransformerMixin:
    """Mixin class for all transformers in scikit-learn.

    This mixin defines the following functionality:

    - a `fit_transform` method that delegates to `fit` and `transform`;
    - a `set_output` method to output `X` as a specific container type.

    If :term:`get_feature_names_out` is defined, then :class:`BaseEstimator` will
    automatically wrap `transform` and `fit_transform` to follow the `set_output`
    API. See the :ref:`developer_api_set_output` for details.

    :class:`OneToOneFeatureMixin` and
    :class:`ClassNamePrefixFeaturesOutMixin` are helpful mixins for
    defining :term:`get_feature_names_out`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, TransformerMixin
    >>> class MyTransformer(TransformerMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         return self
    ...     def transform(self, X):
    ...         return np.full(shape=len(X), fill_value=self.param)
    >>> transformer = MyTransformer()
    >>> X = [[1, 2], [2, 3], [3, 4]]
    >>> transformer.fit_transform(X)
    array([1, 1, 1])
    """
    def __init__(self):
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: torch.Tensor, **fit_kwargs: Any) -> "TorchTransformerMixin":
        """"""

    @abstractmethod
    def transform(self, X: torch.Tensor, **transform_kwargs: Any) -> torch.Tensor:
        """"""

    @abstractmethod
    def fit_transform(self, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """"""
