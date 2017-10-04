#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Transform sparse matrices into dense matrices."""
from typing import Optional, Any, Dict
import numpy as np
import scipy.sparse
from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument,no-self-use
    """
    transformer step that transforms sparse matrices to dense ones
    for algorithms that don't support sparse matrices
    """

    def transform(self, X: scipy.sparse.spmatrix, y: Optional[np.ndarray] = None, **fit_params: Any) -> np.ndarray:
        """Convert sparse matrix into dense matrix"""
        return X.todense()

    def fit(self, X: scipy.sparse.spmatrix, y: Optional[np.ndarray] = None, **fit_params: Any) -> 'DenseTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[None, None]:
        """Get parameters for this Transformer."""
        return {}

    def set_params(self, **params: Any) -> None:
        """Set parameters for this Transformer."""
        pass
