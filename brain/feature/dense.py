#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):
    """
    transformer step that transforms sparse matrices to dense ones
    for algorithms that don't support sparse matrices
    """

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return {}
