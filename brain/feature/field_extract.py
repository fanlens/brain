#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a lemma based tokenizer"""

from sklearn.base import TransformerMixin
from functools import singledispatch


class FieldExtractTransformer(TransformerMixin):
    """extract fields from arbitrary objects that have attribute access mechanisms"""

    def __init__(self, key=None):
        assert key is not None
        self._key = key

    def transform(self, X, y=None, **transform_params):
        return [x[self._key] for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(key=self._key)
