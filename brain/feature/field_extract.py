#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Transformer to extract fields from multip field samples"""
from typing import Iterable, Any, Optional, Mapping, Dict, Union

import numpy as np

from sklearn.base import TransformerMixin

TKey = Union[str, int]


class FieldExtractTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument
    """extract fields from arbitrary objects that have attribute access mechanisms"""

    def __init__(self, key: TKey) -> None:
        self._key = key

    def transform(self,
                  X: Iterable[Mapping[TKey, Any]],
                  y: Optional[np.ndarray] = None,
                  **transform_params: Any) -> Iterable[Any]:
        """
        :param X: list of mapping type string-subscriptable objects
        :param y: unused
        :param transform_params: unused
        :return: projection of the input list to the key specified
        """
        return [x[self._key] for x in X]

    def fit(self,
            X: Iterable[Mapping[TKey, Any]],
            y: Optional[np.ndarray] = None,
            **fit_params: Any) -> 'FieldExtractTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[str, TKey]:
        """Get parameters for this Transformer."""
        return dict(key=self._key)

    def set_params(self, **params: TKey) -> None:
        """Set parameters for this Transformer."""
        self._key = params.get('key', self._key)
