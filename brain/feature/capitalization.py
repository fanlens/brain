"""Module for a capitalization based tokenizer"""

import re
from typing import Optional, Iterable, Any, Dict

import numpy as np
from sklearn.base import TransformerMixin


class CapitalizationTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument
    """extract all capitalization"""
    uppercase_pattern = re.compile(r'[A-Z]')  # todo: also unicode characters?
    lowercase_pattern = re.compile(r'[a-z]')

    def __init__(self, fraction: bool = True) -> None:
        self._fraction = fraction

    def transform(self,
                  X: Iterable[str],
                  y: Optional[np.ndarray] = None,
                  **transform_params: Any) -> Iterable[Iterable[float]]:
        """
        :param X: List of text samples to transform
        :param y: unused
        :param transform_params: unused
        :return: List of floats as singleton lists to keep dimension
        """
        return [[self(x)] for x in X]

    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None, **fit_params: Any) -> 'CapitalizationTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Get parameters for this Transformer."""
        return dict(fraction=self._fraction)

    def set_params(self, **params: bool) -> None:
        """Set parameters for this Transformer."""
        self._fraction = params.get('fraction', self._fraction)

    def __call__(self, doc: str) -> float:
        """
        :param doc: the text sample to convert
        :return: the amount of captialized letters or their fraction depending on configuration
        """
        num_uppercase = len(self.uppercase_pattern.findall(doc))
        if not self._fraction:
            return num_uppercase

        num_lowercase = len(self.lowercase_pattern.findall(doc))
        return num_uppercase / max((num_uppercase + num_lowercase), 1)
