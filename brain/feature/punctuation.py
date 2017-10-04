#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a punctuation based tokenizer"""

import re
from typing import Iterable, Any, Optional, Union, Dict

import numpy as np
from sklearn.base import TransformerMixin

from . import output, TOutputType, TOutput, to_output_type


class PunctuationTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument
    """extract all punctuation"""
    strict_punctuation_pattern = re.compile(r'[!?.,:;\'"]')
    noisy_punctuation_pattern = re.compile(r'[^\w\d\s]')

    def __init__(self, strict: bool = True, output_type: TOutputType = dict) -> None:
        self._output_type = output_type
        self._strict = strict

    def transform(self, X: Iterable[str], y: Optional[np.ndarray] = None, **transform_params: Any) -> Iterable[TOutput]:
        """
        :param X: source of texts to extract punctuation for
        :param y: unused
        :param transform_params: unused
        :return: extracted punctuation according to output type
        """
        return [self(x) for x in X]

    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None, **fit_params: Any) -> 'PunctuationTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Union[bool, TOutputType]]:
        """Get parameters for this Transformer."""
        return dict(output_type=self._output_type, strict=True)

    def set_params(self, **params: Union[bool, TOutputType]) -> None:
        """Set parameters for this Transformer."""
        strict = params.get('strict', self._strict)
        assert isinstance(strict, bool)
        self._strict = strict

        self._output_type = to_output_type(params.get('output_type', self._output_type))

    def __call__(self, doc: str) -> TOutput:
        """
        :param doc: text sample to extract punctuation from
        :return: extracted punctuation according to output type
        """
        pattern = self.strict_punctuation_pattern if self._strict else self.noisy_punctuation_pattern
        punctuation = pattern.findall(doc)
        return output(self._output_type, punctuation)
