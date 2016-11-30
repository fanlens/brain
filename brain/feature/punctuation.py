#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a punctuation based tokenizer"""

import typing
import re
from sklearn.base import TransformerMixin

from . import output


class PunctuationTransformer(TransformerMixin):
    """extract all punctuation"""
    strict_punctuation_pattern = re.compile(r'[!?.,:;\'"]')
    noisy_punctuation_pattern = re.compile(r'[^\w\d\s]')

    def __init__(self, strict=True, output_type: typing.Union[list, str, dict] = dict):
        self._output_type = output_type
        self._strict = strict

    def transform(self, X, y=None, **transform_params):
        return [self(x) for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(output_type=self._output_type, strict=True)

    def set_params(self, **params):
        self._output_type = params.get('output_type', self._output_type)
        self._strict = params.get('strict', self._strict)

    def __call__(self, doc):
        pattern = self.strict_punctuation_pattern if self._strict else self.noisy_punctuation_pattern
        punctuation = pattern.findall(doc)
        return output(self._output_type, punctuation)
