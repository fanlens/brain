#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a punctuation based tokenizer"""

import re

from sklearn.base import TransformerMixin


class CapitalizationTransformer(TransformerMixin):
    """extract all capitalization"""
    # todo: also unicode characters
    uppercase_pattern = re.compile(r'[A-Z]')
    lowercase_pattern = re.compile(r'[a-z]')

    def __init__(self, fraction=True):
        self._fraction = fraction

    def transform(self, X, y=None, **transform_params):
        # todo: make output type configurable?
        return [[self(x)] for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(fraction=True)

    def set_params(self, **params):
        self._fraction = params.get('fraction', self._fraction)

    def __call__(self, doc):
        num_uppercase = len(self.uppercase_pattern.findall(doc))
        if not self._fraction:
            return num_uppercase

        num_lowercase = len(self.lowercase_pattern.findall(doc))
        return num_uppercase / (num_uppercase + num_lowercase)
