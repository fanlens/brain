#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import pycld2
from sklearn.base import TransformerMixin


def language_detect(text: str) -> str:
    """:return: the language short string detected in the text"""
    if isinstance(text, tuple):
        text = text[0]
    return pycld2.detect(text)[2][0][1]


def is_english(text: str) -> bool:
    return language_detect(text) == 'en'


def is_german(text: str) -> bool:
    return language_detect(text) == 'de'


class LanguageTransformer(TransformerMixin):
    _vecfun = numpy.vectorize(language_detect)

    def transform(self, X: numpy.array, y=None, **transform_params):
        return self._vecfun(X)

    def fit(self, X: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return {}

    def set_params(self, **_):
        pass
