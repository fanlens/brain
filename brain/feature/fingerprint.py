#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import retinasdk
from sklearn.base import TransformerMixin

from config.db import Config

retina = retinasdk.FullClient(Config("cortical")["api_key"])


def get_fingerprints(texts):
    return retina.getFingerprintsForTexts(texts)


def get_fingerprint(text):
    return retina.getFingerprintForText(text)


class FingerprintTransformer(TransformerMixin):
    # todo inefficient! use batching
    _vecfun = numpy.vectorize(get_fingerprint)

    def transform(self, X: numpy.array, y=None, **transform_params):
        return self._vecfun(X)

    def fit(self, X: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return {}
