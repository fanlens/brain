#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import retinasdk

from scipy.sparse import lil_matrix
from sklearn.base import TransformerMixin
from config.db import Config

retina = retinasdk.FullClient(Config("cortical")["api_key"])


def get_fingerprints(texts):
    return retina.getFingerprintsForTexts(texts)


def get_fingerprint(text):
    return retina.getFingerprintForText(text)


def unpack_fingerprint(sample):
    sample_1d = numpy.zeros(128 * 128, dtype=numpy.float64)
    sample_1d[numpy.array(sample, dtype=numpy.uint16)] = 1.0
    return sample_1d


def sparsify_fingerprint(sample):
    sparse = lil_matrix((1, 128 * 128), dtype=numpy.float64)
    sparse[0, sample] = 1.0
    return sparse


class DenseFingerprintTransformer(TransformerMixin):
    _vecfun = numpy.vectorize(unpack_fingerprint)

    def transform(self, X: numpy.array, y=None, **transform_params):
        return numpy.array([unpack_fingerprint(x) for x in X], dtype=numpy.float64)

    def fit(self, X: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return {}

    def set_params(self, **_):
        pass


class SparseFingerprintTransformer(TransformerMixin):
    def transform(self, X, y=None, **transform_params):
        sparse = lil_matrix((len(X), 128 * 128), dtype=numpy.float64)
        for idx, x in enumerate(X):
            sparse[idx, x] = 1.0
        return sparse

    def fit(self, X: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return {}

    def set_params(self, **_):
        pass
