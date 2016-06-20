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


def unpack_fingerprint(sample):
    sample_1d = numpy.zeros(128 * 128, dtype='u1')
    sample_1d[numpy.array(sample, dtype='u4')] = 1
    return sample_1d


class DenseFingerprintTransformer(TransformerMixin):
    _vecfun = numpy.vectorize(unpack_fingerprint)

    def transform(self, X: numpy.array, y=None, **transform_params):
        return numpy.array([unpack_fingerprint(x) for x in X], dtype='u1')

    def fit(self, X: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return {}
