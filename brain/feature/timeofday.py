#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a lemma based tokenizer"""

import numpy
import dateutil.parser
import datetime
from scipy.sparse import lil_matrix
from sklearn.base import TransformerMixin


class TimeOfDayTransformer(TransformerMixin):
    def __init__(self, dense=False, resolution=1):
        """
        :param dense: should a dense numpy array be returned?
        :param resolution: how many ticks per hour? default 1
        """
        self._dense = dense
        self._resolution = resolution

    def transform(self, X, y=None, **transform_params):
        num_ticks = 24 * self._resolution
        if self._dense:
            mat = numpy.zeros((len(X), num_ticks), dtype=numpy.float64)
        else:
            mat = lil_matrix((len(X), num_ticks), dtype=numpy.float64)
        for idx, x in enumerate(X):
            mat[idx, self(x)] = 1
        return mat

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(dense=self._dense, resolution=self._resolution)

    def set_params(self, **params):
        self._dense = params.get('dense', self._dense)
        self._resolution = params.get('resolution', self._resolution)

    def __call__(self, timestamp):
        if isinstance(timestamp, int):
            if 0 <= timestamp < 24:  # in hour format
                return self._format_return(timestamp * self._resolution)
            else:  # interpreted as unix time
                timestamp = datetime.datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):  # assuming parsable
            timestamp = dateutil.parser.parse(timestamp)

        assert isinstance(timestamp, datetime.datetime)
        midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        minutes = (timestamp - midnight).seconds / 60
        tick = int(round(minutes / (60 / self._resolution))) % (24 * self._resolution)  # 24 -> 0 via modulo
        return tick
