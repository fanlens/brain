#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a lemma based tokenizer"""

import functools
import re
import numpy
from sklearn.base import TransformerMixin
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords


class FieldExtractTransformer(TransformerMixin):
    """tokenize based on lemmas"""

    def __init__(self, key=None):
        assert key is not None
        self._key = key

    def transform(self, X: numpy.array, y=None, **transform_params):
        return X[self._key]

    def fit(self, X: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(key=self._key)
