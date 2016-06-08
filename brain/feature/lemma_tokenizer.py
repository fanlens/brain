#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a lemma based tokenizer"""

import functools
import re
import numpy
from sklearn.base import TransformerMixin
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords


def compose(*functions):
    """convenience function to chain transformations"""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


class LemmaTokenTransformer(TransformerMixin):
    """tokenize based on lemmas"""
    _ident = lambda _: _
    _urls_reg = re.compile(r"(?:(http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?")
    _word_reg = re.compile(r"^[\w\d]+$")

    _stops = set(stopwords.words('english'))
    _wnl = WordNetLemmatizer()

    def __init__(self, short_url=False, output_type='string'):
        self._short_url = short_url
        self._output_type = output_type

    def transform(self, X: numpy.array, y=None, **transform_params):
        vecfun = numpy.vectorize(self.__call__)
        return vecfun(X)

    def fit(self, X: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(extractor=self._ident, short_url=self._short_url, output_type=self._output_type)

    def _lemmatize(self, tokens: list) -> list:
        return [self._wnl.lemmatize(w).lower() for w, t in tokens if t[:2] in ('NN', 'VB', 'JJ', 'RB') or True]

    def _stop(self, tokens: list) -> list:
        return [t for t in tokens if t not in self._stops]

    def __call__(self, doc):
        urls = self._urls_reg.findall(doc)
        if self._short_url:
            urls = [main for _, main, _ in urls]
        else:
            urls = ['%s://%s%s' % parts for parts in urls]
        tokenized = compose(
            functools.partial(self._urls_reg.sub, ' '),
            word_tokenize,
            functools.partial(filter, self._word_reg.match),
            self._stop,
            pos_tag,
            self._lemmatize,
            self._stop,  # lets do it before and after for good measure
            lambda res: res + urls
        )(doc)
        if self._output_type == 'string':
            return ' '.join(tokenized)
        else:
            return tokenized
