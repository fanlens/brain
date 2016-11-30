#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a lemma based tokenizer"""

import functools
import re
import typing

from brain.feature import output
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin

_ident = lambda _: _
_urls_reg = re.compile(r"(?:(http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?")
_word_reg = re.compile(r"^[\w\d]+$")

_stops = set(stopwords.words('english'))
_wnl = WordNetLemmatizer()


def compose(*functions):
    """convenience function to chain transformations"""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


class LemmaTokenTransformer(TransformerMixin):
    """tokenize based on lemmas"""

    def __init__(self, pass_through=False, short_url=False, output_type: typing.Union[list, str, dict] = dict):
        """
        :param pass_through: perform simple split on whitespace
        :param short_url: remove scheme and path from url only leaving domain
        :param output_type: list of tokens, or ' ' seperated string of tokens, or token->count dict
        """
        self._short_url = short_url
        self._output_type = output_type
        self._pass_through = pass_through

    def transform(self, X: typing.List[typing.AnyStr], y=None, **transform_params):
        return [self(x) for x in X]

    def fit(self, X: typing.List[typing.AnyStr], y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(short_url=self._short_url, output_type=self._output_type)

    def set_params(self, **params):
        self._short_url = params.get('short_url', self._short_url)
        self._output_type = params.get('output_type', self._output_type)

    def _lemmatize(self, tokens: list) -> list:
        return [_wnl.lemmatize(w).lower() for w, t in tokens if t[:2] in ('NN', 'VB', 'JJ', 'RB')]

    def _stop(self, tokens: list) -> list:
        return [t for t in tokens if t not in _stops]

    def __call__(self, doc):
        urls = _urls_reg.findall(doc)
        if self._short_url:
            urls = [main for _, main, _ in urls]
        else:
            urls = ['%s://%s%s' % parts for parts in urls]
        if self._pass_through:
            tokenized = doc.split()
        else:
            tokenized = compose(
                functools.partial(_urls_reg.sub, ' '),
                word_tokenize,
                functools.partial(filter, _word_reg.match),
                self._stop,
                pos_tag,
                self._lemmatize,
                lambda res: res + urls
            )(doc)
        return output(self._output_type, tokenized)


if __name__ == "__main__":
    tokenizer = LemmaTokenTransformer(pass_through=False)
    tokens = tokenizer.fit_transform([
        """To add overloaded implementations to the function, use the register() attribute of the
        generic function. It is a decorator, taking a type parameter and decorating a function implementing the
        operation for that type"""
    ])
    print(tokens)
