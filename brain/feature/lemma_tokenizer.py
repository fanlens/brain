#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a lemma based tokenizer"""

import functools
import re
from typing import Iterable, Optional, Any, Union, Dict, Tuple

import numpy as np
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin

from utils.compose import compose
from . import output, TOutput, TOutputType, to_output_type

_URLS_REG = re.compile(r"(?:(http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?")
_WORD_REG = re.compile(r"^[\w\d]+$")

_STOPS = set(stopwords.words('english'))
_WORD_NET_LEMMATIZER = WordNetLemmatizer()


def lemmatize(tokens: Iterable[Tuple[str, str]]) -> Iterable[str]:
    """
    Lemmatize the pos tagged sequence
    :param tokens: pos tagged sequence
    :return: lemmatized sequence
    """
    return [_WORD_NET_LEMMATIZER.lemmatize(w).lower() for w, t in tokens if t[:2] in ('NN', 'VB', 'JJ', 'RB')]


def stop(tokens: Iterable[str]) -> Iterable[str]:
    """
    Filter stop words from word tokens
    :param tokens: source of words
    :return: smaller list with stop words removed
    """
    return [t for t in tokens if t not in _STOPS]


class LemmaTokenTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument
    """tokenize based on lemmas"""

    def __init__(self, pass_through: bool = False, short_url: bool = False, output_type: TOutputType = dict) -> None:
        """
        :param pass_through: perform simple split on whitespace
        :param short_url: remove scheme and path from url only leaving domain
        :param output_type: list of tokens, or ' ' seperated string of tokens, or token->count dict
        """
        self._short_url = short_url
        self._output_type = output_type
        self._pass_through = pass_through

    def transform(self, X: Iterable[str], y: Optional[np.ndarray] = None, **transform_params: Any) -> Iterable[TOutput]:
        """
        :param X: List of text samples to transform
        :param y: unused
        :param transform_params: unused
        :return: lemmatized strings in output format specified
        """
        return [self(x) for x in X]

    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None, **fit_params: Any) -> 'LemmaTokenTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Union[bool, TOutputType]]:
        """Get parameters for this Transformer."""
        return dict(short_url=self._short_url, output_type=self._output_type)

    def set_params(self, **params: Union[bool, TOutputType]) -> None:
        """Set parameters for this Transformer."""
        short_url = params.get('short_url', self._short_url)
        assert isinstance(short_url, bool)
        self._short_url = short_url

        self._output_type = to_output_type(params.get('output_type', self._output_type))

    def __call__(self, doc: str) -> TOutput:
        urls = _URLS_REG.findall(doc)
        if self._short_url:
            urls = [main for _, main, _ in urls]
        else:
            urls = ['%s://%s%s' % parts for parts in urls]
        if self._pass_through:
            tokenized = doc.split()
        else:
            tokenized = compose(
                functools.partial(_URLS_REG.sub, ' '),
                word_tokenize,
                functools.partial(filter, _WORD_REG.match),
                stop,
                pos_tag,
                lemmatize,
                lambda res: res + urls
            )(doc)
        return output(self._output_type, tokenized)


if __name__ == "__main__":
    _TOKENIZER = LemmaTokenTransformer(pass_through=False)
    _TOKENS = _TOKENIZER.fit_transform([
        """To add overloaded implementations to the function, use the register() attribute of the
        generic function. It is a decorator, taking a type parameter and decorating a function implementing the
        operation for that type"""
    ])
    print(_TOKENS)
