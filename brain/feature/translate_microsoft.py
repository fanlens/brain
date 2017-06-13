#!/usr/bin/env python
# -*- coding: utf-8 -*-
import typing

import numpy
import html
from config.db import Config
from sklearn.base import TransformerMixin
from mstranslator import Translator


_key = Config("azure")["translator"]

def translate(text: typing.Union[typing.AnyStr, typing.List[typing.AnyStr]], target_language='en', short_output=True):
    if not isinstance(text, list):
        text = [text]
    # todo: reuse enum from language_detect
    translate_client = Translator(_key)
    translations = [translate_client.translate(t, lang_to=target_language) for t in text]
    if len(translations) == 0:
        raise Exception('no translations')

    return translations


class TranslationTransformer(TransformerMixin):
    def __init__(self, target_language='en'):
        self._target_language = target_language

    def transform(self, X: typing.List[typing.AnyStr], y=None, **transform_params):
        return translate(X, target_language=self._target_language)

    def fit(self, X: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(target_language='en')

    def set_params(self, **params):
        self._target_language = params.get('target_language', self._target_language)


if __name__ == '__main__':
    string_vec = ['hallo welt', 'wie geht es dir heute, an diesem schönen tag']
    transformer = TranslationTransformer()
    print(transformer.fit_transform(string_vec))
