#!/usr/bin/env python
# -*- coding: utf-8 -*-
import typing

import numpy
from google.cloud import translate as google_translate
from sklearn.base import TransformerMixin
import html


def translate(text: typing.Union[typing.AnyStr, typing.List[typing.AnyStr]], target_language='en', short_output=True):
    # todo: reuse enum from language_detect
    translate_client = google_translate.Client()
    translations = translate_client.translate(text, target_language=target_language)
    head_only = False
    if not isinstance(translations, list):
        translations = [translations]
        head_only = True

    if short_output:
        output = [html.unescape(translation.get('translatedText', '')) for translation in translations]
    else:
        output = translations
    if len(output) == 0:
        raise Exception('no translations')

    return output if not head_only else output[0]


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
    string_vec = ['hallo welt', 'ade du schnöde welt']
    transformer = TranslationTransformer()
    print(transformer.fit_transform(string_vec))
