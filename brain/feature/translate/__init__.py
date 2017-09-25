#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union, List

import numpy
from sklearn.base import TransformerMixin

from config import get_config
from db.models.activities import Lang

_config = get_config(max_depth=3)
TRANSLATION_IMPLEMENTATION = _config.get('BRAIN', 'translation_implementation')

if TRANSLATION_IMPLEMENTATION == 'google':
    from .google import translate as translate_impl
elif TRANSLATION_IMPLEMENTATION == 'microsoft':
    from .microsoft import translate as translate_impl
elif TRANSLATION_IMPLEMENTATION == 'watson':
    from .watson import translate as translate_impl
else:
    raise NotImplementedError('No implementation for specified language platform: ' + TRANSLATION_IMPLEMENTATION)


def translate(text: Union[str, List[str]], target_language: Lang = Lang.en) -> List[str]:
    """
    translate a text or a list of text to the specified target_language
    :param text: text or list of texts to be translated
    :param target_language: which language to translate to, default: english
    :returns a list of translations
    """
    return translate_impl(text, target_language)


class TranslationTransformer(TransformerMixin):
    """translation function wrapped into a Scikit Transformer"""

    def __init__(self, target_language: Lang = Lang.en):
        """:param target_language: which language to translate to, default: english"""
        self._target_language = target_language

    def transform(self, xs: List[str], y=None, **transform_params):
        return translate(xs, target_language=self._target_language)

    def fit(self, xs: numpy.array, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(target_language=self._target_language)

    def set_params(self, **params):
        self._target_language = params.get('target_language', self._target_language)


__all__ = [translate.__name__, TranslationTransformer.__name__]

if __name__ == '__main__':
    string_vec = ['hallo welt', 'ade du schnöde welt']
    transformer = TranslationTransformer()
    print(transformer.fit_transform(string_vec))
