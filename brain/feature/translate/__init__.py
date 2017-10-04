#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Translation transformers with configurable translation engine"""
from typing import Union, List, Optional, Any, Iterable, Dict, Callable, cast

import numpy as np
from sklearn.base import TransformerMixin

from config import get_config
from db.models.activities import Lang

_CONFIG = get_config(max_depth=3)
TRANSLATION_IMPLEMENTATION = _CONFIG.get('BRAIN', 'translation_implementation')

_TTRANSLATE_IMPL = Callable[[Union[str, List[str]], Lang], List[str]]

if TRANSLATION_IMPLEMENTATION == 'google':
    from .google import translate as translate_google

    _TRANSLATE_IMPL = cast(_TTRANSLATE_IMPL, translate_google)
elif TRANSLATION_IMPLEMENTATION == 'microsoft':
    from .microsoft import translate as translate_microsoft

    _TRANSLATE_IMPL = cast(_TTRANSLATE_IMPL, translate_microsoft)
elif TRANSLATION_IMPLEMENTATION == 'watson':
    from .watson import translate as translate_watson

    _TRANSLATE_IMPL = cast(_TTRANSLATE_IMPL, translate_watson)
else:
    raise NotImplementedError('No implementation for specified language platform: ' + TRANSLATION_IMPLEMENTATION)


def translate(text: Union[str, List[str]], target_language: Lang = Lang.en) -> List[str]:
    """
    translate a text or a list of text to the specified target_language
    :param text: text or list of texts to be translated
    :param target_language: which language to translate to, default: english
    :returns a list of translations
    """
    return _TRANSLATE_IMPL(text, target_language)


class TranslationTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument
    """translation function wrapped into a Scikit Transformer"""

    def __init__(self, target_language: Lang = Lang.en) -> None:
        """:param target_language: which language to translate to, default: english"""
        self._target_language = target_language

    def transform(self, X: Iterable[str], y: Optional[np.ndarray] = None, **transform_params: Any) -> Iterable[str]:
        """
        :param X: list of strings to be translated
        :param y: unused
        :param transform_params: unused
        :return: translated texts
        """
        return translate(list(X), target_language=self._target_language)

    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None, **fit_params: Any) -> 'TranslationTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Lang]:
        """Get parameters for this Transformer."""
        return dict(target_language=self._target_language)

    def set_params(self, **params: Lang) -> None:
        """Set parameters for this Transformer."""
        target_language = params.get('target_language', self._target_language)
        assert isinstance(target_language, Lang)
        self._target_language = target_language


__all__ = [translate.__name__, TranslationTransformer.__name__]
