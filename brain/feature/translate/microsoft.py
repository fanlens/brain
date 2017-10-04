#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Microsoft translation implementation"""
import warnings
from typing import Union, List

from mstranslator import Translator

from config import get_config
from db.models.activities import Lang

_CONFIG = get_config(max_depth=3)
_KEY = _CONFIG.get('AZURE', 'key')


def translate(texts: Union[str, List[str]], target_language: Lang = Lang.en) -> List[str]:
    """Microsoft translation implementation, see `brain.feature.translate`"""
    warnings.warn("Azure is not used as translation service anymore", DeprecationWarning, stacklevel=2)
    if not isinstance(texts, list):
        texts = [texts]
    # todo: reuse enum from language_detect
    translate_client = Translator(_KEY)
    translations = [translate_client.translate(t, lang_to=target_language.name) for t in texts]
    if not translations:
        raise Exception('no translations')

    return translations


__all__ = [translate.__name__]

if __name__ == '__main__':
    STRING_VEC = ['hallo welt', 'wie geht es dir heute, an diesem schönen tag']
    print(translate(STRING_VEC))
