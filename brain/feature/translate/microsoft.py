#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from typing import Union, List

from mstranslator import Translator

from config import get_config
from db.models.activities import Lang

_config = get_config(max_depth=3)
_key = _config.get('AZURE', 'key')


def translate(texts: Union[str, List[str]], target_language: Lang = Lang.en) -> List[str]:
    warnings.warn("Azure is not used as translation service anymore", DeprecationWarning, stacklevel=2)
    if not isinstance(texts, list):
        texts = [texts]
    # todo: reuse enum from language_detect
    translate_client = Translator(_key)
    translations = [translate_client.translate(t, lang_to=target_language.name) for t in texts]
    if len(translations) == 0:
        raise Exception('no translations')

    return translations


__all__ = [translate.__name__]

if __name__ == '__main__':
    string_vec = ['hallo welt', 'wie geht es dir heute, an diesem schönen tag']
    print(translate(string_vec))
