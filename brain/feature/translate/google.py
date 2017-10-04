#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Google translation implementation"""
import html
import warnings
from typing import Union, List

from google.cloud import translate as google_translate  # false positive, setup.py pylint: disable=no-name-in-module

from common.db.models.activities import Lang


def translate(texts: Union[str, List[str]], target_language: Lang = Lang.en, short_output: bool = True) -> List[str]:
    """Google translation implementation, see `brain.feature.translate`"""
    warnings.warn("Google is not used as translation service anymore", DeprecationWarning, stacklevel=2)
    translate_client = google_translate.Client()
    response: Union[List[dict], dict] = translate_client.translate(texts, target_language=target_language.name)
    translations: List[dict] = []
    if not isinstance(response, list):
        translations.append(response)
    else:
        translations = response

    assert all(isinstance(translation, dict) for translation in translations)

    if short_output:
        output = [html.unescape(translation.get('translatedText', '')) for translation in translations]
    else:
        output = translations
    if not output:
        raise Exception('no translations')

    return output


__all__ = [translate.__name__]

if __name__ == '__main__':
    STRING_VEC = ['hallo welt', 'ade du schnöde welt']
    print(translate(STRING_VEC))
