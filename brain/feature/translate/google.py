#!/usr/bin/env python
# -*- coding: utf-8 -*-
import html
import warnings
from typing import Union, List

from google.cloud import translate as google_translate

from db.models.activities import Lang


def translate(texts: Union[str, List[str]], target_language: Lang = Lang.en, short_output: bool = True) -> List[str]:
    warnings.warn("Google is not used as translation service anymore", DeprecationWarning, stacklevel=2)
    translate_client = google_translate.Client()
    translations = translate_client.translate(texts, target_language=target_language.name)
    head_only = False
    if not isinstance(translations, list):
        translations: List[dict] = [translations]
        head_only = True

    if short_output:
        output = [html.unescape(translation.get('translatedText', '')) for translation in translations]
    else:
        output = translations
    if len(output) == 0:
        raise Exception('no translations')

    return output if not head_only else output[0]


__all__ = [translate.__name__]

if __name__ == '__main__':
    string_vec = ['hallo welt', 'ade du schnöde welt']
    print(translate(string_vec))
