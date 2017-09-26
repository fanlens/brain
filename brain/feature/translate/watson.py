#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Union, List, Optional, Dict, Tuple

from watson_developer_cloud import LanguageTranslatorV2

from config import get_config
from db.models.activities import Lang
from ..language_detect import language_detect

_config = get_config(max_depth=3)

_language_translator = LanguageTranslatorV2(
    url=_config.get('WATSON', 'url'),
    username=_config.get('WATSON', 'username'),
    password=_config.get('WATSON', 'password'))

# todo: only limited set of languages for now
# _identifiable_languages = {Lang[lang_str_obj['language']] for lang_str_obj in
#                            _language_translator.get_identifiable_languages().get('languages', [])
#                            if lang_str_obj['language'] in Lang.__members__}
_identifiable_language = {Lang.de, Lang.es}


def translate(texts: Union[str, List[str]],
              target_language: Lang = Lang.en,
              source_languages: Optional[Union[Lang, List[Lang]]] = None) -> List[str]:
    """
    :param texts: text(s) to translate
    :param target_language: which language to translate to
    :param source_languages: which language are the text(s) in? optional, if not provided the language will be auto detected
    :return: a list of translations, single translations can be None if none could be found
    """
    # convert both texts and langs into lists
    if not isinstance(texts, list):
        texts = [texts]

    if source_languages is not None:
        if not isinstance(source_languages, list):
            source_languages = [source_languages]
        assert len(texts) == len(source_languages)
    else:
        source_languages = [Lang[language_detect(text)] for text in texts]

    # group the texts into batches keyed by their source language
    # this is done because the watson api only allows a scalar value for the source language but mutliple texts
    grouped = defaultdict(lambda: list())  # type: Dict[Lang, List[Tuple[int, str]]]
    for orig_idx, (text, source_language) in enumerate(zip(texts, source_languages)):
        grouped[source_language].append((orig_idx, text))

    # translate all texts and add the keyed translations into an unsorted list
    unsorted_translations = []
    for source_language in grouped.keys():
        language_group = grouped[source_language]

        if source_language == target_language:
            group_translations_with_idx = language_group
        else:
            group_orig_idxs, group_texts = zip(*language_group)
            if source_language in _identifiable_languages:
                group_translation_data = dict(text=group_texts,
                                              source=source_language.name,
                                              target=target_language.name)
                # the default method only allows for one text it seems
                group_translations = _language_translator.request(method='POST',
                                                                  url='/v2/translate',
                                                                  json=group_translation_data,
                                                                  accept_json=True)
                group_translations_with_idx = zip(group_orig_idxs,
                                                  [row.get('translation') for row in
                                                   group_translations.get('translations')])
            else:
                group_translations_with_idx = zip(group_orig_idxs, [None] * len(group_orig_idxs))

        unsorted_translations.extend(group_translations_with_idx)

    # sort the list back to the original idx
    _, translations = zip(*sorted(unsorted_translations, key=lambda tup: tup[0]))
    return list(translations)


__all__ = [translate.__name__]

if __name__ == '__main__':
    string_vec = ['hallo du schnöde welt', 'Adiós muchachos, compañeros de mi vida', 'ade du schnöde welt']
    print(translate(string_vec))
