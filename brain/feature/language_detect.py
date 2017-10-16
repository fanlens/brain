"""Language detection tools for feature extraction"""
import enum
import logging
from typing import Union, Any, Dict, Iterable, Optional, cast

import numpy as np
import pycld2
from sklearn.base import TransformerMixin

_LOGGER = logging.getLogger()


def _create_enum_type(to_string: bool = True) -> Union[str, enum.Enum]:
    """
    Create a new enum type at runtime, mostly for internal use
    :param to_string:
    :return:
    """
    members = dict((code, value) for value, code in pycld2.LANGUAGES)
    members['un'] = 'UNKNOWN'
    if to_string:
        return 'Enum(\'Lang\', %s)' % str(members)
    return enum.Enum('Lang', members)


def language_detect(text: str) -> str:
    """:return: the language short string detected in the text or 'un' if an error occurs"""
    if isinstance(text, tuple):
        text = text[0]
    try:
        return cast(str, pycld2.detect(text.replace('\x7f', '').replace('\b', ''))[2][0][1])
    except pycld2.error:
        _LOGGER.exception('couldn\'t process input: %s', text)
        return 'un'


def is_english(text: str) -> bool:
    """
    :param text: text to check
    :return: is text in English?
    """
    return language_detect(text) == 'en'


def is_german(text: str) -> bool:
    """
    :param text: text to check
    :return: is text in German?
    """
    return language_detect(text) == 'de'


class LanguageTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument,no-self-use
    """transform texts into their language short string e.g. ['hallo'] -> ['de']"""

    def transform(self, X: Iterable[str], y: Optional[np.ndarray] = None, **transform_params: Any) -> Iterable['str']:
        """
        :param X: Source of texts
        :param y: unused
        :param transform_params: unused
        :return: texts transformed into their language short strin e.g. ['hallo'] -> ['de']g
        """
        return [language_detect(x) for x in X]

    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None, **fit_params: Any) -> 'LanguageTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[None, None]:
        """unused"""
        return {}

    def set_params(self, **_: Any) -> None:
        """unused"""
        pass


if __name__ == '__main__':
    print(_create_enum_type())
    print(language_detect('hello world'))
