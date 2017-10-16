"""module for a emoji based tokenizer"""

import re
from typing import Iterable, Optional, Any, Dict

import numpy as np
from sklearn.base import TransformerMixin

from . import output, TOutputType, TOutput


class EmojiTransformer(TransformerMixin):
    # match signatures, pylint: disable=unused-argument
    """extract all emojis"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    def __init__(self, output_type: TOutputType = dict) -> None:
        self._output_type = output_type

    def transform(self, X: Iterable[str], y: Optional[np.ndarray] = None, **transform_params: Any) -> Iterable[TOutput]:
        """
        :param X: list of text samples
        :param y: unused
        :param transform_params: unused
        :return: extracted emojis according to output type
        """
        return [self(x) for x in X]

    def fit(self, X: Iterable[str], y: Optional[np.ndarray] = None, **fit_params: Any) -> 'EmojiTransformer':
        """unused"""
        return self

    def get_params(self, deep: bool = False) -> Dict[str, TOutputType]:
        """Get parameters for this Transformer."""
        return dict(output_type=self._output_type)

    def set_params(self, **params: TOutputType) -> None:
        """Set parameters for this Transformer."""
        self._output_type = params.get('output_type', self._output_type)

    def __call__(self, doc: str) -> TOutput:
        """
        :param doc: the text sample to extract emojis for
        :return:
        """
        emojis = self.emoji_pattern.findall(doc)
        return output(self._output_type, emojis)
