#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module for a lemma based tokenizer"""

import re
from sklearn.base import TransformerMixin


class EmojiTransformer(TransformerMixin):
    """extract all emojis"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    def __init__(self, output_type='string'):
        self._output_type = output_type

    def transform(self, X, y=None, **transform_params):
        return [self(x) for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return dict(output_type=self._output_type)

    def set_params(self, **params):
        self._output_type = params.get('output_type', self._output_type)

    def __call__(self, doc):
        emojis = self.emoji_pattern.findall(doc)
        if self._output_type == 'string':
            return ''.join(emojis)
        else:
            return emojis
