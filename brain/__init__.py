#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""core machine learning module"""

import os
import nltk

from config.env import Environment

_paths = Environment('PATHS')
nltk.data.path.append(os.path.join(_paths['data_dir'], 'nltk_data'))
