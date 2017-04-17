#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""package used for the ai components of the fanlens project"""

from setuptools import setup, find_packages

setup(
    name="fl-brain",
    version="3.0.0",
    long_description=__doc__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fl-base',
        'pycld2',
        'nltk',
        'retinasdk',
        'numpy',
        'scipy',
        'scikit-learn',
        'nltk',
        'sqlalchemy',
        'google-cloud-translate',
        'python-dateutil'
    ],
    package_data={'fl-brain.nltk_data': ['*']}
)
