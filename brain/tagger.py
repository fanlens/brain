#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""a module for tagging texts based on semi supervised methods"""

import os
import pickle
import uuid

import numpy
import scipy.stats
import retinasdk
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.semi_supervised import LabelSpreading
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from brain.feature.dense import DenseTransformer
from brain.feature.field_extract import FieldExtractTransformer
from config.env import Environment
from db import DB, get_count
from db.models.facebook import FacebookCommentEntry

PATHS = Environment("PATHS")


def _get_model_file_path(name):
    return os.path.join(PATHS['model_dir'], '%s.pickle' % name)


def _get_model_file(name, mode='rb'):
    pickle_file_path = _get_model_file_path(name)
    return open(pickle_file_path, mode)


def _to_1d(sample):
    sample_1d = numpy.zeros(128 * 128, dtype='u1')
    sample_1d[sample] = 1
    return sample_1d


def to_sample(tokens, fingerprint: list):
    return ' '.join(tokens) if isinstance(tokens, list) else tokens, _to_1d(fingerprint)


class Tagger(object):
    """a auto tagger based on semi supervised data"""

    def __init__(self, name, clf=(None, None)):
        self._name = name
        if clf[1] is None:
            with _get_model_file(name) as pickle_file:
                self._tags, self._clf = pickle.load(pickle_file)  # type: Pipeline
        else:
            self._tags, self._clf = clf

    def predict(self, sample):
        return list(self.predict_all([sample]))[0]

    def predict_all(self, samples):
        xs = numpy.array(samples, dtype=[('tokens', object), ('fingerprint', 'u1', 128 * 128)])
        predicted = self._clf.predict_proba(xs)
        return [list(zip(p, self._tags)) for p in predicted]

    @property
    def name(self):
        return self._name


def fetch_samples(pages, tags):
    print("DATA")
    print("=======================================")
    print("Fetching...")
    with DB().ctx() as session:
        query = session.query(FacebookCommentEntry).filter(
            (FacebookCommentEntry.meta['tokens'] != None) & (FacebookCommentEntry.meta['fingerprint'] != None) & (
                FacebookCommentEntry.meta['lang'].astext == 'en'))
        if pages is not None:
            query = query.filter(FacebookCommentEntry.meta['page'].astext.in_(pages))
        count = get_count(query)
        xs = numpy.recarray(count, dtype=[('tokens', object), ('fingerprint', 'u1', 128 * 128)])
        ys = numpy.empty(count, dtype='i1')
        i = 0
        for entry in query:
            sample = to_sample(entry.data['tokens'], entry.data['fingerprint'])
            relevant_tags = set(entry.data.get('tags', [])).intersection(set(tags))
            if relevant_tags:
                tag = list(relevant_tags)[0]
                tag_idx = tags.index(tag)
            else:
                tag_idx = -1
            xs[i] = sample
            ys[i] = tag_idx
            i += 1
    print("Total samples:\t", xs.shape)
    print("=======================================")
    return xs, ys


def train_test_split(xs, ys, limit_train=-1):
    print("Extracting Seeds")
    print("=======================================")
    seed_xs = xs[ys >= 0]
    seed_ys = ys[ys >= 0]
    length = len(seed_xs)
    limit = int(0.75 * length)
    keep = numpy.random.permutation(length)[:limit]
    xs_train = numpy.concatenate((xs[ys < 0], seed_xs[keep]))
    ys_train = numpy.concatenate((ys[ys < 0], seed_ys[keep]))
    if limit_train > 0:
        xs_train = xs_train[-1:-limit_train:-1]
        ys_train = ys_train[-1:-limit_train:-1]
    xs_test = seed_xs[~keep]
    ys_test = seed_ys[~keep]
    print("Train samples:\t", xs_train.shape)
    print("Test samples:\t", xs_test.shape)
    print("=======================================")
    return xs_train, ys_train, xs_test, ys_test


def get_pipeline():
    return Pipeline([
        ('text_features', FeatureUnion([
            ('tokens', Pipeline([
                ('tokens', FieldExtractTransformer(key='tokens')),
                ('vect', CountVectorizer(analyzer='word',
                                         # token_pattern=r"[^\s][^\s]+",
                                         stop_words='english')),
                ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
                ('features', FeatureUnion([
                    ('select', Pipeline([
                        ('chi2', SelectKBest(chi2)),
                        ('to_dense', DenseTransformer())])),
                    ('reduce', Pipeline([
                        ('to_dense', DenseTransformer()),
                        ('pca', PCA())
                    ]))
                ])),
            ])),
            ('fingerprint', Pipeline([
                ('fingerprint', FieldExtractTransformer(key='fingerprint')),
                ('pca', PCA())
            ]))
        ])),
        ('clf', LabelSpreading(kernel='rbf'))
    ])


def search_params(tags: list, name=uuid.uuid1(), pages=None, overwrite=False) -> Tagger:
    if not overwrite and os.path.exists(_get_model_file_path(name)):
        return Tagger(name)

    xs, ys = fetch_samples(pages, tags)
    xs_train, ys_train, xs_test, ys_test = train_test_split(xs, ys, limit_train=3000)
    del xs, ys

    parameters_space = {
        'text_features__tokens__vect__ngram_range': [(1, 1), (1, 2)],
        'text_features__tokens__vect__max_df': scipy.stats.uniform(loc=0.01, scale=0.99),
        'text_features__tokens__features__select__chi2__k': scipy.stats.randint(1, 64),
        'text_features__tokens__features__reduce__pca__n_components': scipy.stats.randint(1, 5),
        'text_features__fingerprint__pca__n_components': scipy.stats.randint(1, 5),
        'clf__max_iter': scipy.stats.randint(30, 160),
        'clf__gamma': scipy.stats.expon(scale=20),
    }
    gs_clf = RandomizedSearchCV(get_pipeline(), parameters_space, n_jobs=-1, verbose=3, n_iter=20)
    gs_clf.fit(xs_train, ys_train)
    predicted = gs_clf.predict(xs_test)
    print(metrics.classification_report(ys_test, predicted))
    print(gs_clf.best_params_)

    with _get_model_file(name, 'wb') as pickle_file:
        pickle.dump((tags, gs_clf), pickle_file)
    return Tagger(name)


def train(tags: list, name=uuid.uuid1(), pages=None, overwrite=False) -> Tagger:
    if not overwrite and os.path.exists(_get_model_file_path(name)):
        return Tagger(name)

    xs, ys = fetch_samples(pages, tags)

    best_parameters = {
        'text_features__tokens__vect__max_df': 0.1370232981023436,
        'text_features__fingerprint__pca__n_components': 4,
        'text_features__tokens__vect__ngram_range': (1, 1), 'clf__max_iter': 53,
        'clf__gamma': 6.729225136765749,
        'text_features__tokens__features__select__chi2__k': 52,
        'text_features__tokens__features__reduce__pca__n_components': 2
    }
    # 97%, 18.9minutes
    # best_parameters = {'text_features__fingerprint__pca__n_components': 4, 'text_features__tokens__vect__ngram_range': (1, 1), 'text_features__tokens__features__reduce__pca__n_components': 4, 'clf__gamma': 7.08723054987839, 'clf__max_iter': 78, 'text_features__tokens__vect__max_df': 0.5449386304928278, 'text_features__tokens__features__select__chi2__k': 2}
    pipeline = get_pipeline()
    pipeline.set_params(**best_parameters)
    pipeline.fit(xs, ys)

    # with _get_model_file(name, 'wb') as pickle_file:
    #    pickle.dump((tags, pipeline), pickle_file)
    # return Tagger(name)


if __name__ == '__main__':
    searched = search_params(['spam', 'ham'], name='debug_tagger_search', pages=['ladygaga'], overwrite=True)
    # tagger = train(['spam', 'ham'], name='debug_tagger', pages=['ladygaga'], overwrite=True)
