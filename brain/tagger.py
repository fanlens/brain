#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""a module for tagging texts based on semi supervised methods"""

import logging
import os
import pickle
import uuid

import numpy
import scipy.stats
from collections import defaultdict

from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.semi_supervised import LabelSpreading
from sklearn.externals import joblib

from brain.feature.dense import DenseTransformer
from brain.feature.field_extract import FieldExtractTransformer
from brain.feature.fingerprint import DenseFingerprintTransformer
from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from brain.feature.emoji import EmojiTransformer
from brain.feature.punctuation import PunctuationTransformer
from brain.feature.timeofday import TimeOfDayTransformer
from config.env import Environment
from db import DB, insert_or_update
from db.models.facebook import FacebookCommentEntry
from db.models.tags import TagSet, UserToTagSet, UserToTagToComment
from db.models.brain import Model

PATHS = Environment("PATHS")


def _get_model_file_path(name):
    return os.path.join(PATHS['model_dir'], str(name))


def _get_model_file(name, mode='rb'):
    pickle_file_path = _get_model_file_path(str(name))
    return open(pickle_file_path, mode)


class Tagger(object):
    """a auto tagger based on semi supervised data"""

    def __init__(self, name, tags: set, classifier: Pipeline):
        self._name = name
        self._tags = tags
        self._classifier = classifier

    def predict(self, sample):
        return list(self.predict_all([sample]))[0]

    def predict_all(self, samples):
        predicted = self._classifier.predict_proba(samples)
        return [list(zip(p, self.tags)) for p in predicted]

    @property
    def tags(self):
        return self._tags

    @property
    def name(self):
        return str(self._name)


class TaggerFactory(object):
    @staticmethod
    def taggger_pipeline():
        return Pipeline([
            ('features', FeatureUnion([
                ('message', Pipeline([
                    ('extract', FieldExtractTransformer(key=0)),
                    ('features', FeatureUnion([
                        ('tokens', make_pipeline(
                            LemmaTokenTransformer(short_url=True, output_type=list),
                            FeatureHasher(input_type='string', non_negative=True),
                            TfidfTransformer(use_idf=True, smooth_idf=True),
                            SelectKBest(chi2),
                            DenseTransformer(),
                        )),
                        ('emoji', make_pipeline(  # todo proper evaluation of feature, seemed to make quite a difference
                            EmojiTransformer(output_type=list),
                            FeatureHasher(input_type='string', non_negative=True),
                            # why tfidf?
                            TfidfTransformer(use_idf=True, smooth_idf=True),
                            SelectKBest(chi2),
                            DenseTransformer(),
                        )),
                        ('punctuation', make_pipeline(  # todo proper evaluation of feature
                            PunctuationTransformer(strict=False, output_type=list),
                            FeatureHasher(input_type='string', non_negative=True),
                            # why tfidf?
                            TfidfTransformer(use_idf=True, smooth_idf=True),
                            SelectKBest(chi2),
                            DenseTransformer(),
                        )),
                    ])),
                ])),
                ('fingerprint', make_pipeline(
                    FieldExtractTransformer(key=1),
                    DenseFingerprintTransformer(),
                    RandomizedPCA())),
                ('timeofday', make_pipeline(  # todo proper evaluation of feature
                    FieldExtractTransformer(key=2),
                    TimeOfDayTransformer(resolution=3, dense=True))),
            ])),
            ('clf', LabelSpreading(kernel='rbf'))
        ])

    # todo ngrams?
    parameters_space = {
        'features__message__features__tokens__lemmatokentransformer__short_url': [True, False],
        'features__message__features__tokens__selectkbest__k': scipy.stats.randint(1, 64),
        'features__message__features__emoji__selectkbest__k': scipy.stats.randint(1, 64),
        'features__message__features__punctuation__punctuationtransformer__strict': [True, False],
        'features__message__features__punctuation__selectkbest__k': scipy.stats.randint(1, 64),
        'features__fingerprint__randomizedpca__n_components': scipy.stats.randint(1, 5),
        'features__timeofday__timeofdaytransformer__resolution': scipy.stats.randint(1, 6),
        'clf__max_iter': scipy.stats.randint(30, 160),
        'clf__gamma': scipy.stats.expon(scale=20),

    }

    def __init__(self):
        self._pages = []
        self._tagset = None
        self._name = uuid.uuid1()
        self._params = {}
        self._trained = None
        self._tags = []
        self._user_id = None
        self._xs = []
        self._ys = numpy.array([])
        self._seed_idxs = defaultdict(list)
        self._sources = tuple()
        self._score = 0.0

    def user_id(self, user_id):
        self._user_id = user_id
        return self

    def tagset(self, tagset_id):
        if tagset_id is None:
            return self
        with DB().ctx() as session:
            self._tagset = session.query(TagSet).join(UserToTagSet).filter(
                (UserToTagSet.user_id == self._user_id) &
                (TagSet.id == tagset_id)
            )
            self._tagset, = list(self._tagset)
            self._tags = {tag.tag for tag in self._tagset.tags}
        return self

    def pages(self, pages):
        self._pages = pages
        return self

    def name(self, name):
        """:param: the file name for this model, ignore if None"""
        if name is not None:
            self._name = name
        if os.path.isfile(_get_model_file_path(self._name)):
            logging.warning("model with name %s already exists!" % self._name)
        return self

    def sources(self, sources=tuple()):
        if isinstance(sources, str):
            sources = (sources,)
        self._sources = tuple(set(sources))  # ensure set condition
        return self

    def _fetch_samples(self):
        logging.debug("=======================================")
        logging.debug("DATA")
        logging.debug("Fetching...")
        with DB().ctx() as session:
            tagged_query = session.query(UserToTagToComment).filter(UserToTagToComment.tag.in_(self._tags))
            if self._user_id is not None:
                tagged_query = tagged_query.filter(UserToTagToComment.user_id == self._user_id)
            tagged_query = tagged_query.subquery()
            query = session.query(FacebookCommentEntry, tagged_query.c.tag).outerjoin(tagged_query).filter(
                (FacebookCommentEntry.meta['fingerprint'] != None) &
                (FacebookCommentEntry.meta['lang'].astext == 'en')
            )
            if self._sources:
                query = query.filter(
                    (FacebookCommentEntry.meta['page'].astext.in_(self._sources))
                )
            if self._pages:
                query = query.filter(FacebookCommentEntry.meta['page'].astext.in_(self._pages))

            self._xs = []
            self._ys = []
            self._tags = []
            tags_idx = {None: -1}
            for idx, (entry, tag) in enumerate(query):
                if tag not in tags_idx:
                    tags_idx[tag] = len(self._tags)
                    self._tags.append(tag)
                x = (entry.data['message'], entry.meta['fingerprint'], entry.data['created_time'])
                y = tags_idx[tag]
                self._xs.append(x)
                self._ys.append(y)
                if y >= 0:
                    self._seed_idxs[y].append(idx)
            self._ys = numpy.array(self._ys, dtype='i1')
            logging.debug("Total samples:\t%d" % len(self._xs))
            logging.debug("=======================================")

    def _seeded_train_test_split(self, limit_train=-1, n_folds=3):
        for i in range(0, n_folds):
            logging.debug("=======================================")
            logging.debug("Creating Fold")
            num_samples = len(self._ys)
            train_samples = numpy.random.permutation(num_samples)[:limit_train if limit_train >= 0 else num_samples]
            test_samples = numpy.array([], dtype=train_samples.dtype)
            for y, idxs in self._seed_idxs.items():
                num_seeds = len(idxs)
                ratio = max(1, int(num_seeds * 0.75))
                random_seeds = numpy.random.permutation(len(idxs))
                idxs = numpy.array(idxs, train_samples.dtype)
                train = idxs[random_seeds[:ratio]]
                test = idxs[random_seeds[ratio:]]
                train_samples = numpy.append(train_samples, train)
                test_samples = numpy.append(test_samples, test)
            train_samples = numpy.unique(train_samples)
            logging.debug("Train samples (%s):\t%d" % (train_samples.dtype, len(train_samples)))
            logging.debug("Test samples (%s):\t%d" % (test_samples.dtype, len(test_samples)))
            logging.debug("=======================================")
            yield train_samples, test_samples

    def params(self, params=None, use_existing=None):
        if params:
            self._params = params
        if use_existing:
            with DB().ctx() as session:
                model = list(session.query(Model)
                             .filter_by(tagset_id=self._tagset.id, user_id=self._user_id)
                             .order_by(Model.trained_ts.desc())
                             .limit(1))
                if model:
                    self._params = model[0].params
                    self._score = model[0].score
        return self

    def _simple_train(self):
        logging.debug("=======================================")
        logging.debug("Training with provided parameters %s" % self._params)

        pipeline = self.taggger_pipeline()
        pipeline.set_params(**self._params)
        logging.debug("fitting model")
        pipeline.fit(self._xs, self._ys)
        self._trained = pipeline
        score_xs = []
        score_ys = []
        for y, idxs in self._seed_idxs.items():
            for idx in idxs:
                score_xs.append(self._xs[idx])
                score_ys.append(self._ys[idx])
        logging.debug("calculating score")
        self._score = self._trained.score(score_xs, numpy.array(score_ys))
        logging.debug("Score for provided params: " + str(self._score))
        logging.debug("=======================================")

    def _search_train(self):
        folds = list(self._seeded_train_test_split(limit_train=-1))
        logging.debug("=======================================")
        logging.debug("Searching optimal parameters")
        gs_clf = RandomizedSearchCV(self.taggger_pipeline(), self.parameters_space, n_jobs=-1, verbose=3, n_iter=20,
                                    cv=folds)
        gs_clf.fit(self._xs, self._ys)
        logging.debug("Score %s can be reached using %s" % (gs_clf.best_score_, gs_clf.best_params_))
        logging.debug("=======================================")
        self._score = gs_clf.best_score_
        self._params = gs_clf.best_params_
        self._trained = gs_clf.best_estimator_

    def train(self):
        if not self._xs:
            self._fetch_samples()
        if self._params:
            self._simple_train()
        else:
            self._search_train()
        return self

    def persist(self, overwrite=False):
        assert self._trained is not None
        with DB().ctx() as session:
            joblib.dump((self._tags, self._trained, self._params), _get_model_file_path(self._name))
            model = Model(id=self._name,
                          tagset_id=self._tagset.id,
                          user_id=self._user_id,
                          params=self._params,
                          score=self._score)
            if overwrite:
                insert_or_update(session, model, 'id')
            else:
                session.add(model)
            session.commit()
        return self

    @property
    def tagger(self) -> Tagger:
        if self._trained is None:
            filename = _get_model_file_path(self._name)
            if os.path.isfile(filename):
                self._tags, self._trained, self._params = joblib.load(filename, mmap_mode='r')
            else:
                raise FileNotFoundError('model file not found')
        return Tagger(self._name, self._tags, self._trained)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='provide id of comment_id')
    parser.add_argument('-u', '--user', type=int, help='the user id', required=True)
    parser.add_argument('-t', '--tagset', type=int, help='the tagset id', required=True)
    parser.add_argument('-n', '--name', type=str, help='the name of the model (optional)', required=False, default=None)
    parser.add_argument('-s', '--sources', type=str, metavar='S', nargs='+', help='the sources id', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d (threadName)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    factory = (TaggerFactory()
               .user_id(args.user)
               .tagset(args.tagset)
               .sources(tuple(args.sources))
               .name(args.name)
               .params(use_existing=True)
               .train()
               .persist(overwrite=True))
    logging.info('created tagger with name ' + factory.tagger.name)
