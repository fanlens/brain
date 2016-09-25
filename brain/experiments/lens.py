#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""a module for tagging texts based on semi supervised methods"""

import logging
import os
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
from sklearn.ensemble import AdaBoostClassifier

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
from db.models.users import User  # important so the foreign keys know about it
from db.models.tags import TagSet, UserToTagSet, UserToTagToComment
from db.models.brain import Model

PATHS = Environment("PATHS")


class LensFactory(object):
    def __init__(self, progress=None):
        self._tagset = 1
        self._name = uuid.uuid1()
        self._params = {}
        self._trained = None
        self._tags = {}
        self._user_id = 7
        self._xs = []
        self._ys = numpy.array([])
        self._seed_idxs = defaultdict(list)
        self._sources = tuple()
        self._score = 0.0
        self._progress = progress

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

    def _fetch_samples(self):
        logging.debug("Fetching...")
        with DB().ctx() as session:
            query = session.execute('''
            WITH tagged AS (SELECT
                  utc.tag,
                  comment_id AS id
                FROM user_tag_comment AS utc, data.facebook_comments AS comments, meta.tag_tagset AS tagset
                WHERE utc.user_id = 7 AND tagset.tagset_id = 1 AND utc.tag = tagset.tag AND comment_id = comments.id)
            SELECT tag, sample.id, data->>'message', meta->'fingerprint', data->'created_time' from ((
              SELECT
                tag,
                id
              FROM (
                     SELECT
                       id,
                       tag,
                       row_number()
                       OVER (PARTITION BY tag
                         ORDER BY random()) AS rnr
                     FROM tagged
                   ) AS rnrd
              WHERE rnr <= 2
            )
            UNION ALL (
              SELECT
                null,
                id
              FROM data.facebook_comments
                TABLESAMPLE SYSTEM (10)
              WHERE id NOT IN (SELECT id
                               FROM tagged) and
                    meta->'fingerprint' is not null
              LIMIT 28
            )) as sample, data.facebook_comments as comm where sample.id = comm.id
            ''')
            self._tags = {None: -1}
            for tag, id, message, fingerprint, created_time in query:
                if tag not in self._tags:
                    self._tags[tag] = len(self._tags) - 1
                yield self._tags[tag], (message, fingerprint, created_time)
            print(self._tags)

    def train_stub(self):
        ys, xs = zip(*self._fetch_samples())
        ys = numpy.array(ys)
        pipeline = self.taggger_pipeline()
        params = {
            # "clf__gamma": 12.095692803269314,
            "clf__gamma": 2.5,
            "clf__max_iter": 107,
            "features__fingerprint__randomizedpca__n_components": 3,
            "features__timeofday__timeofdaytransformer__resolution": 1,
            "features__message__features__emoji__selectkbest__k": 49,
            "features__message__features__tokens__selectkbest__k": 14,
            "features__message__features__punctuation__selectkbest__k": 16,
            "features__message__features__tokens__lemmatokentransformer__short_url": False,
            "features__message__features__punctuation__punctuationtransformer__strict": False
        }
        logging.debug('training')
        pipeline.set_params(**params)
        pipeline.fit(xs, ys)
        return pipeline

    def train(self, n_estimators=100):
        return [self.train_stub() for _ in range(0, n_estimators)]

    def predict(self, bag, xs, resolve_tag=True):
        votes = defaultdict(lambda: defaultdict(lambda: 0))
        for ys in [estimator.predict(xs) for estimator in bag]:
            for idx, y in enumerate(ys):
                votes[idx][y] += 1
        for idx, vote in votes.items():
            cur_max = 0
            cur_tag = None
            total_score = 0
            for tag, score in vote.items():
                total_score += score
                if score > cur_max:
                    cur_max = score
                    cur_tag = tag
            #yield cur_tag, cur_max/total_score
            yield cur_tag


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    lens = LensFactory()

    bag = lens.train(n_estimators=10)
    with DB().ctx() as session:
        query = session.execute('''
        SELECT
          utc.tag,
          data->>'message',
          meta->'fingerprint',
          data->'created_time'
        FROM user_tag_comment AS utc, data.facebook_comments AS comments, meta.tag_tagset AS tagset
        WHERE utc.user_id = 7 AND tagset.tagset_id = 1 AND utc.tag = tagset.tag AND comment_id = comments.id''')
        ys_test, xs_test = zip(*[(tag, (message, fingerprint, tod)) for tag, message, fingerprint, tod in query])
        predicted = lens.predict(bag, xs_test)

        for line in zip(ys_test, predicted, [xs[0] for xs in xs_test]):
            print(str(line))
        err_count = 0
        for yt, yp in zip(ys_test, predicted):
            if yt != ('ham' if yp == 0 else 'spam'):
                err_count += 1
        print('encountered', str(err_count), 'errors in', str(len(ys_test)), 'samples')
        print('that is', str(err_count/len(ys_test) * 100), 'percent')
