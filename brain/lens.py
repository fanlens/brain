#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""a module for tagging texts based on semi supervised methods"""

import logging
import typing
import numpy
import pickle
import uuid
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

from sqlalchemy import text

from brain.feature.dense import DenseTransformer
from brain.feature.field_extract import FieldExtractTransformer
from brain.feature.fingerprint import DenseFingerprintTransformer
from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from brain.feature.emoji import EmojiTransformer
from brain.feature.punctuation import PunctuationTransformer
from brain.feature.timeofday import TimeOfDayTransformer

from db import DB
from db.models.activities import TagSet, Source
from db.models.brain import Model, ModelFile

_random_sample = text('''
  WITH tagged AS (
    SELECT dataT.id, taggingT.tag_id, langT.language, row_number() OVER (PARTITION BY taggingT.tag_id ORDER BY random()) AS rnr
    FROM activity.data AS dataT
         INNER JOIN activity.tagging AS taggingT ON dataT.id = taggingT.data_id
         INNER JOIN activity.tag_tagset AS tag_tagsetT ON taggingT.tag_id = tag_tagsetT.tag_id AND tag_tagsetT.id = :tagset_id
         INNER JOIN activity.source AS srcT ON dataT.source_id = srcT.id AND srcT.id IN :sources
         INNER JOIN activity.language AS langT ON dataT.id = langT.data_id AND langT.language IN :langs
  ),
  random_ids AS (
    (
      SELECT dataT.id, NULL AS tag_id, langT.language
      FROM activity.data AS dataT TABLESAMPLE BERNOULLI (30)
           INNER JOIN activity.source AS srcT ON dataT.source_id = srcT.id AND srcT.id IN :sources
           INNER JOIN activity.language AS langT ON dataT.id = langT.data_id AND langT.language IN :langs
      WHERE dataT.id NOT IN (SELECT id FROM tagged)
      LIMIT :num_samples
    ) UNION ALL (
      SELECT id, tag_id, language FROM tagged WHERE rnr <= :num_truths
    )
  )
  SELECT dataT.id, random_ids.tag_id, random_ids.language, textT.text, timeT.time, fpT.fingerprint
  FROM activity.data AS dataT
       INNER JOIN random_ids ON dataT.id = random_ids.id
       INNER JOIN activity.source AS srcT ON dataT.source_id = srcT.id
       INNER JOIN activity.text AS textT ON dataT.id = textT.data_id
       INNER JOIN activity.fingerprint AS fpT ON dataT.id = fpT.data_id
       INNER JOIN activity.time AS timeT ON dataT.id = timeT.data_id
''')


class Lens(object):
    @classmethod
    def load_from_id(cls, model_id: uuid.UUID):
        with DB().ctx() as session:
            try:
                model = session.query(Model).get(model_id)
                return model and cls.load_from_model(model)
            finally:
                session.expunge(model.file)

    @classmethod
    def load_from_model(cls, model: Model):
        bag = pickle.loads(model.file)
        return cls(model, bag)

    def __init__(self, model: Model, estimator_bag: typing.List[Pipeline]):
        self._estimator_bag = estimator_bag
        self._model = model

    @property
    def model(self):
        return self._model

    @property
    def estimator_bag(self):
        return self._estimator_bag

    def predict_proba(self, xs: typing.List):
        votes = defaultdict(lambda: defaultdict(lambda: []))
        for ys, proba in [estimator.predict_proba(xs) for estimator in self.estimator_bag]:
            for idx, y in enumerate(ys):
                votes[idx][y] += proba
        for idx, vote in votes.items():
            max_num_votes = -1
            max_score = 0
            max_tag = None
            for tag, scores in vote.items():
                norm = len(scores)
                if norm > max_num_votes:
                    max_num_votes = norm
                    max_score = sum(scores) / norm
                    max_tag = tag
            yield max_tag, max_score

    def predict(self, xs: typing.List):
        tag, _ = zip(*self.predict_proba(xs))
        yield from tag


class LensTrainer(object):
    def __init__(self, tagset: TagSet, sources: typing.Iterable[Source] = list(), progress=None):
        assert tagset and sources
        assert all(source.user_id == tagset.user_id for source in sources)
        self._tagset = tagset
        self._sources = sources
        self._progress = progress
        self._DB = DB()

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
        with self._DB.ctx() as session:
            query = session.execute(_random_sample,
                                    dict(tagset_id=self._tagset.id,
                                         num_samples=30,
                                         num_truths=2,
                                         langs=tuple(['en']),
                                         sources=tuple(source.id for source in self._sources)))
            for id, tag_id, language, message, created_time, fingerprint in query:
                yield tag_id or -1, (message, fingerprint, created_time)

    def _find_params(self) -> dict:
        return {
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

    def _train_stub(self, params: dict):
        ys, xs = zip(*self._fetch_samples())
        ys = numpy.array(ys)
        pipeline = self.taggger_pipeline()
        logging.debug('training')
        pipeline.set_params(**params)
        pipeline.fit(xs, ys)
        return pipeline

    def train(self, n_estimators=100, params: dict = {}):
        params = params or self._find_params()
        bag = [self._train_stub(params) for _ in range(0, n_estimators)]
        # todo real score
        model = Model(tagset_id=self._tagset.id, user_id=self._tagset.user_id, params=params, score=1.0)
        for source in self._sources:
            model.sources.append(source)
        return Lens(model, bag)

    def persist(self, lens: Lens):
        with self._DB.ctx() as session:
            lens.model.file = ModelFile(model_id=lens.model.id, file=pickle.dumps(lens.estimator_bag))
            session.add(lens.model)
            session.commit()
            return lens.model.id


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    with DB().ctx() as session:
        tagset = session.query(TagSet).get(1)
        sources = session.query(Source).filter(Source.id.in_((1, 2))).all()
        factory = LensTrainer(tagset, sources)
        lens = factory.train(n_estimators=2)
        factory.persist(lens)
