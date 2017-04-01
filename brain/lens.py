#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""a module for tagging texts based on semi supervised methods"""
import logging
import operator
import pickle
import uuid
from collections import defaultdict

import numpy
import scipy.stats
import typing
from brain.feature.capitalization import CapitalizationTransformer
from brain.feature.emoji import EmojiTransformer
from brain.feature.field_extract import FieldExtractTransformer
from brain.feature.fingerprint import SparseFingerprintTransformer
from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from brain.feature.punctuation import PunctuationTransformer
from brain.feature.timeofday import TimeOfDayTransformer
from db import DB
from db.models.activities import TagSet, Source
from db.models.brain import Model, ModelFile
from sklearn.decomposition import TruncatedSVD
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.semi_supervised import LabelSpreading
from sqlalchemy import text

_random_sample = text('''
  WITH tagged AS (
    SELECT dataT.id, taggingT.tag_id, langT.language, row_number() OVER (PARTITION BY taggingT.tag_id ORDER BY random()) AS rnr
    FROM activity.data AS dataT
         INNER JOIN activity.tagging AS taggingT ON dataT.id = taggingT.data_id
         INNER JOIN activity.tag_tagset AS tag_tagsetT ON taggingT.tag_id = tag_tagsetT.tag_id AND tag_tagsetT.tagset_id = :tagset_id
         INNER JOIN activity.source AS srcT ON dataT.source_id = srcT.id AND srcT.id IN :sources
         INNER JOIN activity.language AS langT ON dataT.id = langT.data_id AND langT.language IN :langs
  ),
  random_ids AS (
    (
      SELECT dataT.id, NULL AS tag_id, langT.language
      FROM activity.data AS dataT TABLESAMPLE BERNOULLI (30)
           INNER JOIN activity.source AS srcT ON dataT.source_id = srcT.id AND srcT.id IN :sources
           INNER JOIN activity.language AS langT ON dataT.id = langT.data_id AND langT.language IN :langs
           INNER JOIN activity.text AS textT ON dataT.id = textT.data_id AND char_length(textT.text) > 64
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


# http://stackoverflow.com/questions/1257413/iterate-over-pairs-in-a-list-circular-fashion-in-python
def ntuples(lst, n):
    return zip(*[lst[i:] + lst[:i] for i in range(n)])


class Lens(object):
    @classmethod
    def load_from_id(cls, model_id: uuid.UUID):
        with DB().ctx() as session:
            model = session.query(Model).get(str(model_id))
            try:
                return model and cls.load_from_model(model)
            finally:
                session.expunge(model.file)

    @classmethod
    def load_from_model(cls, model: Model):
        bag = pickle.loads(model.file.file)
        return cls(model, bag)

    def __init__(self, model: Model, estimator_bag: typing.List[Pipeline]):
        self._estimator_bag = estimator_bag
        self._model = model

    @property
    def model(self) -> Model:
        return self._model

    @property
    def estimator_bag(self):
        return self._estimator_bag

    def predict_proba(self, xs: typing.List):
        n_bags = len(self.estimator_bag)
        for idx, votes in enumerate(zip(*[estimator.predict(xs) for estimator in self.estimator_bag])):
            counter = defaultdict(int)
            for vote in votes:
                counter[vote] += 1
            max_score = 0
            max_tag = None
            for tag, score in counter.items():
                if score > max_score:
                    max_score = score
                    max_tag = tag
            yield [int(max_tag), max_score / n_bags]

    def predict(self, xs: typing.List):
        for tags_proba in self.predict_proba(xs):
            tag, _ = max(tags_proba, operator.itemgetter(1))
            yield tag


import random


def make_weight_dist(*weight_names, num_samples=50):
    for s in range(0, num_samples):
        yield dict((name, random.random()) for name in weight_names)


class LensTrainer(object):
    parameters_space = {
        'features__message__features__capitalization__capitalizationtransformer__fraction': [True, False],
        'features__message__features__tokens__lemmatokentransformer__short_url': [True, False],
        'features__message__features__tokens__truncatedsvd__n_components': scipy.stats.randint(1, 200),
        'features__message__features__tokens__truncatedsvd__n_iter': scipy.stats.randint(3, 12),
        'features__message__features__emoji__truncatedsvd__n_components': scipy.stats.randint(1, 20),
        'features__message__features__emoji__truncatedsvd__n_iter': scipy.stats.randint(3, 12),
        'features__message__features__punctuation__punctuationtransformer__strict': [True, False],
        'features__message__features__punctuation__truncatedsvd__n_components': scipy.stats.randint(1, 20),
        'features__message__features__punctuation__truncatedsvd__n_iter': scipy.stats.randint(3, 12),
        'features__fingerprint__truncatedsvd__n_components': scipy.stats.randint(1, 10),
        'features__fingerprint__truncatedsvd__n_iter': scipy.stats.randint(3, 12),
        'features__timeofday__timeofdaytransformer__resolution': scipy.stats.randint(1, 6),
        'features__transformer_weights': list(make_weight_dist('message', 'fingerprint', 'timeofday')),
        # 'features__transformer_weights': [dict(message=0.7, fingerprint=1.0, timeofday=0.2)],
        'clf__max_iter': scipy.stats.randint(30, 160),
        'clf__gamma': scipy.stats.expon(scale=20),
        'clf__alpha': scipy.stats.expon(scale=0.2),
        'clf__n_neighbors': scipy.stats.randint(2, 15),
        'clf__kernel': ['knn', 'rbf'],
    }

    def __init__(self, tagset: TagSet, sources: typing.Iterable[Source] = list(), progress=None):
        assert tagset and sources
        assert all(source.user.id == tagset.user.id for source in sources)
        self._tagset = tagset
        self._sources = sources
        self._progress = progress

    @staticmethod
    def taggger_pipeline():
        # todo: use weights for the features! nonlinear?!
        return Pipeline([
            ('features', FeatureUnion([
                ('message', Pipeline([
                    ('extract', FieldExtractTransformer(key=0)),
                    ('features', FeatureUnion([
                        ('tokens', make_pipeline(
                            LemmaTokenTransformer(output_type=dict),
                            FeatureHasher(non_negative=True, n_features=200),
                            TfidfTransformer(use_idf=True, smooth_idf=True),
                            TruncatedSVD(algorithm='randomized')
                        )),
                        ('emoji', make_pipeline(
                            EmojiTransformer(output_type=dict),
                            FeatureHasher(non_negative=True, n_features=20),
                            TfidfTransformer(use_idf=True, smooth_idf=True),
                            TruncatedSVD(algorithm='randomized'),
                        )),
                        ('punctuation', make_pipeline(
                            PunctuationTransformer(strict=False, output_type=dict),
                            FeatureHasher(non_negative=True, n_features=20),
                            TfidfTransformer(use_idf=True, smooth_idf=True),
                            TruncatedSVD(algorithm='randomized'),
                        )),
                        ('capitalization', make_pipeline(
                            CapitalizationTransformer(),
                        )),
                    ])),
                ])),
                ('fingerprint', make_pipeline(
                    FieldExtractTransformer(key=1),
                    SparseFingerprintTransformer(),
                    TruncatedSVD(algorithm='randomized'),
                    Normalizer(copy=False),
                )),
                ('timeofday', make_pipeline(
                    FieldExtractTransformer(key=2),
                    TimeOfDayTransformer(dense=True),
                    Normalizer(copy=False),
                )),
            ])),
            ('clf', LabelSpreading())
        ])

    def _fetch_samples(self, num_truths: int, num_samples: int):
        logging.debug("fetching sampleset with %d truths and %d samples..." % (num_truths, num_samples))
        with DB().ctx() as session:
            query = session.execute(_random_sample,
                                    dict(tagset_id=self._tagset.id,
                                         num_samples=num_samples,
                                         num_truths=num_truths,
                                         langs=tuple(['en']),
                                         sources=tuple(source.id for source in self._sources)))
            for id, tag_id, language, message, created_time, fingerprint in query:
                yield tag_id or -1, (message, fingerprint, created_time)
        logging.debug("... done fetching sampleset")

    def _find_params(self, num_folds=3) -> tuple:
        logging.debug("search best parameters using %d folds..." % num_folds)
        folds = [list(self._fetch_samples()) for _ in range(0, num_folds)]
        folds_idx = []
        cur_idx = 0
        for fold in folds:
            stop = cur_idx + len(fold)
            truths = []
            for idx, sample in enumerate(fold):
                if sample[0] != -1:
                    truths.append(cur_idx + idx)
            folds_idx.append((cur_idx, stop, truths))
            cur_idx = stop
        search = RandomizedSearchCV(
            self.taggger_pipeline(),
            self.parameters_space,
            n_jobs=1, verbose=10, n_iter=20,
            cv=[(numpy.array(range(start, stop)), numpy.array(truths)) for (start, stop, _), (_, _, truths) in
                ntuples(folds_idx, 2)])
        ys, xs = zip(*[sample for fold in folds for sample in fold])
        search.fit(xs, ys)
        logging.debug("... done searching parameters, achieved score of %f with params %s" % (
            search.best_score_, search.best_params_,))
        return search.best_params_, search.best_score_

    def _train_stub(self, params: dict, num_truths: int, num_samples: int, _stub_id=uuid.uuid1()):
        ys, xs = zip(*self._fetch_samples(num_truths=num_truths, num_samples=num_samples))
        ys = numpy.array(ys)
        logging.debug('training stub %s...' % _stub_id)
        pipeline = self.taggger_pipeline()
        pipeline.set_params(**params)
        pipeline.fit(xs, ys)
        logging.debug('... done training stub %s' % _stub_id)
        return pipeline

    def train(self, n_estimators=100, num_truths: int = 48, num_samples: int = 128, _params: dict = None,
              _score: float = 0.0):
        logging.debug('training estimator bag with %d estimators...' % n_estimators)
        if _params:
            params, score = _params, _score
        else:
            params, score = self._find_params()

        bag = Parallel(n_jobs=-1)(delayed(self._train_stub)(params, num_truths, num_samples, i) for i in range(0, n_estimators))
        model = Model(tagset_id=self._tagset.id, user_id=self._tagset.user.id, params=params, score=score)
        for source in self._sources:
            model.sources.append(source)
        logging.debug('... done training estimator bag')
        return Lens(model, bag)

    def persist(self, lens: Lens):
        logging.debug('persisting trained model...')
        with DB().ctx() as session:
            lens.model.file = ModelFile(model_id=lens.model.id,
                                        file=pickle.dumps(lens.estimator_bag, protocol=pickle.HIGHEST_PROTOCOL))
            session.add(lens.model)
            session.commit()
            logging.debug('... done persisting trained model, new id is: %s' % lens.model.id)
            return lens.model.id


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    with open('/Users/chris/Projects/fanlens/data/testset.pickle', 'rb') as testsetfile:
        ys_test, xs_test = zip(*pickle.load(testsetfile))
        num_test = len(ys_test)

    with DB().ctx() as session:
        tagset = session.query(TagSet).get(1)
        sources = session.query(Source).filter(Source.id.in_((2,))).all()
        factory = LensTrainer(tagset, sources)
        params = {'features__fingerprint__truncatedsvd__n_iter': 10,
                  'features__message__features__tokens__truncatedsvd__n_components': 106,
                  'features__timeofday__timeofdaytransformer__resolution': 5,
                  'features__message__features__punctuation__truncatedsvd__n_iter': 3, 'clf__max_iter': 111,
                  'clf__n_neighbors': 2, 'clf__gamma': 17.222983787280945,
                  'features__message__features__emoji__truncatedsvd__n_components': 16,
                  'features__fingerprint__truncatedsvd__n_components': 5, 'clf__alpha': 0.3307067042583709,
                  'features__message__features__tokens__truncatedsvd__n_iter': 4,
                  'features__message__features__emoji__truncatedsvd__n_iter': 10, 'clf__kernel': 'rbf',
                  'features__message__features__tokens__lemmatokentransformer__short_url': False,
                  'features__message__features__punctuation__punctuationtransformer__strict': False,
                  'features__message__features__punctuation__truncatedsvd__n_components': 17,
                  'features__message__features__capitalization__capitalizationtransformer__fraction': True}
        params = {'clf__max_iter': 151, 'clf__gamma': 15.207618172582782,
                  'features__message__features__punctuation__truncatedsvd__n_components': 3,
                  'features__fingerprint__truncatedsvd__n_components': 3,
                  'features__message__features__tokens__truncatedsvd__n_iter': 8,
                  'features__message__features__tokens__truncatedsvd__n_components': 106,
                  'features__message__features__emoji__truncatedsvd__n_components': 1, 'clf__kernel': 'rbf',
                  'features__fingerprint__truncatedsvd__n_iter': 5,
                  'features__transformer_weights': {'timeofday': 0.2030570157381013, 'message': 0.6336048802942279,
                                                    'fingerprint': 0.7186676931811034},
                  'features__message__features__punctuation__punctuationtransformer__strict': True,
                  'features__message__features__tokens__lemmatokentransformer__short_url': True,
                  'features__message__features__capitalization__capitalizationtransformer__fraction': True,
                  'clf__alpha': 0.00010680435630783633, 'features__message__features__emoji__truncatedsvd__n_iter': 10,
                  'clf__n_neighbors': 13, 'features__message__features__punctuation__truncatedsvd__n_iter': 10,
                  'features__timeofday__timeofdaytransformer__resolution': 4}
        # lens = factory.train(n_estimators=10)
        # lens = factory.train(n_estimators=10, _params=params, _score=0.834)
        lens = factory.train(n_estimators=1, num_truths=1000, num_samples=2000, _params=params, _score=0.834)
        ys_test = [24 if ys == 'spam' else 9 if ys == 'ham' else -1 for ys in ys_test]
        predicted = lens.predict_proba(xs_test)
        plable, pscore = zip(*list(predicted))
        num_wrong = sum([1 if a != b else 0 for a, b in zip(plable, ys_test)])
        print('predicted %d wrong samples from %d (%f correct)' % (num_wrong, num_test, 1 - num_wrong / num_test))
        # factory.persist(lens)
