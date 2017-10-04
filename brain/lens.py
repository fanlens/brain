#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""a module for tagging texts based on semi supervised methods"""
import datetime
import logging
import operator
import os
import pickle
import random
import uuid
from typing import NamedTuple, NewType, List, Iterable, Union, Any, Dict, Optional, DefaultDict, Tuple, cast

import numpy
import scipy.stats
from sklearn.decomposition import TruncatedSVD
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.semi_supervised import LabelSpreading
from sqlalchemy import text
from sqlalchemy.orm import Session

from brain.feature.capitalization import CapitalizationTransformer
from brain.feature.emoji import EmojiTransformer
from brain.feature.field_extract import FieldExtractTransformer
from brain.feature.fingerprint import SparseFingerprintTransformer, TFingerprint
from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from brain.feature.punctuation import PunctuationTransformer
from brain.feature.timeofday import TimeOfDayTransformer
from config import get_config
from db import get_session
from db.models.activities import TagSet, Source, User
from db.models.brain import Model
from utils.progress import ProgressCallbackBase

TPrediction = NewType('TPrediction', int)
ScoredPrediction = NamedTuple('ScoredPrediction', [('tag_id', int), ('score', float)])
TScoredPredictionSet = List[ScoredPrediction]
TEstimatorBag = List[Pipeline]
Sample = NamedTuple('Sample', [('text', str), ('fingerprint', TFingerprint), ('time', datetime.datetime)])

_LOGGER = logging.getLogger()
_CONFIG = get_config()
MODEL_FILE_ROOT = _CONFIG.get('BRAIN', 'modelfilepath')

_RANDOM_SAMPLE_SQL = text('''
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
  SELECT dataT.id, random_ids.tag_id, random_ids.language, textT.text, timeT.time, fpT.fingerprint, translationT.translation
  FROM activity.data AS dataT
       INNER JOIN random_ids ON dataT.id = random_ids.id
       INNER JOIN activity.source AS srcT ON dataT.source_id = srcT.id
       INNER JOIN activity.text AS textT ON dataT.id = textT.data_id
       INNER JOIN activity.fingerprint AS fpT ON dataT.id = fpT.data_id
       INNER JOIN activity.time AS timeT ON dataT.id = timeT.data_id
       LEFT OUTER JOIN activity.translation as translationT ON textT.id = translationT.text_id AND translationT.target_language = 'en'
''')


def model_file_path(model_id: Union[uuid.UUID, str]) -> str:
    """
    :param model_id: id of the model
    :return: the full path to the model
    """
    return str(os.path.join(MODEL_FILE_ROOT, str(model_id)))


# http://stackoverflow.com/questions/1257413/iterate-over-pairs-in-a-list-circular-fashion-in-python
def ntuples(lst: List[Any], group_size: int) -> Iterable[Iterable[Any]]:
    """
    Iterate over pairs in a list in circular fashion
    e.g. ntuples([3, 5, 6, 7], 3) -> [(3, 5, 6), (5, 6, 7), (6, 7, 3), (7, 3, 5)]
    :param lst: original list
    :param group_size: size of the groups the list is split into
    :return: chunked up list
    """
    chunked_iter: Iterable[Iterable[Any]] = zip(*[lst[i:] + lst[:i] for i in range(group_size)])
    return chunked_iter


class Lens(object):
    """A predictor that analyzes texts and predicts class tags"""

    def __init__(self, model: Model, estimator_bag: TEstimatorBag) -> None:
        self._estimator_bag = estimator_bag
        self._model = model

    @classmethod
    def load_from_id(cls, model_id: uuid.UUID) -> 'Lens':
        """
        Load a model from the specified id.
        :param model_id: id of the model
        :return: a `Lens` instance loaded from the system
        :raises RuntimeError: if the model could not be loaded
        """
        with get_session() as session:
            model = session.query(Model).get(model_id)  # type: Model

        estimator_bag = []  # type: TEstimatorBag
        try:
            with open(model_file_path(model_id), 'rb') as model_file:
                estimator_bag = pickle.load(model_file)
        except FileNotFoundError:
            _LOGGER.exception('Could not load model %s', str(model_id))

        if not model or not estimator_bag:
            raise RuntimeError("Could not load model %s", str(model_id))

        return Lens(model, estimator_bag)

    @property
    def model(self) -> Model:
        """:return: the underlying `Model`"""
        return self._model

    @property
    def estimator_bag(self) -> TEstimatorBag:
        """:return: the underlying estimators"""
        return self._estimator_bag

    def predict_proba(self, xs: Iterable[Sample]) -> Iterable[TScoredPredictionSet]:
        """
        Create predictions for all provided samples.
        Predictions are of the form [tag_id, score]
        A Prediction set contains multiple Predictions [[tag_id1, score1], [tag_id2, score2]]
        :param xs: samples
        :return: predictions for all samples
        """
        n_bags = len(self.estimator_bag)
        for votes in zip(*[estimator.predict(xs) for estimator in self.estimator_bag]):
            # transposed predictions
            # xs:    [x1, x2, x3]
            # [
            #   cls1:  [v1.1, v1.2, v1.3]   # from 1 of the classifiers
            #   cls2:  [v2.1, v2.2, v2.3]   # from 1 of the classifiers
            # ]
            # zip(*...) -> [
            #  votes1:  [v1.1, v2.1]   # all votes for sample
            #  votes2:  [v1.2, v2.2]   # all votes for sample
            # ]
            counter = DefaultDict[int, int](int)
            for vote in votes:
                counter[vote] += 1
            yield [ScoredPrediction(int(tag), score / n_bags) for tag, score in counter.items()]

    def predict(self, xs: Iterable[Sample]) -> Iterable[TPrediction]:
        """
        Same as predict_proba but only returns the tag id which has the highest score
        :param xs: samples
        :return: predicted tag id for all samples
        """
        for tags_proba in self.predict_proba(xs):
            tag, _ = max(tags_proba, operator.itemgetter(1))
            yield tag


def make_weight_dist(*weight_names: str, num_samples: int = 50) -> Iterable[Dict[str, float]]:
    """
    Helper method to create a quick weight distribution
    :param weight_names: the name of the transformers whose weights are set
    :param num_samples: how many random weight instances to deliver
    :return: a list of dictionaries which are keyed by the transformer name and have random values
    """
    for _ in range(0, num_samples):
        yield dict((name, random.random()) for name in weight_names)


class LensTrainer(object):
    """Factory to create and persist `Lens` objects"""
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

    def __init__(self, user: User, tagset: TagSet, sources: Iterable[Source],
                 progress: Optional[ProgressCallbackBase] = None) -> None:
        """
        :param user: who is training this `Lens`
        :param tagset: the `TagSet` this `Lens` is based on
        :param sources: the `Source` this `Lens` is based on
        :param progress: an optional progress callback that informs external services about the current state
        """
        assert tagset and sources
        assert tagset.user.filter_by(id=user.id).one_or_none()
        assert all(source.users.filter_by(id=user.id).one_or_none() for source in sources)
        self._tagset = tagset
        self._sources = sources
        self._user = user
        self._progress = progress

    @staticmethod
    def taggger_pipeline() -> Pipeline:
        """:return: a new Pipeline object"""
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

    def _fetch_samples(self, num_truths: int, num_samples: int) -> Iterable[Tuple[int, Sample]]:
        _LOGGER.debug("fetching sampleset with %d truths and %d samples...", num_truths, num_samples)
        with get_session() as session:
            query = session.execute(_RANDOM_SAMPLE_SQL,
                                    dict(tagset_id=self._tagset.id,
                                         num_samples=num_samples,
                                         num_truths=num_truths,
                                         langs=tuple(['en', 'de']),  # todo find better way to filter for translations
                                         sources=tuple(source.id for source in self._sources)))
            for _id, tag_id, _language, message, created_time, fingerprint, translation in query:
                yield tag_id or -1, Sample(translation or message, fingerprint, created_time)
        _LOGGER.debug("... done fetching sampleset")

    def _find_params(self, num_truths: int, num_samples: int, num_folds: int = 3) -> Tuple[Dict[str, Any], float]:
        _LOGGER.debug("search best parameters using %d folds...", num_folds)
        folds = [list(self._fetch_samples(num_truths=num_truths, num_samples=num_samples)) for _ in range(0, num_folds)]
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
        _LOGGER.debug("... done searching parameters, achieved score of %f with params %s", search.best_score_,
                      search.best_params_)
        return search.best_params_, search.best_score_

    def _train_stub(self, params: Dict[str, Any], num_truths: int, num_samples: int,
                    _stub_id: uuid.UUID = uuid.uuid1()) -> Pipeline:
        ys, xs = zip(*self._fetch_samples(num_truths=num_truths, num_samples=num_samples))
        ys = numpy.array(ys)
        _LOGGER.debug('training stub %s...', _stub_id)
        pipeline = self.taggger_pipeline()
        pipeline.set_params(**params)
        pipeline.fit(xs, ys)
        _LOGGER.debug('... done training stub %s', _stub_id)
        return pipeline

    def train(self,
              n_estimators: int = 100,
              num_truths: int = 48,
              num_samples: int = 128,
              _params: Optional[Dict[str, Any]] = None,
              _score: float = 0.0) -> Lens:
        """
        Train this lens. A lens is trained semi supervised, so the ratio of truths and samples can be specified
        :param n_estimators: how many estimators to train for the estimation bag
        :param num_truths: how many truth values should each estimator have
        :param num_samples: how many unlabeled samples to include
        :param _params: (dangerous) override parameters
        :param _score: (dangerous) override final score
        :return: a newly trained `Lens` instance.
        """
        _LOGGER.debug('training estimator bag with %d estimators...', n_estimators)
        if _params:
            params, score = _params, _score
        else:
            params, score = self._find_params(num_truths=num_truths, num_samples=num_samples)

        bag = Parallel(n_jobs=-1)(
            delayed(self._train_stub)(params, num_truths, num_samples, i) for i in range(0, n_estimators))
        model = Model(tagset_id=self._tagset.id, created_by_user_id=self._user.id, params=params, score=score)
        model.users.append(self._user)
        for source in self._sources:
            model.sources.append(source)
        _LOGGER.debug('... done training estimator bag')
        return Lens(model, bag)

    @staticmethod
    def persist(lens: Lens, session: Session) -> uuid.UUID:
        """
        Persist `Lens`. Store Metadata in the database and serialize the estimator to the file system.
        :param lens: the `Lens` to persist
        :param session: the database session to use
        :return: the newly persisted lens' model id
        """
        _LOGGER.debug('persisting trained model at...')
        session.add(lens.model)
        session.commit()
        with open(model_file_path(lens.model.id), 'wb') as model_file:
            pickle.dump(lens.estimator_bag, model_file, protocol=pickle.HIGHEST_PROTOCOL)
        _LOGGER.debug('... done persisting trained model, new id is: %s', lens.model.id)
        return cast(uuid.UUID, lens.model.id)

# if __name__ == "__main__":
#    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
#    logger = logging.getLogger(__name__)
#    logger.setLevel(logging.DEBUG)
#    ch = logging.StreamHandler()
#    ch.setLevel(logging.DEBUG)
#    logger.addHandler(ch)
#
#    # with open('/Users/chris/Projects/fanlens/data/testset.pickle', 'rb') as testsetfile:
#    #     ys_test, xs_test = zip(*pickle.load(testsetfile))
#    #     num_test = len(ys_test)
#
#    with get_session() as session:
#        # tagset = session.query(TagSet).get(1)
#        # sources = session.query(Source).filter(Source.id.in_((2,))).all()
#        # factory = LensTrainer(tagset, sources)
#        params = {'features__fingerprint__truncatedsvd__n_iter': 10,
#                  'features__message__features__tokens__truncatedsvd__n_components': 106,
#                  'features__timeofday__timeofdaytransformer__resolution': 5,
#                  'features__message__features__punctuation__truncatedsvd__n_iter': 3, 'clf__max_iter': 111,
#                  'clf__n_neighbors': 2, 'clf__gamma': 17.222983787280945,
#                  'features__message__features__emoji__truncatedsvd__n_components': 16,
#                  'features__fingerprint__truncatedsvd__n_components': 5, 'clf__alpha': 0.3307067042583709,
#                  'features__message__features__tokens__truncatedsvd__n_iter': 4,
#                  'features__message__features__emoji__truncatedsvd__n_iter': 10, 'clf__kernel': 'rbf',
#                  'features__message__features__tokens__lemmatokentransformer__short_url': False,
#                  'features__message__features__punctuation__punctuationtransformer__strict': False,
#                  'features__message__features__punctuation__truncatedsvd__n_components': 17,
#                  'features__message__features__capitalization__capitalizationtransformer__fraction': True}
#        params = {'clf__max_iter': 151, 'clf__gamma': 15.207618172582782,
#                  'features__message__features__punctuation__truncatedsvd__n_components': 3,
#                  'features__fingerprint__truncatedsvd__n_components': 3,
#                  'features__message__features__tokens__truncatedsvd__n_iter': 8,
#                  'features__message__features__tokens__truncatedsvd__n_components': 106,
#                  'features__message__features__emoji__truncatedsvd__n_components': 1, 'clf__kernel': 'rbf',
#                  'features__fingerprint__truncatedsvd__n_iter': 5,
#                  'features__transformer_weights': {'timeofday': 0.2030570157381013, 'message': 0.6336048802942279,
#                                                    'fingerprint': 0.7186676931811034},
#                  'features__message__features__punctuation__punctuationtransformer__strict': True,
#                  'features__message__features__tokens__lemmatokentransformer__short_url': True,
#                  'features__message__features__capitalization__capitalizationtransformer__fraction': True,
#                  'clf__alpha': 0.00010680435630783633, 'features__message__features__emoji__truncatedsvd__n_iter': 10,
#                  'clf__n_neighbors': 13, 'features__message__features__punctuation__truncatedsvd__n_iter': 10,
#                  'features__timeofday__timeofdaytransformer__resolution': 4}
#        # lens = factory.train(n_estimators=10)
#        # lens = factory.train(n_estimators=10, _params=params, _score=0.834)
#        # lens = factory.train(n_estimators=1, num_truths=1000, num_samples=2000, _params=params, _score=0.834)
#        # ys_test = [24 if ys == 'spam' else 9 if ys == 'ham' else -1 for ys in ys_test]
#        # predicted = lens.predict_proba(xs_test)
#        # plable, pscore = zip(*list(predicted))
#        # num_wrong = sum([1 if a != b else 0 for a, b in zip(plable, ys_test)])
#        # print('predicted %d wrong samples from %d (%f correct)' % (num_wrong, num_test, 1 - num_wrong / num_test))
#        # factory.persist(lens)
#
#        tagset = session.query(TagSet).get(6)
#        sources = session.query(Source).filter(Source.id.in_((9,))).all()
#        factory = LensTrainer(tagset, sources)
#        kjero_params = {'clf__n_neighbors': 10,
#                        'features__message__features__tokens__lemmatokentransformer__short_url': True,
#                        'features__message__features__punctuation__punctuationtransformer__strict': False,
#                        'features__fingerprint__truncatedsvd__n_iter': 11,
#                        'features__timeofday__timeofdaytransformer__resolution': 1,
#                        'features__message__features__tokens__truncatedsvd__n_iter': 5,
#                        'features__transformer_weights': {'timeofday': 0.5821752747117653,
#                                                          'message': 0.5767438061256892,
#                                                          'fingerprint': 0.7043198536827283},
#                        'clf__alpha': 0.3742258966108718,
#                        'features__message__features__emoji__truncatedsvd__n_iter': 4,
#                        'features__message__features__punctuation__truncatedsvd__n_iter': 11,
#                        'features__message__features__capitalization__capitalizationtransformer__fraction': True,
#                        'features__message__features__emoji__truncatedsvd__n_components': 4,
#                        'features__fingerprint__truncatedsvd__n_components': 6,
#                        'features__message__features__tokens__truncatedsvd__n_components': 25, 'clf__max_iter': 58,
#                        'clf__gamma': 38.065100724023573,
#                        'features__message__features__punctuation__truncatedsvd__n_components': 2,
#                        'clf__kernel': 'rbf'}
#
#        lens = factory.train(n_estimators=10, num_truths=200, num_samples=200, _params=kjero_params)
#
#        print("predicting...")
#        from db.models.activities import Data, Tagging
#
#        xs_test = [(entry.text.translations.one().translation, entry.fingerprint.fingerprint, entry.time.time) for entry
#                   in session.query(Data).join(Tagging, Tagging.data_id == Data.id).filter(Tagging.tag_id == 300)]
#        ys_pred = list(lens.predict_proba(xs_test))
#        num_right = sum([1 if y == 300 else 0 for y, _ in ys_pred])
#        print(num_right, 'right predictions', num_right / len(ys_pred) * 100, '%')
