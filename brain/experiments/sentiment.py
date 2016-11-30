#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""train a sentiment model, based on scikit tutorial"""

import os
import enum
import typing
import pickle

from config.env import Environment


class Sentiment(enum.Enum):
    """values a sentiment can be"""
    negative = 0
    positive = 1
    neutral = 2
    unknown = 3


class SentimentLens(object):
    """tool to get sentiment for texts"""
    _sentiment_classifier = None
    _sentiment_target_names = None

    def __init__(self):
        if self._sentiment_classifier is None:
            self._load_model()

    def sentiment(self, text: str) -> Sentiment:
        """:return: sentiment for the provided text"""
        return self.sentiment_batch([text])[0]

    def sentiment_batch(self, texts: typing.List[str]) -> typing.List[Sentiment]:
        """:return: sentiments for the provided texts"""
        predicted_sentiments = self._sentiment_classifier.predict_text(texts)
        return [Sentiment(predicted) for predicted in predicted_sentiments]

    @classmethod
    def _load_model(cls):
        settings = Environment('PATHS')
        sentiment_model_file_path = os.path.join(settings['model_dir'], 'sentiment.pickle')
        with open(sentiment_model_file_path, 'rb') as input_sentiment_model_file:
            cls._sentiment_classifier, cls._sentiment_target_names = pickle.load(input_sentiment_model_file)


if __name__ == "__main__":
    import sys

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.grid_search import GridSearchCV
    from sklearn.datasets import load_files
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = sys.argv[1]
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    clf = Pipeline([
        ('vectorizer', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('classifier', LinearSVC(C=1000)),
    ])

    # Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
    }
    grid_search = GridSearchCV(clf, parameters, n_jobs=-1)
    grid_search.fit(docs_train, y_train)

    # print the cross-validated scores for the each parameters set
    # explored by the grid search
    print(grid_search.grid_scores_)

    # Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = grid_search.predict(docs_test)
    # y_predicted = grid_search.best_estimator_.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    # from matplotlib import pylab as pl
    # pl.matshow(cm, cmap=pl.cm.jet)
    # pl.show()

    with open('sentiment.pickle', 'wb') as sentiment_model_file:
        pickle.dump((grid_search, dataset.target_names), sentiment_model_file)
