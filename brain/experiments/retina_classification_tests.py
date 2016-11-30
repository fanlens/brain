#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import retinasdk
import numpy as np

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split

from db import DB
from tools.facebook_retina_top import FacebookRetinaEntry
from config.db import Config
from config.env import Environment

SETTINGS = Environment("PATHS")
PICKLE_FILE_PATH = os.path.join(SETTINGS['model_dir'], 'retina_page_classifier.pickle')


def train():
    print('fetch and prepare data...')
    xs, ys, ts = [], [], []
    with DB().ctx() as session:
        for entry in session.query(FacebookRetinaEntry):
            sample_retina = np.zeros(128 * 128, dtype=int)
            sample_retina[entry.data['retina']] = 1
            xs.append(sample_retina)
            ys.append(entry.slug)
            ts.append(entry.data['tokens'])

    xs, ys = np.array(xs), np.array(ys)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.33, random_state=42)
    print('... done;')

    print('training classifier...')
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    print(metrics.classification_report(y_test, predicted))
    print('done')

    with open(PICKLE_FILE_PATH, 'wb') as pickle_file:
        pickle.dump(clf, pickle_file)


if __name__ == "__main__":
    if not os.path.exists(PICKLE_FILE_PATH):
        train()
    with open(PICKLE_FILE_PATH, 'rb') as pickle_file:
        clf = pickle.load(pickle_file)
        config = Config("cortical")
        fullClient = retinasdk.FullClient(config["api_key"])
        test_txt = "Burning Down The House only sounds perfect when The Talking Heads perform it."
        positions = fullClient.getFingerprintForText(test_txt).positions
        test_sample = np.zeros(128 * 128, dtype=int)
        test_sample[positions] = 1
        print(clf.predict_text([test_sample]))
