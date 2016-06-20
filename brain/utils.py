#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy


def np_seeded_train_test_split(xs, ys, limit_train=-1):
    logging.debug("Extracting Seeds")
    logging.debug("=======================================")
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
    logging.debug("Train samples:\t", xs_train.shape)
    logging.debug("Test samples:\t", xs_test.shape)
    logging.debug("=======================================")
    return xs_train, ys_train, xs_test, ys_test
