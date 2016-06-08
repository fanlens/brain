#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.pipeline import Pipeline, FeatureUnion

#stack = FeatureUnion([
#    ('text', FeatureUnion([
#        ('language', LanguageExtractor()),
#        ('retina', RetinaExtractor()),
#        ('pos', ([
#
#        ]))
#            text stats (num_adj, num_noun, num_verb)
#            tokens >
#                count
#                tfidf
#        length
#        url_hosts >
#            count
#    ])
#    general >
#        hourofday
#    labels >
#        count
#        tfidf
#
#])
