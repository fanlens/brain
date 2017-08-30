#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
import redis
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Perceptron as CLFCLS
from sklearn.pipeline import Pipeline

from brain.feature.language_detect import is_english
from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from db import get_session

PostReaction = namedtuple('PostReaction', ['post_id', 'type', 'rank', 'count', 'texts'])

r = redis.StrictRedis(host='localhost', port=6379, db=1)

print('fetch and preapre data...')
tokenizer = LemmaTokenTransformer(short_url=True)

meta_key, force = 'agglo', False
if not r.exists(meta_key) or force:
    r.delete(meta_key)
    with get_session() as session:
        top_reactions = session.execute("""
    select agg.post_id, agg.type, agg.r, agg.c, array_agg(comments.comment::jsonb->>'message') as text from (
      select post_id, type, dense_rank() over (partition by post_id order by count(type) desc) as r, count(*) as c
      from facebook_reactions
      where type != 'LIKE' and page = '%(page)s'
      group by post_id, type
      ) as agg, facebook_comments as comments
    where agg.post_id = comments.post_id and agg.r <= 2 and agg.c >= 2 and char_length(comments.comment::jsonb->>'message') > 140
    group by agg.post_id, agg.type, agg.r, agg.c
    order by agg.post_id, agg.r""" % {'page': 'ladygaga'})
        for reaction in map(lambda r: PostReaction(*r), top_reactions):
            reaction = reaction  # type: PostReaction
            c = 0
            for text in reaction.texts:
                if not is_english(text):
                    continue
                tokenized = tokenizer(text)
                r.rpush(reaction.post_id + '_' + str(c), *tokenized)
                r.set(reaction.post_id + '_' + str(c) + '_orig', text)
                c += 1
        r.set(meta_key, True)

print('... done;')

print('fetching samples from redis...')
X, X_orig, Y = [], [], []
for post_id in map(lambda id: id.decode("utf-8"), r.keys('*_*_*')):
    if post_id.endswith('_orig'):
        continue
    X_orig.append(r.get(post_id + '_orig').decode("utf-8"))
    X.append(' '.join([b.decode("utf-8") for b in r.lrange(post_id, 0, -1)]))
    Y.append('_'.join(post_id.split('_')[:2]))
for sent in list(zip(X, X_orig))[-1:-5:-1]:
    print('\n'.join(sent), "\n\n")
exit()
print(len(X), len(Y))
X, Y = np.array(X), np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print('... done;')

print('classifying...')
pipeline = Pipeline([
    ('vec', CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.01, max_df=0.9, strip_accents='unicode')),
    ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)),
    ('clf', CLFCLS()),
])
pipeline.fit(x_train, y_train)
predicted = pipeline.predict(x_test)
print(metrics.classification_report(y_test, predicted))
