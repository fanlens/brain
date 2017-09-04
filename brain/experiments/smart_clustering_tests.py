#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
import redis
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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
    order by agg.post_id, agg.r""" % {'page': 'Formula1'})
        for reaction in map(lambda r: PostReaction(*r), top_reactions):
            reaction = reaction  # type: PostReaction
            c = 0
            for text in reaction.texts:
                if not is_english(text):
                    continue
                tokenized = tokenizer(text)
                r.rpush(reaction.post_id + '_' + str(c), *tokenized)
                c += 1
        r.set(meta_key, True)

print('... done;')

print('fetching samples from redis...')
X = []
for post_id in r.keys('*_*_*'):
    X.append(' '.join([b.decode("utf-8")for b in r.lrange(post_id, 0, -1)]))
print(len(X))
X = np.array(X)
print('... done;')

print('clustering...')
# the experience in Information Retrieval (Salton's Vector Space Model) is that those terms ocurring between 1% and 10% of the documents are the most discriminative ones, that is, the ones that help best to separate the space of documents.
# already tokenized
pipeline = Pipeline([
    ('vec', CountVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.01, max_df=1, strip_accents='unicode')),
    ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)),
    ('clu', DBSCAN(eps=0.5, min_samples=10, metric='cosine', algorithm='brute')),
])
predicted = pipeline.fit_predict(X)
print("num clusters:", len(set(pipeline.named_steps['clu'].labels_)))
for label, sample in sorted(zip(predicted, X)):
    print(label, sample)
print('... done.')
