#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tests to predict reactions from comments"""

import enum
import pickle
from collections import namedtuple

import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from db import DB


class Reaction(enum.Enum):
    LOVE = 0
    ANGRY = 1
    HAHA = 2
    WOW = 3
    SAD = 4


PostReaction = namedtuple('Reaction', ['post_id', 'type', 'rank', 'count', 'text'])

import redis
page = 'Formula1'
force = False

r = redis.StrictRedis(host='localhost', port=6379, db=0)
if page not in r or force:
    with DB().ctx() as session:
        print('filling cache')
        r.delete(page)
        r.rpush(page, *list(map(pickle.dumps, session.execute("""
select agg.post_id, agg.type, agg.r, agg.c, string_agg(comments.comment::jsonb->>'message', E'\n') as text from (
  select post_id, type, dense_rank() over (partition by post_id order by count(type) desc) as r, count(*) as c
  from facebook_reactions
  where type != 'LIKE' and page = '%(page)s'
  group by post_id, type
  ) as agg, facebook_comments as comments
where agg.post_id = comments.post_id and agg.r <= 2 and agg.c >= 2 and char_length(comments.comment::jsonb->>'message') > 140
group by agg.post_id, agg.type, agg.r, agg.c
order by agg.post_id, agg.r""" % {'page': page}))))

XY = {}
top_reactions = r.lrange(page, 0, -1)
for reaction in map(pickle.loads, top_reactions):
    print(reaction)
exit()
c = 0
for reaction in (PostReaction(*reaction_tup) for reaction_tup in top_reactions):
    c += 1
    current_reactions, _ = XY.get(reaction.post_id, ([], ''))
    current_reactions.append(Reaction[reaction.type].value)
    XY[reaction.post_id] = (current_reactions, reaction.text)

flat_XY = [(labels, text.strip()) for labels, texts in XY.values() for text in texts.split("\n")]

Y, X = map(np.array, zip(*flat_XY))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
y_train_bin = MultiLabelBinarizer().fit_transform(y_train)
y_test_bin = MultiLabelBinarizer().fit_transform(y_test)

classifier = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word', tokenizer=LemmaTokenTransformer(), stop_words='english', min_df=5)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('clf', OneVsRestClassifier(MultinomialNB())),
])

classifier.fit(x_train, y_train_bin)
predicted = classifier.predict(x_test)
print(metrics.classification_report(y_test_bin, predicted))

# for item, labels in zip(x_test, predicted):
#    print(labels)
#    print(item, ', '.join(Reaction(label).name for (label, hit) in enumerate(labels) if hit))
