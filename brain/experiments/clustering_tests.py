#!/usr/bin/env python
# -*- coding: utf-8 -*-


import enum
from collections import namedtuple

import numpy as np
import pycld2
from sklearn.cluster import Birch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from db import DB, Session


class Reaction(enum.Enum):
    LOVE = 0
    ANGRY = 1
    HAHA = 2
    WOW = 3
    SAD = 4


PostReaction = namedtuple('Reaction', ['post_id', 'type', 'rank', 'count', 'text'])

XY = {}
with DB().ctx() as session:
    session = session  # type: Session
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
    c = 0
    for reaction in (PostReaction(*reaction_tup) for reaction_tup in top_reactions):
        c += 1
        current_reactions, _ = XY.get(reaction.post_id, ([], ''))
        current_reactions.append(Reaction[reaction.type].value)
        XY[reaction.post_id] = (current_reactions, reaction.text)

flat_XY = [(labels, text.strip()) for labels, texts in XY.values() for text in texts if
           pycld2.detect(text)[2][0][1] == 'en']
print(len(flat_XY))

Y, X = map(np.array, zip(*flat_XY))

tokenizer = LemmaTokenTransformer()
classifier = Pipeline([
    ('vectorizer',
     #the experience in Information Retrieval (Salton's Vector Space Model) is that those terms ocurring between 1% and 10% of the documents are the most discriminative ones, that is, the ones that help best to separate the space of documents.
     CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=tokenizer, stop_words='english', min_df=0.01, max_df=0.1)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('clf', Birch(branching_factor=150, n_clusters=None, threshold=0.7, compute_labels=True)),
])

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

classifier.fit(X)
labels = classifier.predict(X)

histos = {}
for lbl, txt in zip(labels, X):
    if lbl not in histos:
        histos[lbl] = {}
    histo = histos[lbl]
    tokens = tokenizer(txt)
    for token in tokens:
        if token in stops or "'" in token or token == 'wa':
            continue
        if token not in histo:
            histo[token] = 1
        else:
            histo[token] += 1

print(len(histos.keys()), 'labels')

for lbl, histo in histos.items():
    max_val = max(histo.values())
    print("\n\nhistogram for label:", lbl)
    for txt, val in histo.items():
        if val / max_val > 0.8:
            print(txt, '\t', val / max_val * 100)

