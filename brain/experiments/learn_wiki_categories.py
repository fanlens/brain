#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB as CLS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from db import get_session, Session

split_AXY = []
with get_session() as session:  # type: Session
    categories_with_text = session.execute("""
SELECT ab.article, array_agg(ct.category), first_value(ab.abstract) OVER (PARTITION BY ab.article)
FROM dbpedia_article_abstracts AS ab, dbpedia_article_categories AS ct
WHERE ab.article = ct.article AND ab.article IN (
    SELECT article FROM dbpedia_article_categories
    WHERE category IN ('Formula_One', 'Sports_cars', 'Sports_car_manufacturers' ))
GROUP BY ab.article
    """)
    for a, y, x in categories_with_text:
        for line in x.split("\n"):
            split_AXY.append((a, y, line.strip()))

A, Y, X = map(np.array, zip(*split_AXY))
print(X)
x_train = X
y_train = Y
multilabel = MultiLabelBinarizer()
multilabel = multilabel.fit(Y)
y_train_bin = multilabel.transform(y_train)

classifier = Pipeline([
    ('vectorizer', CountVectorizer(max_df=0.9, min_df=0.001, stop_words='english')),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(CLS())),
])

x_test = np.array([
    """See more sky, set less limits, and be cool for the summer. Fuel consumption and CO2 emissions for the BMW M4 Convertible: Fuel consumption in l/100 km (combined): 9.1 - 8.7 CO2 emissions in g/km (combined): 213 - 203 Further information about the official fuel consumption and the official specific CO2 emissions for new passenger automobiles can be found in the 'New Passenger Vehicle Fuel Consumption and CO2 Emission Guidelines', which are available free of charge at all sales outlets and from DAT Deutsche Automobil Treuhand GmbH, Hellmuth-Hirth-Str. 1, 73760 Ostfildern, Germany and on""",
    """Ultimate Driving Technology smile emoticon This technology reminded me of the 1997 James Bond movie Tomorrow Never Dies in which Bond controls his E38 750iL with similar technology . I remember that, in those days, many people were underestimating this technology as it is too fictional, not applicable or possible, but they did not know the power and future vision of BMW""",
    """Whomever who haven't notice the power produced by BMW, I bet they only look at structure in cars... M3 with DSG, M4 GTS... Bust mostly I'm still glued to #E30's 325is box shape, 333is. I love BMW and I picture myself driving the most powerful BMW around the whole world... and I will smile emoticon smile emoticon""",
])

classifier.fit(x_train, y_train_bin)
predicted = classifier.predict(x_test)
print(multilabel.inverse_transform(predicted))
