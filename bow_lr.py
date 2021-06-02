# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.compose import make_column_transformer


NB_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])


SGD_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])


lr_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(penalty='l2',  random_state=42,
                             max_iter=1000, tol=0.0001)),
])



# start lr_plus_clf

titlepipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])

txtpipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])

titletrans = make_column_transformer([titlepipe, 'title'])
txttrans = make_column_transformer([titlepipe, 'text'])


lr_plus_clf = Pipeline([
    ('union', FeatureUnion([('tittrans',titletrans),
                                               ('txttrans', txttrans)])),
    ('clf', LogisticRegression(penalty='l2', random_state=42,
                               max_iter=1000, tol=0.0001)),
])