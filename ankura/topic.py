"""Functions for displaying topics or using them in downstream tasks"""

import sys

import numpy as np
from scipy import sparse
import sklearn.naive_bayes

from . import util


def topic_summary(topics, corpus=None, n=10, stopwords=None):
    """Gets the top n tokens per topic.

    If a vocabulary is provided, the tokens are returned instead of the types.
    """
    if stopwords:
        if not corpus:
            raise ValueError('Corpus cannot be None if stopwords is given')
        stopword_set = set(stopwords)
        include = lambda v: corpus.vocabulary[v] not in stopword_set
    else:
        include = lambda v: True

    summary = []
    for k in range(topics.shape[1]):
        index = []
        for word in np.argsort(topics[:, k])[::-1]:
            if include(word):
                index.append(word)
            if len(index) == n:
                break
        summary.append(index)

    if corpus:
        summary = [[corpus.vocabulary[w] for w in topic] for topic in summary]
    return summary


def highlight(doc, z_attr, highlighter=lambda w, z: '{}:{}'.format(w, z)):
    """Gets the text of a Document with the topic assignments highlighted"""
    chunks = []
    curr = 0

    for token, topic in zip(doc.tokens, doc.metadata[z_attr]):
        start, end = token.loc
        chunks.append(doc.text[curr:start])
        chunks.append(highlighter(doc.text[start:end], topic))
        curr = end
    chunks.append(doc.text[curr:])

    return ''.join(chunks)


def classifier(train, topics, label_attr='label', z_attr='z', alpha=.01):
    K = topics.shape[1]
    V = len(train.vocabulary)

    train_data = _sparse_topic_word(train, K, V, z_attr)
    train_labels = [doc.metadata[label_attr] for doc in train.documents]
    model = sklearn.naive_bayes.MultinomialNB(alpha)
    model.fit(train_data, train_labels)

    return lambda test: model.predict(_sparse_topic_word(test, K, V, z_attr)).tolist()


def _sparse_topic_word(corpus, K, V, z_attr):
    D = len(corpus.documents)
    data = sparse.lil_matrix((D, K*V))
    for d, doc in enumerate(corpus.documents):
        for w, z in zip(doc.tokens, doc.metadata[z_attr]):
            data[d, w.token + z*V] += 1
    return data
