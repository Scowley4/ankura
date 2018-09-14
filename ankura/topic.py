"""Functions for displaying topics or using them in downstream tasks"""

import collections
import sys

import numpy as np
from scipy import spatial, sparse
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


def cross_reference(corpus, theta_attr, xref_attr, title_attr=None,
        n=sys.maxsize, threshold=1, doc_ids=None):
    """Finds the nearest documents by topic similarity.

    The documents of the corpus must include a metadata value for theta_attr
    giving a vector representation of the document. Typically, this is a topic
    distribution obtained with assign_topics. The vector representation is then
    used to compute distances between documents.

    For the purpose of choosing cross references, the closest n documents will
    be considered (default=sys.maxsize), although Documents whose similarity is
    behond the threshold (default=1) are excluded.  A threshold of 1 indicates
    that no filtering should be done, while a 0 indicates that only exact
    matches should be returned. The resulting cross references are stored on
    each document of the Corpus under the xref_attr.
    """
    if doc_ids is None:
        doc_ids = range(len(corpus.documents))
    for d in doc_ids:
        doc = corpus.documents[d]
        doc_theta = doc.metadata[theta_attr]
        dists = [spatial.distance.cosine(doc_theta, d.metadata[theta_attr])
                 if doc is not d else float('nan')
                 for d in corpus.documents]
        dists = np.array(dists)
        xrefs = [corpus.documents[i] for i in dists.argsort()[:n] if dists[i] <= threshold]
        if title_attr:
            xrefs = [doc.metadata[title_attr] for doc in xrefs]
        doc.metadata[xref_attr] = xrefs


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
