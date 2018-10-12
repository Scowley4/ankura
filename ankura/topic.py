"""Functions for displaying topics or using them in downstream tasks"""

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


# TODO Add distance metrics
# TODO Add all references as tuple
def cross_reference(corpus, theta_attr, xref_attr,
        doc_ids=None,
        title_attr=None,
        n=sys.maxsize, threshold=1,
        distance='cosine',
        booleanize=False):
    """Finds the nearest documents by topic similarity.

    The documents of the corpus must include a metadata value for theta_attr
    giving a vector representation of the document. Typically, this is a topic
    distribution obtained with assign_topics. The vector representation is then
    used to compute distances between documents. Optionally, cross referencing
    can be computed for just the documents referenced by doc_ids (can be
    iterable for multiple document ids or an int for one document id).

    For the purpose of choosing cross references, the closest n documents will
    be considered (default=sys.maxsize), although Documents whose distance is
    behond the threshold (default=1) are excluded. The resulting cross
    references are stored on each Document of the Corpus under the xref_attr.
    If the title_attr is given, that attribute of the document is stored
    instead of the Document itself.

    Finally, the way distance is computed can be specified using the distance
    parameter (default='cosine'). This can be the name of a distance metric
    supported by scipy.spatial.distance.cdist, or a function which takes two
    topic vectors as input.
    """
    try:
        # make sure doc_ids is iterable
        doc_ids = (d for d in doc_ids)
    except TypeError:
        if doc_ids is None:
            # default to using all docs
            doc_ids = range(len(corpus.documents))
        else:
            # otherwise, assume docs is a single doc id
            doc_ids = [docs]

    thetas = np.array([doc.metadata[theta_attr] for doc in corpus.documents])
    if booleanize:
        thetas = (thetas != 0).astype(int)

    for d in doc_ids:
        dists = spatial.distance.cdist(thetas[d, None], thetas, distance)[0]
        xrefs = dists.argsort()[:n+1] # +1 to account for d being in the list
        xrefs = [i for i in xrefs if dists[i] <= threshold and i != d]
        xrefs = [corpus.documents[i] for i in xrefs]
        if title_attr:
            xrefs = [doc.metadata[title_attr] for doc in xrefs]
        corpus.documents[d].metadata[xref_attr] = xrefs


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
