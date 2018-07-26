"""Functions for using and displaying topics"""

import collections
import sys

import gensim as gs
import numpy as np
from scipy import spatial, sparse
import sklearn.naive_bayes

from . import util


def topic_summary(topics, corpus=None, n=10):
    """Gets the top n tokens per topic.

    If a vocabulary is provided, the tokens are returned instead of the types.
    """
    summary = []
    for k in range(topics.shape[1]):
        index = []
        for word in np.argsort(topics[:, k])[-n:][::-1]:
            index.append(word)
        summary.append(index)

    if corpus:
        summary = [[corpus.vocabulary[w] for w in topic] for topic in summary]
    return summary


def variational_assign(corpus, topics, theta_attr=None, z_attr=None):
    """Assigns topics using variational inference (via gensim).

    If theta_attr is given, each document will be given a per-document topic
    distribution.  If z_attr is given, each document will be given a sequence
    of topic assignments corresponding to the tokens in the document. One or
    both of these metadata attribute names must be given.
    """
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    V, K = topics.shape
    lda = gs.models.LdaModel(
        num_topics=K,
        id2word={i: i for i in range(V)}, # LdaModel gets V from this dict
    )
    lda.state.sstats = topics.astype(lda.dtype).T * len(corpus.documents)
    lda.sync_state()

    bows = _gensim_bows(corpus)
    _gensim_assign(corpus, bows, lda, theta_attr, z_attr)


def sampling_assign(corpus, topics, theta_attr=None, z_attr=None, alpha=0.1, num_iters=100):
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    T = topics.shape[1]

    c = np.zeros((len(corpus.documents), T))
    z = [np.random.randint(T, size=len(d.tokens)) for d in corpus.documents]
    for d, z_d in enumerate(z):
        for z_dn in z_d:
            c[d, z_dn] += 1

    for _ in range(num_iters):
        for d, (doc, z_d) in enumerate(zip(corpus.documents, z)):
            for n, w_dn in enumerate(doc.tokens):
                c[d, z_d[n]] -= 1
                cond = [alpha + c[d, t] * topics[w_dn.token, t] for t in range(T)]
                z_d[n] = util.sample_categorical(cond)
                c[d, z_d[n]] += 1

    if theta_attr:
        for doc, c_d in zip(corpus.documents, c):
            doc.metadata[theta_attr] = c_d / c_d.sum()
    if z_attr:
        for doc, z_d in zip(corpus.documents, z):
            doc.metadata[z_attr] = z_d.tolist()


def mode_assign(corpus, topics, theta_attr=None, z_attr=None, max_iters=100):
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    T = topics.shape[1]

    c = np.zeros((len(corpus.documents), T))
    z = [np.random.randint(T, size=len(d.tokens)) for d in corpus.documents]
    for d, z_d in enumerate(z):
        for z_dn in z_d:
            c[d, z_dn] += 1

    for i in range(max_iters):
        change = False
        for d, (doc, z_d) in enumerate(zip(corpus.documents, z)):
            for n, w_dn in enumerate(doc.tokens):
                old = z_d[n]
                c[d, old] -= 1

                cond = [c[d, t] * topics[w_dn.token, t] for t in range(T)]
                new = np.argmax(cond)
                change = change or new != old

                z_d[n] = new
                c[d, new] += 1

        if not change:
            break

    if theta_attr:
        for doc, c_d in zip(corpus.documents, c):
            doc.metadata[theta_attr] = c_d / c_d.sum()
    if z_attr:
        for doc, z_d in zip(corpus.documents, z):
            doc.metadata[z_attr] = z_d.tolist()


def mode_assign2(corpus, topics, theta_attr=None, z_attr=None, alpha=.01, max_iters=100, n=10):
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    for doc in corpus:
        best_z, best_c, best_p = _mode(doc, topics, max_iters, alpha)

        for _ in range(1, n):
            cand_z, cand_c, cand_p = _mode(doc, topics, max_iters, alpha)
            if cand_p > best_p:
                best_z, best_c, best_p = cand_z, cand_c, cand_p

        if theta_attr:
            doc.metadata[theta_attr] = best_c / best_c.sum()
        if z_attr:
            doc.metadata[z_attr] = best_z.tolist()


def _mode(doc, topics, max_iters, bound):
    T = topics.shape[1]

    z = np.random.randint(T, size=len(doc.tokens))
    c = np.zeros(T)
    for z_n in z:
        c[z_n] += 1

    for _ in range(max_iters):
        changed = False
        for n, w_n in enumerate(doc.tokens):
            old = z[n]
            c[old] -= 1

            new = np.argmax([c[t] * topics[w_n.token, t] for t in range(T)])
            change = change or new != old

            z[n] = new
            c[new] += 1

        if not change:
            break

    p = sum(np.log(alpha + topics[w_n.token, z_n]) for w_n, z_n in zip(doc.tokens, z))
    return z, c, p



def _gensim_bows(corpus):
    bows = []
    for doc in corpus.documents:
        bow = collections.defaultdict(int)
        for t in doc.tokens:
            bow[t.token] += 1
        bows.append(bow)
    return [list(bow.items()) for bow in bows]


def _gensim_assign(corpus, bows, lda, theta_attr, z_attr):
    for doc, bow in zip(corpus.documents, bows):
        gamma, phi = lda.inference([bow], collect_sstats=z_attr)
        if theta_attr:
            doc.metadata[theta_attr] = gamma[0] / gamma[0].sum()
        if z_attr:
            w = [t.token for t in doc.tokens]
            doc.metadata[z_attr] = phi.argmax(axis=0)[w].tolist()


def _icm(doc, topics, alpha):
    T = topics.shape[1]
    z = np.random.randint(T, size=len(doc.tokens))
    c = np.zeros(T)
    for z_n in z:
        c[z_n] += 1

    while True:
        change = False

        for n, (w_n, z_n) in enumerate(zip(doc.tokens, z)):
            c[z_n] -= 1
            cond = [alpha + c[t] * topics[w_n.token, t] for t in range(T)]
            z[n] = np.argmax(cond)
            c[z[n]] += 1
            if z[n] != z_n:
                change = True

        if not change:
            break

    return z


def cross_reference(corpus, theta_attr, xref_attr, n=sys.maxsize, threshold=1):
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
    for doc in corpus.documents:
        doc_theta = doc.metadata[theta_attr]
        dists = [spatial.distance.cosine(doc_theta, d.metadata[theta_attr])
                 if doc is not d else float('nan')
                 for d in corpus.documents]
        dists = np.array(dists)
        xrefs = list(corpus.documents[i] for i in dists.argsort()[:n] if dists[i] <= threshold)
        doc.metadata[xref_attr] = xrefs


def highlight(doc, z_attr, highlighter=lambda w, z: '{}:{}'.format(w, z)):
    chunks = []
    curr = 0

    for token, topic in zip(doc.tokens, doc.metadata[z_attr]):
        start, end = token.loc
        chunks.append(doc.text[curr:start])
        chunks.append(highlighter(doc.text[start:end], topic))
        curr = end
    chunks.append(doc.text[curr:])

    return ''.join(chunks)


def _sparse_topic_word(corpus, K, V, z_attr):
    D = len(corpus.documents)
    data = sparse.lil_matrix((D, K*V))
    for d, doc in enumerate(corpus.documents):
        for w, z in zip(doc.tokens, doc.metadata[z_attr]):
            data[d, w.token + z*V] += 1
    return data

def classifier(train, topics, label_attr='label', z_attr='z', alpha=.01):
    K = topics.shape[1]
    V = len(train.vocabulary)

    train_data = _sparse_topic_word(train, K, V, z_attr)
    train_labels = [doc.metadata[label_attr] for doc in train.documents]
    model = sklearn.naive_bayes.MultinomialNB(alpha)
    model.fit(train_data, train_labels)

    return lambda test: model.predict(_sparse_topic_word(test, K, V, z_attr)).tolist()
