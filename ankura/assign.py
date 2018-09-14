"""Functions for assigning documents and words to topics.

Generally these assign functions all share a basic signature with four
arguments: a corpus, a set of topics, and either a theta_attr or a z_attr (both
can be specfied as well). Depending on which attribute name is given, the
documents or words are assigned to topics, and the changes are reflected in the
Corpus object as metadata attributes on the constituent Document objects.
Beyond these first four parameters, there may be additional optional parameters
which further specify how the assignments should be made.
"""

import gensim as gs
import numpy as np

from . import util

def variational(corpus, topics, theta_attr=None, z_attr=None):
    """Assigns topics using variational inference"""
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    # Setup gensim bookkeping
    V, K = topics.shape
    lda = gs.models.LdaModel(
        num_topics=K,
        id2word={i: i for i in range(V)}, # LdaModel gets V from this dict
    )
    lda.state.sstats = topics.astype(lda.dtype).T * len(corpus.documents)
    lda.sync_state()

    # Covert docs to gensim format
    bows = []
    for doc in corpus.documents:
        bow = collections.defaultdict(int)
        for t in doc.tokens:
            bow[t.token] += 1
        bows.append(bow)
    bows = [list(bow.items()) for bow in bows]

    # Make assignments using gensim inference
    for doc, bow in zip(corpus.documents, bows):
        gamma, phi = lda.inference([bow], collect_sstats=z_attr)
        if theta_attr:
            doc.metadata[theta_attr] = gamma[0] / gamma[0].sum()
        if z_attr:
            w = [t.token for t in doc.tokens]
            doc.metadata[z_attr] = phi.argmax(axis=0)[w].tolist()


def sampling(corpus, topics, theta_attr=None, z_attr=None, alpha=0.1, num_iters=100):
    """Assigns topics using Gibbs sampling"""
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    T = topics.shape[1]

    # Setup Gibbs sampler bookkeeping
    c = np.zeros((len(corpus.documents), T))
    z = [np.random.randint(T, size=len(d.tokens)) for d in corpus.documents]
    for d, z_d in enumerate(z):
        for z_dn in z_d:
            c[d, z_dn] += 1

    # Perform sampling
    for _ in range(num_iters):
        for d, (doc, z_d) in enumerate(zip(corpus.documents, z)):
            for n, w_dn in enumerate(doc.tokens):
                c[d, z_d[n]] -= 1
                cond = [alpha + c[d, t] * topics[w_dn.token, t] for t in range(T)]
                z_d[n] = util.sample_categorical(cond)
                c[d, z_d[n]] += 1

    # Make assignments
    if theta_attr:
        for doc, c_d in zip(corpus.documents, c):
            doc.metadata[theta_attr] = c_d / c_d.sum()
    if z_attr:
        for doc, z_d in zip(corpus.documents, z):
            doc.metadata[z_attr] = z_d.tolist()


def mode(corpus, topics, theta_attr=None, z_attr=None, alpha=.01, max_iters=100, n=10):
    """Assigns topics using iterated conditional modes with uniform
    initialization and random restart.
    """
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    for doc in corpus.documents:
        best_z, best_c, best_p = _find_mode(doc, topics, max_iters, alpha)

        for _ in range(1, n):
            cand_z, cand_c, cand_p = _find_mode(doc, topics, max_iters, alpha)
            if cand_p > best_p:
                best_z, best_c, best_p = cand_z, cand_c, cand_p

        if theta_attr:
            doc.metadata[theta_attr] = best_c / best_c.sum()
        if z_attr:
            doc.metadata[z_attr] = best_z.tolist()


def _find_mode(doc, topics, max_iters, alpha):
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
            changed = changed or new != old

            z[n] = new
            c[new] += 1

        if not changed:
            break

    p = sum(np.log(alpha + topics[w_n.token, z_n]) for w_n, z_n in zip(doc.tokens, z))
    return z, c, p


def mode_word_init(corpus, topics, theta_attr=None, z_attr=None, max_iters=100):
    """Assigns topics using iterated conditional modes with per word
    maximization as the initialization.
    """
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')
    T = topics.shape[1]
    inits = np.argmax(topics, axis=1)

    for doc in corpus.documents:
        z = [inits[t.token] for t in doc.tokens]
        c = np.zeros(T)
        for z_n in z:
            c[z_n] += 1

        for _ in range(max_iters):
            changed = False
            for n, w_n in enumerate(doc.tokens):
                old = z[n]
                c[old] -= 1

                new = np.argmax([c[t] * topics[w_n.token, t] for t in range(T)])
                changed = changed or new != old

                z[n] = new
                c[new] += 1

            if not changed:
                break

        if theta_attr:
            doc.metadata[theta_attr] = c / c.sum()
        if z_attr:
            doc.metadata[z_attr] = z


