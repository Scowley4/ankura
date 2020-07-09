"""Functions for assigning documents and words to topics.

Generally these assign functions all share a basic signature with four
arguments: a corpus, a set of topics, and either a theta_attr or a z_attr (both
can be specfied as well). Depending on which attribute name is given, the
documents or words are assigned to topics, and the changes are reflected in the
Corpus object as metadata attributes on the constituent Document objects.
Beyond these first four parameters, there may be additional optional parameters
which further specify how the assignments should be made.
"""

import collections

import gensim as gs
import numpy as np

try:
    from . import util
except:
    import util


def sklearn_variational(corpus, topics, theta_attr='theta', docwords_attr=None):
    """Predicts topic assignments for a corpus.

    Topic inference is done using online variational inference with Latent
    Dirichlet Allocation and fixed topics following Hoffman et al., 2010. Each
    document is given a metadata value named by theta_attr corresponding to the
    its predicted topic distribution.

    If docwords_attr is given, then the corpus metadata with that name is
    assumed to contain a pre-computed sparse docwords matrix. Otherwise, this
    docwords matrix will be recomputed.
    """
    V, K = topics.shape
    if docwords_attr:
        docwords = corpus.metadata[docwords_attr]
        if docwords.shape[1] != V:
            raise ValueError('Mismatch between topics and docwords shape')
    else:
        docwords = pipeline.build_docwords(corpus, V)

    lda = sklearn.decomposition.LatentDirichletAllocation(K)
    lda.components_ = topics.T
    lda._check_params()
    lda._init_latent_vars(V)
    theta = lda.transform(docwords)

    for doc, theta_d in zip(corpus.documents, theta):
        doc.metadata[theta_attr] = theta_d


def gensim_variational(corpus, topics, theta_attr=None, z_attr=None, needs_assign=None):
    """Assigns topics using variational inference through gensim."""
    if not theta_attr and not z_attr:
        raise ValueError('Either theta_attr or z_attr must be given')

    # Convert corpus to gensim bag-of-words format
    bows = [list(collections.Counter(tok.token for tok in doc.tokens).items())
                for d, doc in enumerate(corpus.documents)
                if needs_assign is None or d in needs_assign]

    # Build lda with fixed topics
    V, K = topics.shape
    lda = gs.models.LdaModel(
        num_topics=K,
        id2word={i: i for i in range(V)}, # LdaModel gets V from this dict
    )
    # This was originally in byu-aml ankura master
    # lda.state.sstats = topics.astype(lda.dtype).T

    # This line was taken from jefflund ankura
    lda.state.sstats = topics.astype(lda.dtype).T * len(corpus.documents)
    lda.sync_state()

    # Make topic assignments
    if needs_assign is None:
        docs = corpus.documents
    else:
        docs = (corpus.documents[i] for i in needs_assign)

    for d, (doc, bow) in enumerate(zip(docs, bows)):
        gamma, phi = lda.inference([bow], collect_sstats=z_attr)
        if theta_attr:
            doc.metadata[theta_attr] = gamma[0] / gamma[0].sum()
        if z_attr:
            w = [t.token for t in doc.tokens]
            doc.metadata[z_attr] = phi.argmax(axis=0)[w].tolist()


def sampling(corpus, topics, theta_attr=None, z_attr=None, alpha=.01, num_iters=10):
    """Predicts topic assignments for a corpus using Gibbs sampling.

    Topic inference is done using Gibbs sampling with Latent Dirichlet
    Allocation and fixed topics following Griffiths and Steyvers 2004. The
    parameter alpha specifies a symetric Dirichlet prior over the document
    topic distributions. The parameter num_iters controlls how many iterations
    of sampling should be performed.

    If theta_attr is given, each document is given a metadata value describing
    the document-topic distribution as an array. If the z_attr is given, each
    document is given a metadata value describing the token level topic
    assignments. At east one of the attribute names must be given.

    """
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
                cond = (alpha + c[d, :]) * topics[w_dn.token, :]
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
