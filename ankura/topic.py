"""Functions for using and displaying topics"""

import sys
import collections

import numpy as np
from scipy import spatial, sparse
import sklearn.naive_bayes

from . import pipeline, util, assign


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


@util.func_moved('ankura.assign.sampling')
def sampling_assign(corpus, topics, theta_attr=None, z_attr=None,
                    alpha=.01, num_iters=10):
    return assign.sampling(
        corpus=corpus, topics=topics,
        theta_attr=theta_attr, z_attr=z_attr,
        alpha=alpha, num_iters=num_iters)

@util.func_moved('ankura.assign.sklearn_variational')
def variational_assign(corpus, topics, theta_attr='theta', docwords_attr=None):
    return assign.sklearn_variational(
        corpus=corpus, topics=topics,
        theta_attr=theta_attr,
        docwords_attr=docwords_attr)

@util.func_moved('ankura.assign.gensim_variational')
def gensim_assign(corpus, topics, theta_attr=None, z_attr=None, needs_assign=None):
    return assign.gensim_variational(
        corpus=corpus, topics=topics,
        theta_attr=theta_attr, z_attr=z_attr,
        needs_assign=needs_assign)


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

def jeff_classifier(train, topics, label_attr='label', z_attr='z', alpha=.01):
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


def pdists(corpus, theta_attr, metric='cosine'):
    D = len(corpus.documents)
    thetas = np.array([doc.metadata[theta_attr] for doc in corpus.documents])
    dists = spatial.distance.pdist(thetas[:D], metric)
    for ij in np.argsort(dists, axis=None):
        i, j = ij // D, ij % D
        if i == j:
            continue
        yield i, j


def cross_reference(corpus, attr, doc=None, n=sys.maxsize, threshold=1):
    """Finds the nearest documents by topic similarity.

    The documents of the corpus must include a metadata value giving a vector
    representation of the document. Typically, this is a topic distribution
    obtained with an assign function and a metadata_attr. The vector
    representation is then used to compute distances between documents.

    If a document is given, then a list of references is returned for that
    document. Otherwise, cross references for each document in a corpus are
    given in a dict keyed by the documents. Consequently, the documents of the
    corpus must be hashable.

    The closest n documents will be returned (default=sys.maxsize). Documents
    whose similarity is behond the threshold (default=1) will not be returned.
    A threshold of 1 indicates that no filtering should be done, while a 0
    indicates that only exact matches should be returned.
    """
    def _xrefs(doc):
        doc_theta = doc.metadata[attr]
        dists = [spatial.distance.cosine(doc_theta, d.metadata[attr])
                 if doc is not d else float('nan')
                 for d in corpus.documents]
        dists = np.array(dists)
        return list(corpus.documents[i] for i in dists.argsort()[:n]
                    if dists[i] <= threshold)

    if doc:
        return _xrefs(doc)
    else:
        return [_xrefs(doc) for doc in corpus.documents]


def free_classifier(topics, Q, labels, epsilon=1e-7):
    """Creates a topic-based linear classifier. Details forthcoming..."""
    K = len(labels)
    V = Q.shape[0] - K

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # class_given_word
    Q = Q / Q.sum(axis=1, keepdims=True) # row-normalize Q without original
    Q_L = Q[-K:, :V]

    def _classifier(doc, attr='theta'):
        """The document classifier returned by free_classifier"""
        H = np.zeros(V)
        for w_d in doc.tokens:
            H[w_d.token] += 1

        topic_score = A_f.dot(doc.metadata[attr])
        topic_score /= topic_score.sum(axis=0)

        word_score = Q_L.dot(H)
        word_score /= word_score.sum(axis=0)

        return labels[np.argmax(topic_score + word_score)]
    return _classifier

def free_classifier_derpy(topics, Q, labels, epsilon=1e-7):
    """same as function above, with a few minor math fixes"""
    K = len(labels)
    V = Q.shape[0] - K

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # class_given_word
    Q = Q / Q.sum(axis=1, keepdims=True) # row-normalize Q without original
    Q_L = Q[:V, -K:]

    def _classifier(doc, attr='theta'):
        """The document classifier returned by free_classifier_derpy"""
        topic_score = A_f.dot(doc.metadata[attr])
        topic_score /= topic_score.sum(axis=0)

        return labels[np.argmax(topic_score)]
    return _classifier

def free_classifier_revised(topics, Q, labels, epsilon=1e-7):
    """same as function above, with a few minor math fixes"""
    K = len(labels)
    V = Q.shape[0] - K

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # class_given_word
    Q = Q / Q.sum(axis=1, keepdims=True) # row-normalize Q without original
    Q_L = Q[:V, -K:]

    def _classifier(doc, attr='theta'):
        """The document classifier returned by free_classifier_revised"""
        H = np.zeros(V)
        for w_d in doc.tokens:
            H[w_d.token] += 1

        # normalize H
        H = H / H.sum(axis=0)

        topic_score = A_f.dot(doc.metadata[attr])
        topic_score /= topic_score.sum(axis=0)

        word_score = H.dot(Q_L)
        word_score /= word_score.sum(axis=0)

        return labels[np.argmax(topic_score + word_score)]
    return _classifier


def free_classifier_line_not_gibbs(corpus, attr_name, labeled_docs,
                            topics, C, labels, epsilon=1e-7):

    K = len(labels)

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # column normalize topic-label matrix
    C_f = C[0:, -K:]
    C_f /= C_f.sum(axis=0)

    L = np.zeros(K)
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_name = doc.metadata[attr_name]
            i = labels.index(label_name)
            L[i] += 1

    L = L / L.sum(axis=0) # normalize L to get the label probabilities

    def _classifier(doc, attr='z'):
        final_score = np.zeros(K)
        for i, l in enumerate(L):
            product = l
            doc_topic_count = collections.Counter(doc.metadata[attr])
            for topic, count in doc_topic_count.items():
                product *= C_f[topic, i]**count

            final_score[i] = product

        return labels[np.argmax(final_score)]
    return _classifier


def free_classifier_dream(corpus, attr_name, labeled_docs,
                          topics, C, labels, epsilon=1e-7,
                          prior_attr_name=None):
    L = len(labels)

    # column-normalized word-topic matrix without labels
    A_w = topics[:-L]
    A_w /= A_w.sum(axis=0)

    _, K = A_w.shape # K is number of topics

    # column normalize topic-label matrix
    C_f = C[:, -L:]
    C_f /= C_f.sum(axis=0)

    lambda_ = corpus.metadata.get(prior_attr_name) # emperically observed labels
    if lambda_ is None:
        lambda_ = np.zeros(L)
        for d, doc in enumerate(corpus.documents):
            if d in labeled_docs:
                label_name = doc.metadata[attr_name];
                i = labels.index(label_name)
                lambda_[i] += 1
        lambda_ = lambda_ / lambda_.sum(axis=0) # normalize lambda_ to get the label probabilities
        if prior_attr_name:
            corpus.metadata[prior_attr_name] = lambda_

    log_lambda = np.log(lambda_)

    def _classifier(doc, get_probabilities=False, get_log_probabilities=False):
        """The document classifier returned by free_classifier_dream

        By default, returns the label name for the predicted label.

        If get_probabilities is True, returns the probabilities of each label
        instead of the label name.
        """
        results = np.copy(log_lambda)
        token_counter = collections.Counter(tok.token for tok in doc.tokens)
        for l in range(L):
            for w_i in token_counter:
                m = token_counter[w_i] * np.sum(C_f[:, l] * A_w[w_i, :])
                if m != 0: # this gets rid of log(0) warning, but essentially does the same thing as taking log(0)
                    results[l] += np.log(m)
                else:
                    results[l] = float('-inf')

        if get_probabilities:
            return np.exp(results)
        if get_log_probabilities:
            return results
        return labels[np.argmax(results)]
    return _classifier


def free_classifier_line_model(corpus, attr_name, labeled_docs,
                                    topics, C, labels, epsilon=1e-7, num_iters=10):

    L = len(labels)

    # column-normalized word-topic matrix without labels
    A = topics[:-L]
    A /= A.sum(axis=0)

    _, K = A.shape # K is number of topics

    # column normalize topic-label matrix
    C_f = C[0:, -L:]
    C_f /= C_f.sum(axis=0)

    lambda_ = np.zeros(L) # emperically observe labels
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_name = doc.metadata[attr_name];
            i = labels.index(label_name)
            lambda_[i] += 1
    lambda_ = lambda_ / lambda_.sum(axis=0) # normalize lambda_ to get the label probabilities

    def _classifier(doc):
        l = np.random.randint(L)
        z = np.random.randint(K, size=len(doc.tokens))

        for _ in range(num_iters):
            doc_topic_count = collections.Counter(z) # maps topic assignments to counts (this used to be outside of the for loop)
            l_cond = np.log(lambda_) # not in log space: cond = lambda_
            for s in range(L):
                for topic, count in doc_topic_count.items():
                    l_cond[s] += count * np.log(C_f[topic, s]) # not in log space: cond[s] *= C_f[topic, s]**count
            l = util.sample_log_categorical(l_cond)

            for n, w_n in enumerate(doc.tokens):
                doc_topic_count[z[n]] -= 1
                z_cond = C_f[:K,l] * A[w_n.token,:K] # z_cond = [C_f[t, l] * A[w_n.token, t] for t in range(K)] # eq 2
                z[n] = util.sample_categorical(z_cond)
                doc_topic_count[z[n]] += 1

        return labels[l]
    return _classifier


def free_classifier_v_model(corpus, attr_name, labeled_docs,
                                    topics, labels, epsilon=1e-7, num_iters=100):

    L = len(labels)

    # column-normalized word-topic matrix without labels
    A = topics[:-L]
    A /= A.sum(axis=0)

    _, K = A.shape # K is number of topics

    # Smooth and column normalize class-topic weights
    A_f = topics[-L:] + epsilon
    A_f /= A_f.sum(axis=0)

    def _classifier(doc):
        l = np.random.randint(L)
        z = np.random.randint(K, size=len(doc.tokens))

        for _ in range(num_iters):
            doc_topic_count = collections.Counter(z) # maps topic assignments to counts
            l_cond = [sum(A_f[x, topic]*count for topic, count in doc_topic_count.items()) for x in range(L)]

            l = util.sample_categorical(l_cond)
            B = l_cond[l] # B is a constant (summation of A_f[l, z_i])

            for n, w_n in enumerate(doc.tokens):
                B -= A_f[l, z[n]]
                z_cond = [A[w_n.token, t] * (A_f[l, t] + B) for t in range(K)]
                z[n] = util.sample_categorical(z_cond)
                B += A_f[l, z[n]]

        return labels[l]
    return _classifier
