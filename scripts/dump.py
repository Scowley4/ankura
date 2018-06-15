import bisect
import csv
import itertools
import os
import pickle
import sys

import gensim as gs
import nltk
import numpy as np
import scipy.sparse
import sklearn.cluster
import tqdm

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
importr('copula')

import ankura

root_dir = sys.argv[1] if len(sys.argv) == 2 else 'scripts/'
try:
    os.mkdir(root_dir)
except:
    pass


def z(attr):
    return '{}_z'.format(attr)


def theta(attr):
    return '{}_theta'.format(attr)


def get_attr(corpus, model_name, K):
    return '{}_{}_{}'.format(corpus.metadata['name'], model_name, K)


def dump_stuff(attr, corpus, topics, n=1000, window_size=10):
    if attr is None:
        raise Exception('NONE?!')
    print('dumping', attr)

    with open(os.path.join(root_dir, '{}.topics.pickle'.format(attr)), 'wb') as f:
        pickle.dump(topics, f)
    with open(os.path.join(root_dir, '{}.corpus.pickle'.format(attr)), 'wb') as f:
        pickle.dump(corpus, f)
    f = open(os.path.join(root_dir, '{}.csv'.format(attr)), 'w')

    summary = ankura.topic.topic_summary(topics, corpus, n=11)
    highlighter = lambda w, _: '<u>{}</u>'.format(w)

    fields = ['text']
    fields.extend('label-{}'.format(i) for i in [1, 2, 3, 4, 5])
    fields.extend('value-{}'.format(i) for i in [1, 2, 3, 4, 5])
    fields.extend(['z', 'bad', 'model'])
    cf = csv.DictWriter(f, fieldnames=fields)
    cf.writeheader()

    for _ in range(n):
        doc = corpus.documents[np.random.choice(len(corpus.documents))]
        argsort = np.argsort(-doc.metadata[theta(attr)])
        good_options = argsort[:4]
        bad_options = argsort[4:] # argsort[-4:]

        n = np.random.choice(len(doc.tokens))
        token = doc.tokens[n]
        topic = doc.metadata[z(attr)][n]

        options = []
        if topic in good_options:
            options.extend(good_options)
        else:
            options.append(topic)
            options.extend(good_options[:3])
        options.append(np.random.choice(bad_options))

        token_start, token_end = token.loc
        if n - window_size <= 0:
            span_start = 0
            span_prefix = ''
        else:
            span_start = doc.tokens[n-window_size].loc[0]
            span_prefix = '<font color="grey">... </font>'
        if n + window_size >= len(doc.tokens):
            span_end = len(doc.text)
            span_postfix = ''
        else:
            span_end = doc.tokens[n+window_size].loc[1]
            span_postfix = '<font color="grey"> ...</font>'

        left = span_prefix + doc.text[span_start: token_start]
        center = highlighter(doc.text[token_start: token_end], topic)
        right = doc.text[token_end: span_end] + span_postfix
        row = {
            'text': left + center + right,
            'z': topic,
            'bad': options[-1],
            'model': attr,
        }

        token_text = corpus.vocabulary[token.token]
        mod_sum = [[w for w in topic if w != token_text][:10] for topic in summary]
        np.random.shuffle(options)
        row.update({'label-{}'.format(i): ('&lt;' + ', '.join(mod_sum[option]) + '&gt;') for i, option in enumerate(options, 1)})
        row.update({'value-{}'.format(i): option for i, option in enumerate(options, 1)})

        cf.writerow(row)

    f.close()
    print('dumped', attr)


def dump_lda(corpus, K, n=1000):
    attr = get_attr(corpus, 'lda', K)
    print('starting', attr)

    bows = ankura.topic._gensim_bows(corpus)
    lda = gs.models.LdaModel(
        corpus=bows,
        num_topics=K,
    )
    ankura.topic._gensim_assign(corpus, bows, lda, theta(attr), z(attr))
    topics = lda.get_topics().T

    dump_stuff(attr, corpus, topics, n)
    print('finished', attr)


def dump_anchor(corpus, K, n=1000):
    attr = get_attr(corpus, 'anchor', K)
    print('starting', attr)
    topics = ankura.anchor.anchor_algorithm(corpus, K)
    ankura.topic.lda_assign(corpus, topics, theta(attr), z(attr))
    dump_stuff(attr, corpus, topics, n)
    print('finished', attr)


class lda_gibbs_sampling_copula:

    def __init__(self, K=25, alpha=0.5, beta=0.5, copulaFamily="Frank", docs= None, V= None, copula_parameter=2):
        self.K = K
        self.copPar = copula_parameter
        self.family = copulaFamily
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs # a list of lists, each inner list contains the indexes of the words in a doc, e.g.: [[1,2,3],[2,3,5,8,7],[1, 5, 9, 10 ,2, 5]]
        self.V = V # how many different words in the vocabulary i.e., the number of the features of the corpus

        self.z_m_n = [] # topic assignements for each of the N words in the corpus. N: total number of words in the corpus (not the vocabulary size).
        self.n_m_z = np.zeros((len(self.docs), K), dtype=np.float64) + alpha     # |docs|xK topics: number of sentences assigned to topic z in document m
        self.n_z_t = np.zeros((K, V), dtype=np.float64) + beta # (K topics) x |V| : number of times a word v is assigned to a topic z
        self.n_z = np.zeros(K) + V * beta    # (K,) : overal number of words assigned to a topic z
        self.N = 0

        for m, doc in enumerate(docs):         # Initialization of the data structures I need.
            z_doc = []
            for sentence in doc:
                self.N += len(sentence)
                z_n = []
                for t in sentence:
                    z = np.random.randint(0, K) # Randomly assign a topic to a sentence. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                    z_n.append(z)                  # Keep track of the topic assigned
                    self.n_m_z[m, z] += 1          # increase the number of words assigned to topic z in the m doc.
                    self.n_z_t[z, t] += 1   #  .... number of times a word is assigned to this particular topic
                    self.n_z[z] += 1   # increase the counter of words assigned to z topic
                z_doc.append(z_n)
            self.z_m_n.append(np.array(z_doc)) # update the array that keeps track of the topic assignements in the sentences of the corpus.

    def inference(self):
        for m, doc in enumerate(self.docs):
            z_n, n_m_z = self.z_m_n[m], self.n_m_z[m] #Take the topics of the words and the number of words assigned to each topic
            for sid, sentence in enumerate(doc): #Sentence stands for a chunk, that is contiguous words that are generated by topics that are bound.
                # Get Sample from copula
                if len(sentence) > 1: # If the size of chunk is bigger than one, sample the copula, else back-off to standard LDA Gibbs sampling
                    command = "U = rnacopula(1, onacopula('%s', C(%s, 1:%d)))"%(self.family, self.copPar, len(sentence))
                    U = robjects.r(command)
                for n, t in enumerate(sentence): # Dicsount the counters to sample new topics
                    z = z_n[sid][n]
                    n_m_z[z] -= 1
                    self.n_z_t[z, t] -= 1
                    self.n_z[z] -= 1
                for n, t in enumerate(sentence):
                    p = (self.n_z_t[:, t]+self.beta) * (n_m_z+self.alpha) / (self.n_z + self.V*self.beta) #Update probability distributions
                    p = p / p.sum() # normalize the updated distributions
                    if len(sentence)>1: # Copula mechanism over the words of a chunk (noun-phrase or sentence)
                        new_z = self.getTopicIndexOfCopulaSample(p, U[n])
                    else:
                        new_z = np.random.multinomial(1, p).argmax() # Back-off to Gibbs sampling if len(sentence) == 1 for speed.
                    z_n[sid][n] = new_z
                    n_m_z[new_z] += 1
                    self.n_z_t[new_z, t] += 1
                    self.n_z[new_z] += 1

    def getTopicIndexOfCopulaSample(self, probs, sample):
        # Probability integral transform: given a uniform sample from the copula, use the quantile $F^{-1}$ to tranform it to a sample from f
        cdf = 0
        for key, val in enumerate(probs):
            cdf += val
            if sample <= cdf:
                return key


def dump_copula(corpus, K, n=1000):
    attr = get_attr(corpus, 'copula', K)
    print('starting', attr)

    cl_docs = []
    for doc in corpus.documents:
        bounds = np.cumsum([len(s) for s in nltk.tokenize.sent_tokenize(doc.text)])
        indices = [bisect.bisect_left(bounds, t.loc[0]) for t in doc.tokens]

        cl_doc = [[] for _ in range(len(bounds) + 1)]
        for i, t in zip(indices, doc.tokens):
            cl_doc[i].append(t.token)
        cl_docs.append([x for x in cl_doc if x])

    model = lda_gibbs_sampling_copula(K=K, docs=cl_docs, V=len(corpus.vocabulary))
    print('starting inference')
    for _ in tqdm.trange(200):
        model.inference()
    print('finished inference')

    for doc, z_d in zip(corpus.documents, model.z_m_n):
        doc.metadata[z(attr)] = sum(z_d.tolist(), [])
        theta_d = np.zeros(K)
        for z_d_n in z_d:
            theta_d[z_d_n] += 1
        theta_d /= theta_d.sum()
        doc.metadata[theta(attr)] = theta_d

    topics = model.n_z_t.T
    topics /= topics.sum(axis=0)

    dump_stuff(attr, corpus, topics)
    print('finished', attr)


def dump_cluster(corpus, K):
    attr = get_attr(corpus, 'cluster', K)
    print('starting', attr)

    M, V = len(corpus.documents), len(corpus.vocabulary)
    data = scipy.sparse.lil_matrix((M, V))
    for d, doc in enumerate(corpus.documents):
        for t in doc.tokens:
            data[d, t.token] += 1
    data = data.tocsc()

    kmeans = sklearn.cluster.KMeans(K)
    kmeans.fit_predict(data)
    topics = kmeans.cluster_centers_.T

    ankura.topic.lda_assign(corpus, topics, theta(attr), z(attr))
    dump_stuff(attr, corpus, topics, n)
    print('finished', attr)


importers = [
    ankura.corpus.newsgroups,
    ankura.corpus.amazon,
    # ankura.corpus.nyt,
]
# dumps = [
    # dump_cluster,
    # dump_lda,
    # dump_anchor,
    # dump_copula,
# ]
# Ks = [10, 20, 100]

# task_no = -1
# num_nodes = int(os.environ.get('PSSH_NUMNODES', '1'))
# node_num = int(os.environ.get('PSSH_NODENUM', '0'))
# print('node:', num_nodes, '/', num_nodes)

# for importer in importers:
    # for dump, K in itertools.product(dumps, Ks):
        # task_no += 1
        # print('considering:', task_no, str(importer), str(dump), str(K))
        # if task_no % num_nodes != node_num:
            # print('skipping:', task_no)
            # continue
        # print('performing:', task_no)
        # dump(importer(), K)

for importer in importers:
    dump_copula(importer(), 100, 20)
