import ankura
import csv
import functools
import itertools
import os
import sys

import numpy as np

import ankura

task_no = -1
node_num = int(os.environ.get('PSSH_NODENUM', '0'))
num_nodes = int(os.environ.get('PSSH_NUMNODES', '1'))

np.random.seed(314159268)
seeds = [int(np.random.random() * (2*32-1)) for _ in range(1)]

corpora = [
    (lambda : ankura.corpus.newsgroups(), 40, 'newsgroup'),
    (lambda : ankura.corpus.newsgroups(15, None), 2000, 'newsgroup'),
    (lambda : ankura.corpus.amazon(), 20, 'binary_rating'),
    (lambda : ankura.corpus.amazon(15), 1000, 'binary_rating'),
    (lambda : ankura.corpus.nyt(), 10, None),
    (lambda : ankura.corpus.nyt(15), 500, None),
]
algos = [
    ('vari', ankura.assign.variational),
    ('gibb', ankura.assign.sampling),
    ('icms', functools.partial(ankura.assign.mode, n=1)),
    ('icmr', ankura.assign.mode),
    ('icmw', ankura.assign.mode_word_init),
]

writer = csv.DictWriter(sys.stdout, [
    'dataset',
    'algo',
    'k',
    'accuracy',
    'coherence',
    'consistency',
    'sigwuni',
    'sigwvac',
    'sigdbak',
])
writer.writeheader()

for seed, (data, k, label), (algo, assign) in itertools.product(seeds, corpora, algos):
    task_no += 1
    if task_no % num_nodes != node_num:
        continue

    np.random.seed(seed)

    z_attr = '{}_{}_z'.format(algo, k)
    t_attr = '{}_{}_t'.format(algo, k)

    corpus = data()
    train, test = ankura.pipeline.train_test_split(corpus)
    Q = ankura.anchor.build_cooccurrence(corpus)
    anchors = ankura.anchor.gram_schmidt_anchors(corpus, Q, k, 500 if k < 1000 else 10)
    topics = ankura.anchor.recover_topics(Q, anchors)

    topic_cols = [topics[:, t] for t in range(k)]
    summary = ankura.topic.topic_summary(topics)
    assign(corpus, topics, t_attr, z_attr)

    if label:
        classifier = ankura.topic.classifier(train, topics, label, z_attr)
        gold = [doc.metadata[label] for doc in test.documents]
        pred = classifier(test)
        contingency = ankura.validate.Contingency()
        for gp in zip(gold, pred):
            contingency[gp] += 1

    writer.writerow({
        'dataset': corpus.metadata['name'],
        'algo': algo,
        'k': k,
        'accuracy': contingency.accuracy() if label else 0,
        'coherence': ankura.validate.coherence(corpus, summary),
        'consistency': ankura.validate.consistency(corpus, z_attr),
        'sigwuni': np.mean([ankura.validate.significance_wuni(t) for t in topic_cols]),
        'sigwvac': np.mean([ankura.validate.significance_wvac(t, Q) for t in topic_cols]),
        'sigdbak': np.mean([ankura.validate.significance_dback(t, corpus, t_attr) for t in range(k)]),
    })
