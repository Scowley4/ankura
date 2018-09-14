import ankura
import csv
import itertools
import sys
import os

import numpy as np

import ankura

task_no = -1
node_num = int(os.environ.get('PSSH_NODENUM', '0'))
num_nodes = int(os.environ.get('PSSH_NUMNODES', '1'))

corpora = [
    (ankura.corpus.newsgroups, 40, 'newsgroup'),
    (ankura.corpus.newsgroups, 4000, 'newsgroup'),
    (ankura.corpus.amazon, 20, 'binary_rating'),
    (ankura.corpus.amazon, 2000, 'binary_rating'),
    (ankura.corpus.nyt, 10, None),
    (ankura.corpus.nyt, 1000, None),
]
algos = [
    ('vari', ankura.topic.variational_assign),
    ('gibb', ankura.topic.sampling_assign),
    ('icmr', ankura.topic.mode_assign2),
    ('icmw', ankura.topic.mode_init_assign),
    # ('icms', ankura.topic.mode_init_assign),
]

writer = csv.DictWriter(sys.stdout, [
    'algo',
    'k',
    'accuracy',
    'coherence',
    'sigwuni',
    'sigwvac',
    'sigdbak',
    'switchp',
])
writer.writeheader()

for (data, k, label), (algo, assign) in itertools.product(corpora, algos):
    task_no += 1
    if task_no % num_nodes != node_num:
        continue

    z_attr = '{}_{}_z'.format(algo, k)
    t_attr = '{}_{}_t'.format(algo, k)

    corpus = data()
    train, test = ankura.pipeline.train_test_split(corpus)
    Q = ankura.anchor.build_cooccurrence(corpus)
    anchors = ankura.anchor.gram_schmidt_anchors(corpus, Q, k, 500 if k < 1000 else 5)
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
        'algo': algo,
        'k': k,
        'accuracy': contingency.accuracy() if label else 0,
        'coherence': ankura.validate.coherence(corpus, summary),
        'sigwuni': np.mean([ankura.validate.significance_wuni(t) for t in topic_cols]),
        'sigwvac': np.mean([ankura.validate.significance_wvac(t, Q) for t in topic_cols]),
        'sigdbak': np.mean([ankura.validate.significance_dback(t, corpus, t_attr) for t in range(k)]),
        'switchp': ankura.validate.topic_switch_percent(corpus, z_attr),
    })
