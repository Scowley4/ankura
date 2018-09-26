import os
import sys
import csv
import itertools

import numpy as np

import ankura

task_no = -1
node_num = int(os.environ.get('PSSH_NODENUM', '0'))
num_nodes = int(os.environ.get('PSSH_NUMNODES', '1'))

bible = ankura.corpus.bible(remove_stopwords=False, remove_empty=False, use_stemmer=True)
Q = ankura.anchor.build_cooccurrence(bible)

np.random.seed(314159286)
seeds = [int(np.random.random() * (2**32-1)) for _ in range(1)]

writer = csv.DictWriter(sys.stdout, [
    'algo',
    'k',
    'thet',
    'f1',
    'precision',
    'recall',
    'specificity',
    'consistency',
    'significance',
])
writer.writeheader()


def gs_anchors(k):
    return ankura.anchor.gram_schmidt_anchors(bible, Q, k, doc_threshold=5)


def rp_anchors(k):
    anchor_docs = np.random.choice(len(bible.documents), size=k, replace=False)
    anchor_indices = [[t.token for t in bible.documents[d].tokens] for d in anchor_docs]
    return ankura.anchor.tandem_anchors(anchor_indices, Q)


def eval_xref(xref_attr):
    tp = 0
    fp = 0
    fn = 0

    for d in sample:
        verse = bible.documents[d]
        gold = set(verse.metadata['xref'])
        pred = set(verse.metadata[xref_attr])

        tp += len(pred.intersection(gold))
        fp += len(pred.difference(gold))
        fn += len(gold.difference(pred))

    tn = len(sample) * (len(bible.documents) - 1) - tp - fp - fn
    return tp, fp, tn, fn


Ks = [1000, 2000, 3000, 4000]
ThetaTs = [.1, .2, .5]
algos = [
    ('vari', ankura.assign.variational),
    ('icmw', ankura.assign.mode_word_init),
]

for seed, k, theta_t, (algo, assign) in itertools.product(seeds, Ks, ThetaTs, algos):
    task_no += 1
    if task_no % num_nodes != node_num:
        continue

    np.random.seed(seed)
    sample = np.random.choice(len(bible.documents), size=1000, replace=False)

    # anchors = ankura.anchor.gram_schmidt_anchors(bible, Q, k, doc_threshold=5)
    anchors = rp_anchors(k)
    topics = ankura.anchor.recover_topics(Q, anchors)

    theta_attr = '{}_{}_theta'.format(algo, k)
    z_attr = '{}_{}_z'.format(algo, k)
    assign(bible, topics, theta_attr, z_attr)

    xref_attr  = '{}_{}_xrefs{}'.format(algo, k, theta_t)
    ankura.topic.cross_reference(bible, theta_attr, xref_attr, 'verse', threshold=theta_t, doc_ids=sample)
    tp, fp, tn, fn = eval_xref(xref_attr)

    writer.writerow({
        'algo': algo,
        'k': k,
        'thet': theta_t,

        'f1': 2 * tp / (2 * tp + fn + fp),
        'precision': tp / (tp+fp),
        'recall': tp / (tp+fn),
        'specificity': tn / (tn+fp),

        'consistency': ankura.validate.consistency(bible, z_attr),
        'significance': np.mean([ankura.validate.significance_dback(t, bible, theta_attr) for t in range(k)]),
    })
