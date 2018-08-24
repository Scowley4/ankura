import os
import sys
import csv
import itertools

import numpy as np

import ankura

task_no = -1
node_num = int(os.environ.get('PSSH_NODENUM', '0'))
num_nodes = int(os.environ.get('PSSH_NUMNODES', '1'))

# bible = ankura.corpus.bible(remove_empty=True)
bible = ankura.corpus.full_bible_stemmed()
sample = np.random.choice(len(bible.documents), size=1000, replace=False)

writer = csv.DictWriter(sys.stdout, [
    'algo',
    'k',
    'doct',
    'thet',
    'f1',
    'precision',
    'recall',
])
writer.writeheader()


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


Ks = [2000, 3000, 4000]
DocTs = [10, 5]
TheTs = [.1, .2, .3, .4, .5]
algos = [
    ('vari', ankura.topic.variational_assign),
    ('mod2', ankura.topic.mode_assign2),
    ('modi', ankura.topic.mode_init_assign),
]

for k, doc_t, theta_t, (algo, assign) in itertools.product(Ks, DocTs, TheTs, algos):
    task_no += 1
    if task_no % num_nodes != node_num:
        continue

    topics = ankura.anchor.anchor_algorithm(bible, k, doc_t)
    theta_attr = '{}_{}_{}_theta'.format(algo, k, doc_t)
    assign(bible, topics, theta_attr)

    xref_attr  = '{}_{}_{}_xrefs{}'.format(algo, k, doc_t, theta_t)
    ankura.topic.cross_reference(bible, theta_attr, xref_attr, 'verse', threshold=theta_t, doc_ids=sample)
    tp, fp, tn, fn = eval_xref(xref_attr)

    try:
        writer.writerow({
            'algo': algo,
            'k': int(k),
            'doct': doc_t,
            'thet': theta_t,
            'f1': 2 * tp / (2 * tp + fn + fp),
            'precision': tp / (tp+fp),
            'recall': tp / (tp+fn),
        })
    except:
        writer.writerow({
            'algo': algo,
            'k': int(k),
            'doct': doc_t,
            'thet': theta_t,
        })
