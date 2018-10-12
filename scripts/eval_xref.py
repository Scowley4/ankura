import glob
import sys
import csv
import itertools
import os
import pickle

import numpy as np
import scipy.spatial

import ankura

task_no = -1
node_num = int(os.environ.get('PSSH_NODENUM', '0'))
num_nodes = int(os.environ.get('PSSH_NUMNODES', '1'))

D = len(ankura.corpus.bible(remove_stopwords=False, remove_empty=False, use_stemmer=True).documents)
sample = np.random.choice(D, size=D, replace=False)
golds = ['tske', 'obib'] + ['obib{}'.format(i) for i in range(11)]

writer = csv.DictWriter(sys.stdout, ['eval','tp','fp', 'tn', 'fn', 'tpr', 'tnr', 'ppv', 'acc', 'f1', 'fpr'])
writer.writeheader()

def eval_xref(bible, gold_attr, pred_attr):
    tp = 0
    fp = 0
    fn = 0

    for d in sample:
        verse = bible.documents[d]
        gold = set(verse.metadata[gold_attr])
        pred = set(verse.metadata[pred_attr])

        tp += len(pred.intersection(gold))
        fp += len(pred.difference(gold))
        fn += len(gold.difference(pred))

    tn = len(sample) * (len(bible.documents) - 1) - tp - fp - fn
    return tp, fp, tn, fn

ts = [.91, .92, .93, .94, .95, .96, .97, .98, .99]
fnames = glob.glob('/users/scratch/jlund3/bibles/*.pickle')
distances = [
    ('cosine', False),
    # ('euclidean', False),
    # ('sqeuclidean', False),
    # ('chebyshev', False),
]

for t, fname, (dist, booleanize) in itertools.product(ts, fnames, distances):
    task_no += 1
    if task_no % num_nodes != node_num:
        continue

    name = os.path.basename(fname)[:-7]
    t_attr = name + '_theta'

    anchors, topics, bible = pickle.load(open(fname, 'rb'))

    x_attr = '{}_{}_{}_xref'.format(name, t, dist)
    ankura.topic.cross_reference(bible, t_attr, x_attr, sample, 'verse',
            threshold=t, distance=dist, booleanize=booleanize)

    for gold in golds:
        tp, fp, tn, fn = eval_xref(bible, 'xref-{}'.format(gold), x_attr)
        writer.writerow({
            'eval': '{} {}'.format(x_attr, gold),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'tpr': tp/(tp+fn),
            'tnr': tn/(tn+fp),
            'ppv': tp/(tp+fp),
            'acc': (tp+tn)/(tp+tn+fp+fn),
            'f1': 2*tp/(2*tp+tp+fn),
            'fpr': fp/(fp+tn)
        })
