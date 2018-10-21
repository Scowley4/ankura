import collections
import sys
import csv
import glob
import itertools
import os
import pickle

import ankura
import psshlib


def verse(bible, d):
    return bible.documents[d].metadata['verse']


def eval_xref(bible, gold_attr, pred_refs):
    tp = 0
    fp = 0
    fn = 0

    for d, doc in enumerate(bible.documents):
        gold = set(doc.metadata[gold_attr])
        pred = {verse(bible, x) for x in pred_refs[d]}

        tp += len(pred.intersection(gold))
        fp += len(pred.difference(gold))
        fn += len(gold.difference(pred))

    D = len(bible.documents)
    tn = D * (D - 1) - tp - fp - fn
    return tp, fp, tn, fn


fnames = glob.glob('/users/scratch/jlund3/bibles/*.pickle')
golds = ['tske', 'obib0', 'obib5']

for fname in psshlib.pardo(fnames):
    name = os.path.basename(fname)[:-7]
    z_attr = name + '_z'
    bible = pickle.load(open(fname, 'rb'))

    w_index= collections.defaultdict(set)
    z_index = collections.defaultdict(set)

    for d, doc in enumerate(bible.documents):
        for t in doc.tokens:
            w_index[t.token].add(d)
        for z in doc.metadata[z_attr]:
            z_index[z].add(d)

    w_refs = collections.defaultdict(set)
    for wset in w_index.values():
        for d in wset:
            w_refs[d].update(wset)
    z_refs = collections.defaultdict(set)
    for zset in z_index.values():
        for d in zset:
            z_refs[d].update(zset)
    wz_refs = {}
    for d, wset in w_refs.items():
        wz_refs[d] = wset.intersection(z_refs[d])

    metrics = [('wordmatch', w_refs), ('topicmatch', z_refs), ('topicwordmatch', wz_refs)]

    for gold, (metric, refs) in itertools.product(golds, metrics):
        tp, fp, tn, fn = eval_xref(bible, 'xref-'+gold, refs)
        with open('evals/{}_{}_{}.csv'.format(name, metric, gold), 'w') as f:
            writer = csv.DictWriter(f, ['tp', 'fp', 'tn', 'fn'])
            writer.writeheader()
            writer.writerow({
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
            })
        print(name, metric, gold)
