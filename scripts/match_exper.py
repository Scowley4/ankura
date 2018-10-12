import collections
import sys
import csv
import glob
import itertools
import os
import pickle

import ankura
import psshlib

writer = csv.DictWriter(sys.stdout, ['eval','tp','fp', 'tn', 'fn'])
writer.writeheader()

def eval_xref(bible, gold_attr, pred_attr):
    tp = 0
    fp = 0
    fn = 0

    for doc in bible.documents:
        gold = set(doc.metadata[gold_attr])
        pred = set(doc.metadata[pred_attr])

        tp += len(pred.intersection(gold))
        fp += len(pred.difference(gold))
        fn += len(gold.difference(pred))

    D = len(bible.documents)
    tn = D * (D - 1) - tp - fp - fn
    return tp, fp, tn, fn


fnames = glob.glob('/users/scratch/jlund3/bibles/*.pickle')
golds = ['tske', 'obib'] + ['obib{}'.format(i) for i in range(11)]

for fname in psshlib.pardo(fnames):
    name = os.path.basename(fname)[:-7]
    z_attr = name + '_z'
    anchors, topics, bible = pickle.load(open(fname, 'rb'))

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

    w_xref_attr = name + '_w_attr'
    for d, wset in w_refs.items():
        bible.documents[d].metadata[w_xref_attr] = [bible.documents[i].metadata['verse'] for i in wset]
    z_xref_attr = name + '_z_attr'
    for d, zset in z_refs.items():
        bible.documents[d].metadata[z_xref_attr] = [bible.documents[i].metadata['verse'] for i in zset]
    wz_xref_attr = name + '_wz_attr'
    for d, wzset in wz_refs.items():
        bible.documents[d].metadata[wz_xref_attr] = [bible.documents[i].metadata['verse'] for i in wzset]

    for gold in golds:
        for x_attr in [w_xref_attr, z_xref_attr, wz_xref_attr]:
            tp, fp, tn, fn = eval_xref(bible, 'xref-{}'.format(gold), x_attr)
            writer.writerow({
                'eval': '{} {}'.format(x_attr, gold),
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
            })
