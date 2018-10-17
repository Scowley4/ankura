import csv
import itertools
import os
import pickle
import sys

import numpy as np

import ankura
import psshlib

bible = ankura.corpus.bible(remove_stopwords=True, remove_empty=False, use_stemmer=True)
Q = ankura.anchor.build_cooccurrence(bible)


def gs_anchors(k):
    return ankura.anchor.gram_schmidt_anchors(bible, Q, k, doc_threshold=5)


def rp_anchors(k):
    anchor_docs = np.random.choice(len(bible.documents), size=k, replace=False)
    anchor_indices = [[t.token for t in bible.documents[d].tokens] for d in anchor_docs]
    return ankura.anchor.tandem_anchors(anchor_indices, Q)


infers = [
    ('tandem1', rp_anchors, 3000),
    ('tandem2', rp_anchors, 3000),
    ('tandem3', rp_anchors, 3000),
    ('gramschmidt', gs_anchors, 3000),
    ('coarse', gs_anchors, 300),
]

for (name, anchors, k) in psshlib.pardo(infers):
    topics = ankura.anchor.recover_topics(Q, anchors(k))
    theta_attr = '{}_theta'.format(name)
    z_attr = '{}_z'.format(name)
    ankura.assign.variational(bible, topics, theta_attr, z_attr)
    pickle.dump(bible, open('aml/scratch/bibles/{}.pickle'.format(name), 'wb'))
