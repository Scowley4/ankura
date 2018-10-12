import csv
import itertools
import os
import pickle
import sys

import numpy as np

import ankura

task_no = -1
node_num = int(os.environ.get('PSSH_NODENUM', '0'))
num_nodes = int(os.environ.get('PSSH_NUMNODES', '1'))

bible = ankura.corpus.bible(remove_stopwords=True, remove_empty=False, use_stemmer=True)
bible.metadata['xrefs'] = []
Q = ankura.anchor.build_cooccurrence(bible)


def gs_anchors(k):
    return ankura.anchor.gram_schmidt_anchors(bible, Q, k, doc_threshold=5)


def rp_anchors(k):
    anchor_docs = np.random.choice(len(bible.documents), size=k, replace=False)
    anchor_indices = [[t.token for t in bible.documents[d].tokens] for d in anchor_docs]
    return anchor_indices, ankura.anchor.tandem_anchors(anchor_indices, Q)


Ks = [1000, 2000, 3000]
algos = [
    ('vari', ankura.assign.variational),
    ('icmw', ankura.assign.mode_word_init),
]

for k, (algo, assign) in itertools.product(Ks, algos):
    task_no += 1
    if task_no % num_nodes != node_num:
        continue

    # anchors = ankura.anchor.gram_schmidt_anchors(bible, Q, k, doc_threshold=5)
    anchor_indices, anchors = rp_anchors(k)
    topics = ankura.anchor.recover_topics(Q, anchors)

    theta_attr = '{}_{}_theta'.format(algo, k)
    z_attr = '{}_{}_z'.format(algo, k)
    assign(bible, topics, theta_attr, z_attr)

    pickle.dump((anchor_indices, topics, bible), open('aml/scratch/bibles/{}_{}.pickle'.format(algo, k), 'wb'))
