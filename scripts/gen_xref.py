import glob
import sys
import os
import pickle

import numpy as np

import ankura
import psshlib

dists = [
    'cosine',
    'euclidean',
    'chebyshev',
    'cityblock',
    'braycurtis',
]
fnames = glob.glob('/users/scratch/jlund3/bibles/*.pickle')

for metric, fname in psshlib.pproduct(dists, fnames):
    name = os.path.basename(fname)[:-7]
    bible = pickle.load(open(fname, 'rb'))
    verse = lambda d: bible.documents[d].metadata['verse']
    theta_attr = '{}_theta'.format(name)

    with open('aml/scratch/xrefs/{}_{}.txt'.format(name, metric), 'w') as f:
        for i, j in ankura.topic.pdists(bible, theta_attr, metric):
            f.write('{} {}\n'.format(verse(i), verse(j))
