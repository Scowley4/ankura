import glob
import sys
import os
import pickle

from scipy.spatial import distance
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
    D = len(bible.documents)
    verse = lambda d: bible.documents[d].metadata['verse']
    theta_attr = '{}_theta'.format(name)

    thetas = np.array([doc.metadata[theta_attr] for doc in bible.documents])
    dists = distance.squareform(distance.pdist(thetas, metric))

    with open('aml/scratch/xrefs/{}_{}.txt'.format(name, metric), 'w') as f:
        for ij in np.argsort(dists, axis=None):
            i, j = ij // D, ij % D
            if i == j:
                continue
            f.write('{} {}\n'.format(verse(i), verse(j)))
