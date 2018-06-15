import csv
import collections
import pickle
import glob
import os
import sys

import ankura

import matplotlib.pyplot as plt


def z(attr):
    return '{}_z'.format(attr)


def theta(attr):
    return '{}_theta'.format(attr)


root_dir = sys.argv[1] if len(sys.argv) == 2 else 'scripts/'

paths = {path.split('.')[0] for path in glob.glob(os.path.join(root_dir, '*'))}

f = csv.writer(open(os.path.join(root_dir, 'results.csv'), 'w'))
f.writerow(['corpus', 'algorithm', 'k', 'switchp', 'switchv', 'worddiv', 'windowp'])
for path in paths:
    attr = os.path.basename(path)
    topics = pickle.load(open('{}.topics.pickle'.format(path), 'rb'))
    corpus = pickle.load(open('{}.corpus.pickle'.format(path), 'rb'))
    f.writerow(attr.split('_') + [
        ankura.validate.topic_switch_percent(corpus, z(attr)),
        ankura.validate.topic_switch_vi(corpus, z(attr)),
        ankura.validate.topic_word_divergence(corpus, topics, z(attr)),
        ankura.validate.window_prob(corpus, topics, z(attr)),
    ])
