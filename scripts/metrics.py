import os
import glob
import ankura
import pickle
import csv
import numpy as np

fields = ['corpus', 'algorithm', 'k', 'switchp', 'switchv', 'worddiv', 'windowp', 'avgrank', 'coherence', 'sigwuni', 'sigwvac', 'sigdback']
params, metrics = fields[:3], fields[3:]

new_fields = ['avglen', 'majper']

fnames = [fname for fname in glob.glob('/users/home/jlund3//aml/scratch/tlta*/*.csv') if 'cluster' not in fname]

new_csv = csv.DictWriter(open('scripts/metrics.csv', 'w'), fields+new_fields)
new_csv.writeheader()

for fname in fnames:
    model = os.path.splitext(os.path.basename(fname))[0]

    print('Computing', model)
    corpus = pickle.load(open(fname.replace('.csv', '.corpus.pickle'), 'rb'))
    topics = pickle.load(open(fname.replace('.csv', '.topics.pickle'), 'rb'))
    summary = ankura.topic.topic_summary(topics)
    Q = ankura.anchor.build_cooccurrence(corpus)
    k = topics.shape[1]
    topic_cols = [topics[:, t] for t in range(k)]
    zattr = '{}_z'.format(model)
    tattr = '{}_theta'.format(model)

    cont = ankura.validate.topic_switch_contingency(corpus, zattr)

    new_csv.writerow({
        'corpus': model.split('_')[0],
        'algorithm': model.split('_')[1],
        'k': int(model.split('_')[2]),

        'switchp': ankura.validate.topic_switch_percent(corpus, zattr),
        'switchv': ankura.validate.topic_switch_vi(corpus, zattr),
        'worddiv': ankura.validate.topic_word_divergence(corpus, topics, zattr),
        'windowp': ankura.validate.window_prob(corpus, topics, zattr),

        'avgrank': ankura.validate.avg_word_rank(corpus, topics, zattr),
        'coherence': ankura.validate.coherence(corpus, summary),
        'sigwuni': np.mean([ankura.validate.significance_wuni(t) for t in topic_cols]),
        'sigwvac': np.mean([ankura.validate.significance_wvac(t, Q) for t in topic_cols]),
        'sigdback': np.mean([ankura.validate.significance_dback(t, corpus, tattr) for t in range(k)]),

        'avglen': ankura.validate.topic_run_len(corpus, zattr),
        'majper': ankura.validate.majority_percent(corpus, zattr),
    })
