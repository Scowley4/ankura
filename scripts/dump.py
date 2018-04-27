import csv
import itertools
import pickle
import numpy as np

import gensim as gs

import ankura


def z(attr):
    return '{}_z'.format(attr)


def theta(attr):
    return '{}_theta'.format(attr)


def dump_csv(attr, corpus, topics, n=1000, window_size=10):
    summary = ankura.topic.topic_summary(topics, corpus, n=11)
    highlighter = lambda w, _: '<u>{}</u>'.format(w)
    f = open('scripts/{}.csv'.format(attr), 'w')

    fields = ['text']
    fields.extend('label-{}'.format(i) for i in [1, 2, 3, 4, 5])
    fields.extend('value-{}'.format(i) for i in [1, 2, 3, 4, 5])
    fields.extend(['z', 'bad', 'model'])
    cf = csv.DictWriter(f, fieldnames=fields)
    cf.writeheader()

    for _ in range(n):
        doc = corpus.documents[np.random.choice(len(corpus.documents))]
        argsort = np.argsort(-doc.metadata[theta(attr)])
        good_options = argsort[:4]
        bad_options = argsort[4:]

        n = np.random.choice(len(doc.tokens))
        token = doc.tokens[n]
        topic = doc.metadata[z(attr)][n]

        options = []
        if topic in good_options:
            options.extend(good_options)
        else:
            options.append(topic)
            options.extend(good_options[:3])
        options.append(np.random.choice(bad_options))

        token_start, token_end = token.loc
        if n - window_size <= 0:
            span_start = 0
            span_prefix = ''
        else:
            span_start = doc.tokens[n-window_size].loc[0]
            span_prefix = '<font color="grey">... </font>'
        if n + window_size >= len(doc.tokens):
            span_end = len(doc.text)
            span_postfix = ''
        else:
            span_end = doc.tokens[n+window_size].loc[1]
            span_postfix = '<font color="grey"> ...</font>'

        left = span_prefix + doc.text[span_start: token_start]
        center = highlighter(doc.text[token_start: token_end], topic)
        right = doc.text[token_end: span_end] + span_postfix
        row = {
            'text': left + center + right,
            'z': topic,
            'bad': options[-1],
            'model': attr,
        }

        token_text = corpus.vocabulary[token.token]
        mod_sum = [[w for w in topic if w != token_text][:10] for topic in summary]
        np.random.shuffle(options)
        row.update({'label-{}'.format(i): ('&lt;' + ', '.join(mod_sum[option]) + '&gt;') for i, option in enumerate(options, 1)})
        row.update({'value-{}'.format(i): option for i, option in enumerate(options, 1)})

        cf.writerow(row)

    f.close()


def dump_lda(corpus, K):
    attr = '{}_lda_{}'.format(corpus.metadata['name'], K)

    bows = ankura.topic._gensim_bows(corpus)
    lda = gs.models.LdaModel(
        corpus=bows,
        num_topics=K,
    )
    ankura.topic._gensim_assign(corpus, bows, lda, theta(attr), z(attr))
    topics = lda.get_topics().T

    dump_csv(attr, corpus, topics)


def dump_gs(corpus, K):
    attr = '{}_gs_{}'.format(corpus.metadata['name'], K)
    corpus.metadata[attr] = attr


importers = [
    ankura.corpus.newsgroups,
    ankura.corpus.yelp,
    # ankura.corpus.amazon,
    # ankura.corpus.tripadvisor,
]
dumps = [
    dump_lda,
    dump_gs,
]
Ks = [20, 200]

for importer in importers:
    corpus = importer()
    for dump, K in itertools.product(dumps, Ks):
        dump(corpus, K)
    pickle.dump(corpus, open('scripts/{}.pickle'.format(corpus.metadata['name']), 'wb'))


# # def canned_gs_lda(K):
    # # @ankura.util.pickle_cache('topics.pickle')
    # # def get_gs_lda():
        # # ret = gs_lda(K)
        # # pickle.dump(corpus, open('corpus.pickle', 'wb'))
        # # return ret
    # # return get_gs_lda()


# # def gs_lda(K):
    # # attr = 'GS{}+LDA'.format(K)
    # # anchors = ankura.anchor.gram_schmidt_anchors(corpus, Q, K)
    # # topics = ankura.anchor.recover_topics(Q, anchors)
    # # ankura.topic.lda_assign(corpus, topics, theta_attr=theta(attr), z_attr=z(attr))
    # # return attr, topics


# # def lda(K):
    # # attr = 'LDA{}'.format(K)
    # # topics = ankura.other.gensim_lda(corpus, K, theta_attr=theta(attr), z_attr=z(attr))
    # # return attr, topics


# # def ta_lda(K):
    # # attr = 'TA{}+LDA'.format(K)
    # # seeds = np.random.choice(len(corpus.documents), size=K, replace=False)
    # # indices = [[t.token for t in corpus.documents[d].tokens] for d in seeds]
    # # anchors = ankura.anchor.tandem_anchors(indices, Q)
    # # topics = ankura.anchor.recover_topics(Q, anchors)
    # # ankura.topic.lda_assign(corpus, topics, theta_attr=theta(attr), z_attr=z(attr))
    # # return attr, topics
