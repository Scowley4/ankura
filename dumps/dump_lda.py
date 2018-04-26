import gensim as gs

import ankura

import dumps.dump_csv as dc

corpus = ankura.corpus.amazon()
K = 20
attr = 'lda_{}'.format(20)

bows = ankura.topic._gensim_bows(corpus)
lda = gs.models.LdaModel(
    corpus=bows,
    num_topics=K,
)
ankura.topic._gensim_assign(corpus, bows, lda, dc.theta(attr), dc.z(attr))
topics = lda.get_topics().T

dc.dump_csv('lda_{}'.format(K), corpus, topics)
