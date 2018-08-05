import ankura
import numpy as np

corpus = ankura.corpus.newsgroups()
Q = ankura.anchor.build_cooccurrence(corpus)
train, test = ankura.pipeline.train_test_split(corpus)
gold = [doc.metadata['newsgroup'] for doc in test.documents]

inferences = ['vari', 'samp', 'icm', 'reph']
inferences = ['rephxl']
inferences = ['rephus']


def bound(z_attr, topics, alpha):
    bound = 0
    for doc in corpus.documents:
        for w_n, z_n in zip(doc.tokens, doc.metadata[z_attr]):
            bound += np.log(alpha + topics[w_n.token, z_n])
    return bound


for k in [20, 100, 200, 300]:
    anchors = ankura.anchor.gram_schmidt_anchors(corpus, Q, k)
    topics = ankura.anchor.recover_topics(Q, anchors)

    # ankura.topic.variational_assign(corpus, topics, z_attr='z_vari_%03d' % k)
    # ankura.topic.sampling_assign(corpus, topics, z_attr='z_samp_%03d' % k)
    # ankura.topic.mode_assign(corpus, topics, z_attr='z_icm_%03d' % k)
    # ankura.topic.mode_assign2(corpus, topics, z_attr='z_rephxl_%03d' % k, n=100)
    ankura.topic.mode_assign3(corpus, topics, z_attr='z_rephus_%03d' % k)

    for inf in inferences:
        z_attr = 'z_%s_%03d' % (inf, k)
        model = ankura.topic.classifier(train, topics, 'newsgroup', z_attr)

        acc = sum([g == p for g, p in zip(gold, model(test))]) / len(gold)
        switchp = ankura.validate.topic_switch_percent(corpus, z_attr)
        logp = bound(z_attr, topics, .01)

        print('%s %.3f %.3f %.4g' % (z_attr, acc, switchp, logp))

    print()
