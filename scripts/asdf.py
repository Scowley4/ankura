import ankura

corpus = ankura.corpus.newsgroups()
topics = ankura.anchor.anchor_algorithm(corpus, 100)
summary = ankura.topic.topic_summary(topics, corpus)

sample = ankura.pipeline.sample_corpus(corpus, 10)

ankura.topic.variational_assign(sample, topics, z_attr='z_vari')
ankura.topic.mode_assign2(sample, topics, z_attr='z_reph')

for doc in sample.documents:
    print('****', doc.metadata['id'], '****')
    for topic in set(doc.metadata['z_vari'] + doc.metadata['z_reph']):
        print(topic, ':', ' '.join(summary[topic]))
    print('**')
    print(ankura.topic.highlight(doc, 'z_vari'))
    print('**')
    print(ankura.topic.highlight(doc, 'z_reph'))
    print()
