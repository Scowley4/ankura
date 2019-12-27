"""Prints 1000 predicted biblical cross-reference from 3000 topics"""

import itertools
import ankura

bible = ankura.corpus.bible(remove_stopwords=True, remove_empty=False, use_stemmer=True)

Q = ankura.anchor.build_cooccurrence(bible)
anchors = ankura.anchor.doc_anchors(bible, Q, 3000)
topics = ankura.anchor.recover_topics(Q, anchors)
ankura.assign.variational(bible, topics, 'theta')

for i, j in itertools.islice(ankura.topic.pdists(bible, 'theta'), 5000):
    verse_i = bible.documents[i].metadata['verse']
    verse_j = bible.documents[j].metadata['verse']
    correct = verse_j in bible.documents[i].metadata['xref-tske']
    print(verse_i, verse_j, correct)
