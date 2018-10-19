import os
import pickle
import sys
import tempfile
import itertools
import functools

import numpy as np

import ankura
import ankura.util
import mrs


class TopicRecover(mrs.MapReduce):

    @functools.lru_cache()
    def get_qank(self):
        pickle_path = os.path.join(self.output_dir(), 'qank.pickle')

        if os.path.exists(pickle_path):
            return pickle.load(open(pickle_path, 'rb'))

        bible = ankura.corpus.bible(remove_stopwords=True, remove_empty=False, use_stemmer=True)
        Q = ankura.anchor.build_cooccurrence(bible)
        anchors = ankura.anchor.gram_schmidt_anchors(bible, Q, 3000, doc_threshold=5)

        ankura.util.ensure_dir(self.output_dir())
        with open(pickle_path, 'wb') as f:
            pickle.dump((Q, anchors), f)

        return Q, anchors

    def input_data(self, job):
        Q, anchors = self.get_qank()
        V = Q.shape[0]
        n = 21
        tasks = ((i, range(i, V, n)) for i in range(n))
        return job.local_data(tasks)

    def run(self, job):
        batches = self.input_data(job)
        intermediate = job.map_data(batches, self.map)
        batches.close()
        output = job.reduce_data(intermediate, self.reduce)
        intermediate.close()

        job.wait(output)
        output.fetchall()

        for _, C in output.data():
            with open(os.path.join(self.output_dir(), 'topics.pickle'), 'wb') as f:
                pickle.dump(np.loads(C), f)
        return 0

    @mrs.output_serializers(key=mrs.str_serializer, value=mrs.raw_serializer)
    def map(self, key, words):
        Q, anchors = self.get_qank()
        V, K = Q.shape[0], len(anchors)

        X = anchors / anchors.sum(axis=1)[:, np.newaxis]
        XX = np.dot(X, X.transpose())

        C = np.zeros((V, K))
        for word in words:
            Q[word] = Q[word] / Q[word].sum()
            C[word] = ankura.anchor.exponentiated_gradient(Q[word], X, XX, 2e-6)

        yield '', C.dumps()

    @mrs.output_serializers(key=mrs.str_serializer, value=mrs.raw_serializer)
    def reduce(self, key, values):
        Q, anchors = self.get_qank()
        V, K = Q.shape[0], len(anchors)

        P_w = np.diag(Q.sum(axis=1))
        for word in range(V):
            if np.isnan(P_w[word, word]):
                P_w[word, word] = 1e-16

        C = np.zeros((V, K))
        for part in values:
            C += np.loads(part)

        A = np.dot(P_w, C)
        for k in range(K):
            A[:, k] = A[:, k] / A[:, k].sum()
        yield C.dumps()


if __name__ == '__main__':
    mrs.main(TopicRecover)
