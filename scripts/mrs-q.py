import functools
import sys
import pickle

import scipy.sparse
import numpy as np

import ankura
import mrs

class ConstructQ(mrs.MapReduce):

    @functools.lru_cache()
    def get_corpus(self):
        if len(self.args) != 2:
            print("Requires input and an output.", file=sys.stderr)
            return None

        return pickle.load(open(self.args[0], 'rb'))

    def input_data(self, job):
        corpus = self.get_corpus()
        D = len(corpus.documents)
        n = self.opts.num_tasks
        return job.local_data((i, range(i, D, n)) for i in range(n))

    def run(self, job):
        batches = self.input_data(job)
        intermediate = job.map_data(batches, self.map)
        batches.close()
        output = job.reduce_data(intermediate, self.reduce)
        intermediate.close()

        job.wait(output)
        output.fetchall()

        for _, Q in output.data():
            with open(self.args[1], 'wb') as f:
                pickle.dump(np.loads(Q), f, protocol=4)
        return 0

    @mrs.output_serializers(key=mrs.str_serializer, value=mrs.raw_serializer)
    def map(self, key, batch):
        corpus = self.get_corpus()
        V = len(corpus.vocabulary)
        Q = np.zeros((V, V))

        for d in batch:
            doc = corpus.documents[d]
            n_d = len(doc.tokens)
            if n_d <= 1:
                continue

            norm = 1 / (n_d * (n_d - 1))
            for i, w_i in enumerate(doc.tokens):
                for j, w_j in enumerate(doc.tokens):
                    if i == j:
                        continue
                    Q[w_i.token, w_j.token] += norm

        yield '', pickle.dumps(scipy.sparse.coo_matrix(Q))

    @mrs.output_serializers(key=mrs.str_serializer, value=mrs.raw_serializer)
    def reduce(self, key, values):
        corpus = self.get_corpus()
        V = len(corpus.vocabulary)
        Q = np.zeros((V, V))

        for Q_part in values:
            Q += pickle.loads(Q_part)
        Q /= Q.sum()

        yield Q.dumps()


    @classmethod
    def update_parser(cls, parser):
        parser.add_option('-t', '--tasks',
            dest='num_tasks', type=int,
            help='Number of map tasks to use',
            default=20)
        return parser


if __name__ == '__main__':
    mrs.main(ConstructQ)
