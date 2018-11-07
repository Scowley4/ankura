import glob

import ankura
import psshlib

bible = ankura.corpus.bible(remove_stopwords=True, remove_empty=False, use_stemmer=True)
verses = bible.metadata['verses']
D = len(bible.documents)

datasets = ['tske', 'obib0', 'obib5']
fnames = glob.glob('/users/home/jlund3/aml/scratch/xrefs/*.txt')

for dataset, fname in psshlib.pproduct(datasets, fnames):
    dataset_xref = 'xref-{}'.format(dataset)
    gold = set()
    for doc in bible.documents:
        verse = doc.metadata['verse']
        for ref in doc.metadata[dataset_xref]:
            if ref in verses and ref != verse:
                gold.add((verse, ref))

    with open(fname.replace('xrefs', 'evals').replace('.txt', '_{}.csv'.format(dataset)), 'w') as f:
        print('tp,fp,fn,tn', file=f)

        a, b = 1, 2

        tp = 0
        fp = 0
        fn = len(gold)
        tn = D*(D-1) - fn

        output = lambda: print(tp, fp, fn, tn, sep=',', file=f)

        output()
        for i, line in enumerate(open(fname)):
            ref = tuple(line.split())
            if ref in gold:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1

            if i == a:
                output()
                a, b = b, a+b
        output()
