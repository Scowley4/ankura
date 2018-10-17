import glob

import ankura
import psshlib

bible = ankura.corpus.bible(remove_stopwords=True, remove_empty=False, use_stemmer=True)
# datasets = ['tske', 'obib'] + ['obib{}'.format(i) for i in range(11)]
datasets = ['tske', 'obib0', 'obib5']
fnames = glob.glob('aml/scratch/xrefs/*.txt')

for dataset, fname in psshlib.pproduct(datasets, fnames):
    dataset_xref = 'xref-{}'.format(dataset)
    gold = set()
    for doc in bible.documents:
        verse = doc.metadata['verse']
        for ref in doc.metadata[dataset_xref]:
            gold.add((verse, ref))

    with open(fname.replace('xrefs', 'evals').replace('.txt', '_{}.csv'.format(dataset)), 'w') as f:
        f.write('tp,fp,fn,tn\n')

        a, b = 1, 2

        tp = 0
        fp = 0
        fn = len(gold)
        tn = len(bible.documents) * (len(bible.documents) - 1) - fn

        for i, line in enumerate(open(fname)):
            ref = tuple(line.split())
            if ref in gold:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1

            if i == a:
                f.write('{},{},{},{}\n'.format(tp,fp,fn,tn))
                a, b = b, a+b

        f.write('{},{},{},{}\n'.format(tp,fp,fn,tn))
