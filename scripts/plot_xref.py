import csv
import itertools

import matplotlib.pyplot as plt

plt.style.use('ggplot')
linestyles=['-', '--', ':', '-.']

csv_base = '/users/scratch/jlund3/evals/{}_{}_{}.csv'

models = ['tandem', 'gramschmidt', 'coarse']
metrics = ['cosine', 'cityblock', 'euclidean', 'chebyshev']
datasets = ['tske', 'obib0', 'obib5']

data = {}
for model, metric, dataset in itertools.product(models, metrics, datasets):
    values = list(csv.DictReader(open(csv_base.format(model, metric, dataset))))
    for value in values:
        value['tp'] = int(value['tp'])
        value['fp'] = int(value['fp'])
        value['fn'] = int(value['fn'])
        value['tn'] = int(value['tn'])
        value['tpr'] = value['tp'] / (value['tp'] + value['fn'])
        value['pp'] = value['tp'] + value['fp']
        value['ppv'] = value['tp'] / (value['tp'] + value['fp'])
        value['fpr'] = value['fp'] / (value['fp'] + value['tn'])
    data[model, metric, dataset] = values


def select(mode, metric, dataset, value):
    return [r[value] for r in data[model, metric, dataset]]


def plot(model, metric, dataset, x, y, **kwargs):
    x = select(model, metric, dataset, x)
    y = select(model, metric, dataset, y)
    plt.plot(x, y, **kwargs)


def roc(model, metric, dataset, **kwargs):
    plot(model, metric, dataset, 'fpr', 'tpr', **kwargs)


for dataset in datasets:
    plt.figure(figsize=(6, 6))
    plt.title(dataset)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', dataset, 'pp', 'tp', label=model, linestyle=ls)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xscale('log')
    plt.legend()
    plt.show()
