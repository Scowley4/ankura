import csv
import itertools

import matplotlib.pyplot as plt

plt.style.use('ggplot')
linestyles=['-', '--', ':', '-.']

csv_base = '/users/home/jlund3/aml/scratch/evals/{}_{}_{}.csv'

models = ['tandem', 'gramschmidt', 'coarse']
metrics = ['cosine', 'cityblock', 'euclidean', 'chebyshev']
datasets = ['tske', 'obib0', 'obib5']

def get_data():
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
    return data

data = get_data()

def select(model, metric, dataset, value):
    return [r[value] for r in data[model, metric, dataset]]


def plot(model, metric, dataset, x, y, n=10, **kwargs):
    x = select(model, metric, dataset, x)
    y = select(model, metric, dataset, y)
    plt.plot(x[n:], y[n:], **kwargs)


def cost_mod():
    plt.figure(figsize=(9, 3))

    ax1 = plt.subplot(131)
    plt.title('TSKE')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'tske', 'pp', 'tp', label=model, linestyle=ls)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('True Positives')
    plt.minorticks_off()

    plt.subplot(132, sharey=ax1)
    plt.title('OpenBible +0')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib0', 'pp', 'tp', label=model, linestyle=ls)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Predicted Positives')
    plt.minorticks_off()

    plt.subplot(133, sharey=ax1)
    plt.title('OpenBible +5')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib5', 'pp', 'tp', label=model, linestyle=ls)
    plt.xscale('log')
    plt.yscale('log')
    plt.minorticks_off()
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
# cost_mod()


def prc_mod():
    plt.figure(figsize=(9, 3))

    ax1 = plt.subplot(131)
    plt.title('TSKE')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'tske', 'tpr', 'ppv', 20, label=model, linestyle=ls)
    plt.ylabel('Precision')

    plt.subplot(132)
    plt.title('OpenBible +0')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib0', 'tpr', 'ppv', 20, label=model, linestyle=ls)
    plt.xlabel('Recall')

    plt.subplot(133)
    plt.title('OpenBible +5')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib5', 'tpr', 'ppv', 20, label=model, linestyle=ls)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
# prc_mod()


def roc_mod():
    plt.figure(figsize=(9, 3))

    ax1 = plt.subplot(131)
    plt.title('TSKE')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'tske', 'fpr', 'tpr', label=model, linestyle=ls)
    plt.ylabel('True Positive Rate')

    plt.subplot(132, sharex=ax1)
    plt.title('OpenBible +0')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib0', 'fpr', 'tpr', label=model, linestyle=ls)
    plt.xlabel('False Positive Rate')

    plt.subplot(133, sharex=ax1)
    plt.title('OpenBible +5')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib5', 'fpr', 'tpr', label=model, linestyle=ls)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
# roc_mod()


def roc_me():
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(131)
    plt.title('Tandem')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for metric, ls in zip(metrics, linestyles):
        plot('tandem', metric, 'tske', 'fpr', 'tpr', label=metric, linestyle=ls)
    plt.ylabel('True Positive Rate')

    plt.subplot(132, sharex=ax1)
    plt.title('Gram-Schmidt')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for metric, ls in zip(metrics, linestyles):
        plot('gramschmidt', metric, 'tske', 'fpr', 'tpr', label=metric, linestyle=ls)
    plt.xlabel('False Positive Rate')

    plt.subplot(133, sharex=ax1)
    plt.title('Coarse')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for metric, ls in zip(metrics, linestyles):
        plot('coarse', metric, 'tske', 'fpr', 'tpr', label=metric, linestyle=ls)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
# roc_me()


def prc_me():
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(131)
    plt.title('Tandem')
    for metric, ls in zip(metrics, linestyles):
        plot('tandem', metric, 'tske', 'tpr', 'ppv', label=metric, linestyle=ls)
    plt.ylabel('Precision')

    plt.subplot(132, sharex=ax1)
    plt.title('Gram-Schmidt')
    for metric, ls in zip(metrics, linestyles):
        plot('gramschmidt', metric, 'tske', 'tpr', 'ppv', label=metric, linestyle=ls)
    plt.xlabel('Recall')

    plt.subplot(133, sharex=ax1)
    plt.title('Coarse')
    for metric, ls in zip(metrics, linestyles):
        plot('coarse', metric, 'tske', 'tpr', 'ppv', label=metric, linestyle=ls)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
# prc_me()
