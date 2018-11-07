import csv
import itertools

import matplotlib.pyplot as plt

plt.style.use('ggplot')
linestyles = ['-', '--', ':', '-.']
markers = ['1', '+', 'x']

csv_evals = '/home/jlund3/evals/{}_{}_{}.csv'
csv_bases = '/home/jlund3/evals/baseline/{}_{}_{}.csv'

models = ['tandem', 'gramschmidt', 'coarse']
metrics = ['cosine', 'cityblock', 'euclidean', 'chebyshev']
baselines = ['wordmatch', 'topicmatch', 'topicwordmatch']
datasets = ['tske', 'obib0', 'obib5']


def complete_evals(values):
    for value in values:
        value['tp'] = int(value['tp'])
        value['fp'] = int(value['fp'])
        value['fn'] = int(value['fn'])
        value['tn'] = int(value['tn'])
        value['tpr'] = value['tp'] / (value['tp'] + value['fn'])
        value['pp'] = value['tp'] + value['fp']
        try:
            value['ppv'] = value['tp'] / (value['tp'] + value['fp'])
        except ZeroDivisionError:
            value['ppv'] = 1
        value['fpr'] = value['fp'] / (value['fp'] + value['tn'])
    return values


def get_data():
    data = {}
    for model, metric, dataset in itertools.product(models, metrics, datasets):
        values = list(csv.DictReader(open(csv_evals.format(model, metric, dataset))))
        data[model, metric, dataset] = complete_evals(values)
    for model, base, dataset in itertools.product(models, baselines, datasets):
        values = list(csv.DictReader(open(csv_bases.format(model, base, dataset))))
        data[model, base, dataset] = complete_evals(values)
    return data
data = get_data()


def select(model, metric, dataset, value):
    return [r[value] for r in data[model, metric, dataset]]


def plot(model, metric, dataset, x, y, n=0, **kwargs):
    x = select(model, metric, dataset, x)
    y = select(model, metric, dataset, y)
    plt.plot(x[n:], y[n:], **kwargs)


def cost_mod():
    plt.figure(figsize=(9, 3))

    ax1 = plt.subplot(131)
    plt.title('TSKE')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'tske', 'pp', 'tp', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'tske', 'pp', 'tp', label=model, linestyle=ls)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('True Positives')
    plt.minorticks_off()

    plt.subplot(132, sharey=ax1)
    plt.title('OpenBible+0')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'obib0', 'pp', 'tp', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib0', 'pp', 'tp', label=model, linestyle=ls)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Predicted Positives')
    plt.minorticks_off()

    plt.subplot(133, sharey=ax1)
    plt.title('OpenBible+5')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'obib5', 'pp', 'tp', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib5', 'pp', 'tp', label=model, linestyle=ls)
    plt.xscale('log')
    plt.yscale('log')
    plt.minorticks_off()

    plt.tight_layout()
    plt.figlegend(ncol=7, loc='lower center')
    plt.show()


def prc_mod():
    plt.figure(figsize=(9, 3))

    ax1 = plt.subplot(131)
    plt.title('TSKE')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'tske', 'tpr', 'ppv', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'tske', 'tpr', 'ppv', label=model,
                linestyle=ls,n=10)
    plt.ylabel('Precision')
    plt.ylim(0, .05)

    plt.subplot(132)
    plt.title('OpenBible+0')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'obib0', 'tpr', 'ppv', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib0', 'tpr', 'ppv', label=model,
                linestyle=ls,n=10)
    plt.xlabel('Recall')
    plt.ylim(0, .05)

    plt.subplot(133)
    plt.title('OpenBible+5')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'obib5', 'tpr', 'ppv', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib5', 'tpr', 'ppv', label=model,
                linestyle=ls,n=10)
    plt.ylim(0, .05)

    plt.figlegend(ncol=7, loc='lower center')
    plt.tight_layout()
    plt.show()


def roc_mod():
    plt.figure(figsize=(9, 3))

    ax1 = plt.subplot(131)
    plt.title('TSKE')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'tske', 'fpr', 'tpr', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'tske', 'fpr', 'tpr', label=model, linestyle=ls)
    plt.ylabel('True Positive Rate')

    plt.subplot(132, sharex=ax1)
    plt.title('OpenBible+0')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'obib0', 'fpr', 'tpr', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib0', 'fpr', 'tpr', label=model, linestyle=ls)
    plt.xlabel('False Positive Rate')

    plt.subplot(133, sharex=ax1)
    plt.title('OpenBible+5')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for model, (base, m) in itertools.product(models, zip(baselines, markers)):
        plot(model, base, 'obib5', 'fpr', 'tpr', linestyle='', label=base, color='darkslategray', marker=m)
    for model, ls in zip(models, linestyles):
        plot(model, 'cosine', 'obib5', 'fpr', 'tpr', label=model, linestyle=ls)

    plt.figlegend(ncol=7, loc='lower center')
    plt.tight_layout()
    plt.show()


def roc_me():
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(131)
    plt.title('Tandem')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for base, m in zip(baselines, markers):
        plot('tandem', base, 'tske', 'fpr', 'tpr', color='darkslategray', linestyle='', label=base, marker=m)
    for metric, ls in zip(metrics, linestyles):
        plot('tandem', metric, 'tske', 'fpr', 'tpr', label=metric, linestyle=ls)
    plt.ylabel('True Positive Rate')

    plt.subplot(132, sharex=ax1)
    plt.title('Gram-Schmidt')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for base, m in zip(baselines, markers):
        plot('gramschmidt', base, 'tske', 'fpr', 'tpr', color='darkslategray', linestyle='', label=base, marker=m)
    for metric, ls in zip(metrics, linestyles):
        plot('gramschmidt', metric, 'tske', 'fpr', 'tpr', label=metric, linestyle=ls)
    plt.xlabel('False Positive Rate')

    plt.subplot(133, sharex=ax1)
    plt.title('Coarse')
    plt.plot([0, 1], [0, 1], color='white', linewidth=1, linestyle='-')
    for base, m in zip(baselines, markers):
        plot('coarse', base, 'tske', 'fpr', 'tpr', color='darkslategray', linestyle='', label=base, marker=m)
    for metric, ls in zip(metrics, linestyles):
        plot('coarse', metric, 'tske', 'fpr', 'tpr', label=metric, linestyle=ls)

    plt.figlegend(ncol=7, loc='lower center')
    plt.tight_layout()
    plt.show()


def prc_me():
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(131)
    plt.title('Tandem')
    for base, m in zip(baselines, markers):
        plot('tandem', base, 'tske', 'tpr', 'ppv', color='darkslategray', linestyle='', label=base, marker=m)
    for metric, ls in zip(metrics, linestyles):
        plot('tandem', metric, 'tske', 'tpr', 'ppv', label=metric,
                linestyle=ls,n=10)
    plt.ylabel('Precision')
    plt.ylim(0, .05)

    plt.subplot(132, sharex=ax1)
    plt.title('Gram-Schmidt')
    for base, m in zip(baselines, markers):
        plot('gramschmidt', base, 'tske', 'tpr', 'ppv', color='darkslategray', linestyle='', label=base, marker=m)
    for metric, ls in zip(metrics, linestyles):
        plot('gramschmidt', metric, 'tske', 'tpr', 'ppv', label=metric,
                linestyle=ls,n=10)
    plt.xlabel('Recall')
    plt.ylim(0, .05)

    plt.subplot(133, sharex=ax1)
    plt.title('Coarse')
    for base, m in zip(baselines, markers):
        plot('coarse', base, 'tske', 'tpr', 'ppv', color='darkslategray', linestyle='', label=base, marker=m)
    for metric, ls in zip(metrics, linestyles):
        plot('coarse', metric, 'tske', 'tpr', 'ppv', label=metric,
                linestyle=ls,n=10)
    plt.ylim(0, .05)

    plt.figlegend(ncol=8, loc='lower center')
    plt.tight_layout()
    plt.show()


# # Establish cosine
roc_me()
prc_me()

# # Establish tandem
roc_mod()
prc_mod()

# # Win.
cost_mod()
