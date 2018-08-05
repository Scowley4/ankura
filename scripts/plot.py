import csv
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

question = 'select_the_most_appropriate_topic_for_the_underlined_word'
metrics = ['switchp', 'switchv', 'worddiv', 'windowp', 'avgrank', 'coherence',
        'sigwuni', 'sigwvac', 'sigdback', 'avglen', 'majper']

algo_markers = {
    'cluster': 'o',
    'anchor': 'p',
    'copula': '+',
    'lda': 'x',
}
corp_colors = {
    'newsgroups': 'red',
    'nyt': 'green',
    'amazon': 'blue',
}

data = {}
for row in csv.DictReader(open('scripts/metrics.csv')):
    key = ('{}_{}_{}'.format(row['corpus'], row['algorithm'], row['k']))
    data[key] = {
        'correct': 0,
        'attempts': 0,
        'flub': 0,
    }
    data[key].update({m: row[m] for m in metrics})

report = []
# report += list(csv.DictReader(open('report1.csv')))
# report += list(csv.DictReader(open('report2.csv')))
report += list(csv.DictReader(open('scripts/report3-agg.csv')))
for row in report:
    model = row['model']
    answers = row[question].split()
    data[model]['attempts'] += len(answers)
    for answer in answers:
        if row['z'] == answer:
            data[model]['correct'] += 1

# Make table

# dss = ['amazon', 'newsgroups', 'nyt']
# print('|  |' + ' | '.join(dss) + ' |')
# print('| -- ' * 4 + '|')
# for i, metric in enumerate(metrics):
    # plt.figure(i+1)
    # xs, ys = [], []

    # print('| ' + metric, end=' | ')
    # for ds in dss:
        # for model, results in data.items():
            # corpus, algo, k = model.split('_')
            # if corpus != ds:
                # continue

            # accuracy = results['correct'] / results['attempts']
            # xs.append(accuracy)
            # evaluation = float(results[metric])
            # ys.append(evaluation)
        # _, _, rval, _, _ = scipy.stats.linregress(xs, ys)
        # print('%.4g' % rval**2, end=' | ')
    # print()
# exit()

# Make plot

for i, metric in enumerate(metrics):
    # plt.figure(i+1)
    xs, ys = [], []

    for model, results in data.items():
        corpus, algo, k = model.split('_')
        # if algo == 'cluster' or corpus != 'amazon':
        # if algo == 'cluster' or corpus != 'amazon':
        # if corpus != 'amazon' or int(k) != 100:
        # if False:
        # if int(k) != 100:
            # continue
        if corpus != 'newsgroups':
            continue

        accuracy = results['correct'] / results['attempts']
        xs.append(accuracy)
        evaluation = float(results[metric])
        ys.append(evaluation)

        # plt.scatter(accuracy,
                    # evaluation,
                    # s=int(k)*2,
                    # c=corp_colors[corpus],
                    # marker=algo_markers[algo])

    m, b, rval, _, _ = scipy.stats.linregress(xs, ys)
    # plt.plot(xs, b + m * np.array(xs))
    print(metric, rval**2)
    # plt.xlabel('human accuracy')
    # plt.ylabel(metric)

# plt.show()
