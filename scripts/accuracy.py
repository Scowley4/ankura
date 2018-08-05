import csv
import collections
import matplotlib.pyplot as plt
import numpy as np

question = 'select_the_most_appropriate_topic_for_the_underlined_word'
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
algo_y = {
    'copula': 0,
    'lda': 1,
    'anchor': 2,
}

data = {}
for row in csv.DictReader(open('report3-agg.csv')):
    model = row['model']
    if model not in data:
        data[model] = collections.defaultdict(int)
    answers = row[question].split()
    data[model]['attempts'] += len(answers)
    for answer in answers:
        if row['z'] == answer:
            data[model]['correct'] += 1

for model, stats in data.items():
    data[model] = stats['correct'] / stats['attempts']

asdf = [[], [], []]
for model, accuracy in data.items():
    corpus, algo, k = model.split('_')
    if int(k) > 100:
        continue
    asdf[algo_y[algo]].append(accuracy)
    # plt.scatter(accuracy,
                # algo_y[algo],
                # s=int(k)*2,
                # c=corp_colors[corpus],
               # )

# for a in asdf:
    # print(np.mean(a), np.median(a))
# plt.show()

plt.style.use('ggplot')
plt.boxplot(asdf)
plt.ylabel('Human-Model Agreement')
plt.xticks([1, 2, 3], ['CopulaLDA', 'LDA', 'Anchor Words'])

plt.show()
