import csv
import collections
import krippendorff as kr
import numpy as np

question = 'select_the_most_appropriate_topic_for_the_underlined_word'
values = ['value-{}'.format(i+1) for i in range(5)]
rows = list(csv.DictReader(open('report3.csv')))

models = set()
data = collections.defaultdict(list)
for rid, row in enumerate(rows):
    row_data = [0] * 5
    lookup = {row[v]: i for i, v in enumerate(values)}
    for ans in row[question].split():
        row_data[lookup[ans]] += 1

    data['all'].append(row_data)
    data[row['model']].append(row_data)

aggreement = {model: kr.alpha(value_counts=np.array(counts), level_of_measurement='nominal') for model, counts in data.items()}
sorted_models = sorted(aggreement, key=aggreement.get, reverse=True)
for model in sorted_models:
    print(model, aggreement[model])
