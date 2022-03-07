import csv
import numpy as np
import math
import operator
result = [{'pred_label': 0, 'pred_scores': 0.10613499, 'pred_class': 'LongSleeve'}, {'pred_label': 1, 'pred_scores': 0.88286436, 'pred_class': 'ShortSleeve'}, {'pred_label': 2, 'pred_scores': 0.017300867, 'pred_class': 'NoSleeve'}, {'pred_label': 3, 'pred_scores': 0.93226445, 'pred_class': 'Solidcolor'}, {'pred_label': 4, 'pred_scores': 0.06425162, 'pred_class': 'multicolour'}, {'pred_label': 5, 'pred_scores': 0.01219423, 'pred_class': 'lattice'}, {'pred_label': 6, 'pred_scores': 0.4980164, 'pred_class': 'Long'}, {'pred_label': 7, 'pred_scores': 0.15807417, 'pred_class': 'middle'}, {'pred_label': 8, 'pred_scores': 0.41643718, 'pred_class': 'Short'}, {'pred_label': 9, 'pred_scores': 0.015486953, 'pred_class': 'Bald'}]
# print(max(result, key=operator.attrgetter('pred_scores')))
# print(max(row['pred_scores'] for row in result))
print(max(result, key=lambda item: item['pred_scores']))



