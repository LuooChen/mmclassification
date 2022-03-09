import json

lattice_json_path = 'data/json/upper_colors_2_lattice_result_e40.json'
multicolour_json_path = 'data/json/upper_colors_2_multicolour_result_e40.json'

lattice_json = None
multicolour_json = None
with open(lattice_json_path, 'r') as load_f:
    lattice_json = json.load(load_f)
with open(multicolour_json_path, 'r') as load_f:
    multicolour_json = json.load(load_f)
    
score_threshold = 0.2
def get_positive_negative_count(ds):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for name in ds['two']:
        row = ds['two'][name]
        if float(row[2]['pred_scores']) <= score_threshold:
            # two true
            tp = tp + 1
        else:
            # three false
            fn = fn + 1
    for name in ds['three']:
        row = ds['three'][name]
        if float(row[2]['pred_scores']) > score_threshold:
            # three true
            tn = tn + 1
        else:
            # two false
            fp = fp + 1
    return [tp, fp, tn, fn]

def cal_precision(tp, fp):
    if tp+fp == 0:
        return 0
    return tp/(tp+fp)

def cal_recall(tp, fn):
    if tp+fn == 0:
        return 0
    return tp/(tp+fn)

def cal_f1(precision, recall):
    if precision+recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)

def cal_accuracy(tp, fp, tn, fn):
    return (tp+tn)/(tp+fp+tn+fn)

def get_positive_negative_two_info():
    # two positive, three negative
    lattice_tp, lattice_fp, lattice_tn, lattice_fn = get_positive_negative_count(lattice_json)
    # print([lattice_tp, lattice_fp, lattice_tn, lattice_fn])
    # print('lattice two count: ', len(lattice_json['two']))
    # print('lattice three count: ', len(lattice_json['three']))
    
    multicolour_tp, multicolour_fp, multicolour_tn, multicolour_fn = get_positive_negative_count(multicolour_json)
    # print([multicolour_tp, multicolour_fp, multicolour_tn, multicolour_fn])
    # print('multicolour two count: ', len(multicolour_json['two']))
    # print('multicolour three count: ', len(multicolour_json['three']))
    
    total_tp = lattice_tp + multicolour_tp
    total_fp = lattice_fp + multicolour_fp
    total_tn = lattice_tn + multicolour_tn
    total_fn = lattice_fn + multicolour_fn
    return [total_tp, total_fp, total_tn, total_fn]

def cal_two_info():
    # two positive, three negative
    total_tp, total_fp, total_tn, total_fn = get_positive_negative_two_info()
    precision = cal_precision(total_tp, total_fp)
    recall = cal_recall(total_tp, total_fn)
    f1_score = cal_f1(precision, recall)
    accuracy = cal_accuracy(total_tp, total_fp, total_tn, total_fn)
    print('precision of two: ', precision)
    print('recall of two: ', recall)
    print('f1_score of two: ', f1_score)
    print('accuracy of two: ', accuracy)
    return f1_score

def cal_three_info():
    # two negative, three positive
    total_tn, total_fn, total_tp, total_fp = get_positive_negative_two_info()
    precision = cal_precision(total_tp, total_fp)
    recall = cal_recall(total_tp, total_fn)
    f1_score = cal_f1(precision, recall)
    accuracy = cal_accuracy(total_tp, total_fp, total_tn, total_fn)
    print('precision of three: ', precision)
    print('recall of three: ', recall)
    print('f1_score of three: ', f1_score)
    print('accuracy of three: ', accuracy)
    return f1_score

def cal_weight_f1():
    two_f1 = cal_two_info()
    three_f1 = cal_three_info()
    weight_f1 = 0.1*three_f1 + 0.9*two_f1
    print('weight_f1: ', weight_f1)
    return weight_f1

if __name__ == '__main__':
    best_f1 = 0
    best_threshold = 0.0
    for i in range(10):
        score_threshold = 0.1 * i
        weight_f1 = cal_weight_f1()
        if weight_f1 > best_f1:
            best_f1 = weight_f1
            best_threshold = score_threshold
        print('score_threshold: ', score_threshold)
        print('weight_f1: ', weight_f1)
        print()
    print('best_threshold: ', best_threshold)
    print('best_f1: ', best_f1)