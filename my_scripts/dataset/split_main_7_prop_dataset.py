import csv
import numpy as np
import math

train_labels_filepath = 'data/labels/train22_relabeled.csv'
# train_labels_filepath = 'data/labels/train22.csv'
train_dataset_filepath = 'data/labels/main_7_prop_train.csv'
val_dataset_filepath = 'data/labels/main_7_prop_val.csv'

# train 0.9 val 0.1
split_val_ratio = 0.1
# rare samples split ratio
split_rare_val_ratio = 0.2
# set random seed
np.random.seed(13)

labels_header = []
all_labels_data_info = []

# 7 main props
# 'upperLength', 'clothesStyles', 'hairStyles', 'lowerLength',
# 'lowerStyles', 'shoesStyles', 'towards'
main_7_props = ['upperLength', 'clothesStyles', 'hairStyles', 'lowerLength', 'lowerStyles', 'shoesStyles', 'towards']
props_classes = {
    'upperLength': {'ShortSleeve', 'LongSleeve'},
    'clothesStyles': {'multicolour', 'Solidcolor'},
    'hairStyles': {'middle', 'Long', 'Short'},
    'lowerLength': {'Skirt', 'Trousers'},
    'lowerStyles': {'multicolour', 'Solidcolor'},
    'shoesStyles': {'Sandals', 'LeatherShoes', 'Sneaker'},
    'towards': {'right', 'left', 'front', 'back'}
}
class_to_idx = {}
for _prop in props_classes:
    class_to_idx[_prop] = {_class: j for j, _class in enumerate(props_classes[_prop])}

main_7_props_labels_indexes = [
    [[], []],
    [[], []],
    [[], [], []],
    [[], []],
    [[], []],
    [[], [], []],
    [[], [], [], []]
]

# exclude rare classes
exclude_props = ['upperLength', 'hairStyles', 'lowerLength', 'shoesStyles', 'clothesStyles', 'lowerStyles']
upperLength_exclude = {'NoSleeve'}
hairStyles_exclude ={'Bald'}
lowerLength_exclude = {'Shorts'}
shoesStyles_exclude = {'else'}
clothesStyles_exclude = {'lattice'}
lowerStyles_exclude = {'lattice'}

exclude_classes = {
    'upperLength': upperLength_exclude,
    'hairStyles': hairStyles_exclude,
    'lowerLength': lowerLength_exclude,
    'shoesStyles': shoesStyles_exclude,
    'clothesStyles': clothesStyles_exclude,
    'lowerStyles': lowerStyles_exclude
}
exclude_prop_classes_indexes = [[], [], [], [], [], []]

with open(train_labels_filepath) as f:
    reader = csv.DictReader(f)
    upper_labels_header = reader.fieldnames
    print('upper_labels_header: ', upper_labels_header)
    for (row_index, row) in enumerate(reader):
        all_labels_data_info.append(row)
        is_normal = True
        for (j, _prop) in enumerate(exclude_props):
            if row[_prop] in exclude_classes[_prop]:
                exclude_prop_classes_indexes[j].append(row_index)
                is_normal = False
        if is_normal:
            for (prop_index, _prop) in enumerate(main_7_props):
                _class = row[_prop]
                _class_idx = class_to_idx[_prop][_class]
                main_7_props_labels_indexes[prop_index][_class_idx].append(row_index)

def split_by_ratio(val_sample, class_samples, num_of_thr):
    intersection = list(set(val_sample).intersection(set(class_samples)))
    if len(intersection) < num_of_thr:
        unintersection = list(set(class_samples) - set(intersection))
        num_of_split = num_of_thr - len(intersection)
        return unintersection[:num_of_split]
    else:
        return []

def split_val_by_main_7_props_samples():
    val_dataset_indexes = []
    for _prop_info in main_7_props_labels_indexes:
        for _class_idx_list in _prop_info:
            # shuffle samples
            np.random.shuffle(_class_idx_list)
            num_of_val = math.ceil(len(_class_idx_list) * split_val_ratio)
            split_samples = split_by_ratio(val_dataset_indexes, _class_idx_list, num_of_val)
            val_dataset_indexes = val_dataset_indexes + split_samples
    return list(set(val_dataset_indexes))

def split_val_by_exclude_props_samples():
    val_dataset_indexes = []
    for _class_idx_list in exclude_prop_classes_indexes:
        # shuffle samples
        np.random.shuffle(_class_idx_list)
        ratio = split_val_ratio
        if len(_class_idx_list) < 20:
            ratio = split_rare_val_ratio
        num_of_val = math.ceil(len(_class_idx_list) * ratio)
        split_samples = split_by_ratio(val_dataset_indexes, _class_idx_list, num_of_val)
        val_dataset_indexes = val_dataset_indexes + split_samples
    return list(set(val_dataset_indexes))

def generate_csv_file(dataset, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = upper_labels_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in dataset:
            writer.writerow(all_labels_data_info[index])

def get_classes_counter(data_info, classes_list, class_to_idx_map):
    # count classes
    classes_counter = [0 for class_idx in range(len(classes_list))]
    for row in data_info:
        for _prop in main_7_props:
            class_full_name = _prop + '_' + row[_prop]
            class_idx = class_to_idx_map[class_full_name]
            classes_counter[class_idx] = classes_counter[class_idx] + 1
    return classes_counter

def analyze_classes_blance(val_idxes_dataset):
    all_classes_list = []
    for _prop in props_classes:
        _classes = props_classes[_prop]
        if _prop in exclude_classes:
            _classes = _classes.union(exclude_classes[_prop])
        for _class_name in _classes:
            class_full_name = _prop + '_' + _class_name
            all_classes_list.append(class_full_name)
    list.sort(all_classes_list)
    class_to_idx_map = {_class: j for j, _class in enumerate(all_classes_list)}
    # count classes
    all_classes_counter = get_classes_counter(all_labels_data_info, all_classes_list, class_to_idx_map)
    val_labels_data_info = [all_labels_data_info[row_index] for row_index in val_idxes_dataset]
    val_classes_counter = get_classes_counter(val_labels_data_info, all_classes_list, class_to_idx_map)
    # analyze classes
    for class_idx, all_count in enumerate(all_classes_counter):
        class_full_name = all_classes_list[class_idx]
        val_count = val_classes_counter[class_idx]
        ratio = val_count / all_count
        ratio_str = '%.2f' % ratio
        print(class_full_name + ': ')
        print('all: ' + str(all_count) + ', val: ' + str(val_count) + ', ratio: ' + ratio_str)
    total_all_count = sum(all_classes_counter)
    total_val_count = sum(val_classes_counter)
    total_ratio = total_val_count / total_all_count
    total_ratio_str = '%.2f' % total_ratio
    print('all total count: ' + str(total_all_count) + ', val total count: ' + str(total_val_count) + ', ratio: ' + total_ratio_str)

if __name__ == "__main__":
    # split val dataset
    val_idxes_dataset = split_val_by_main_7_props_samples()
    val_idxes_dataset = val_idxes_dataset + split_val_by_exclude_props_samples()
    val_idxes_set = set(val_idxes_dataset)
    val_idxes_dataset = list(val_idxes_set)
    all_idxes_set = set(range(len(all_labels_data_info)))
    # split train dataset
    train_idxes_dataset = list(all_idxes_set - val_idxes_set)
    # sort
    list.sort(val_idxes_dataset)
    list.sort(train_idxes_dataset)
    # analyze classes
    analyze_classes_blance(val_idxes_dataset)
    print('total samples: ' + str(len(all_labels_data_info)) + ', train samples: ' + str(len(train_idxes_dataset)) + ', val samples: ' + str(len(val_idxes_dataset)))
    # generate file
    generate_csv_file(train_idxes_dataset, train_dataset_filepath)
    generate_csv_file(val_idxes_set, val_dataset_filepath)
