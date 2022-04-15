import csv
import numpy as np
import math

train_labels_filepath = 'data/labels/lower_colors/train22_lower_colors_lables.csv'

lower_train_dataset_filepath = 'data/labels/lower_colors/lower_colors_train.csv'
lower_val_dataset_filepath = 'data/labels/lower_colors/lower_colors_val.csv'

# train 0.9 val 0.1
split_val_ratio = 0.1
# set random seed
np.random.seed(13)

# 'clothesStyles'
clothesStyles_classes = ['Solidcolor', 'multicolour', 'lattice']
clothesStyles_class_to_idx = {_class: idx for idx, _class in enumerate(clothesStyles_classes)}

# 'lowerStyles'
lowerStyles_classes = ['Solidcolor', 'multicolour', 'lattice']
lowerStyles_class_to_idx = {_class: idx for idx, _class in enumerate(lowerStyles_classes)}
# lower_colors
lower_colors = ['lowerBlack',
'lowerBrown', 'lowerBlue', 'lowerGreen', 'lowerGray', 'lowerOrange',
'lowerPink', 'lowerPurple', 'lowerRed', 'lowerWhite', 'lowerYellow']

all_labels_data_info = []
labels_header = []
with open(train_labels_filepath) as f:
    reader = csv.DictReader(f)
    labels_header = reader.fieldnames
    for (index, row) in enumerate(reader):
        all_labels_data_info.append(row)

def indexof(list, element) -> int:
    index = -1
    try:
        index = list.index(element)
    finally:
        return index

def get_lower_group_samples() -> list:
    lower_group_samples = []
    for i in range(3):
        lowerStyles_level = [[] for j in range(11)]
        lower_group_samples.append(lowerStyles_level)

    for (index, row) in enumerate(all_labels_data_info):
        # lower
        lowerStyles_index = lowerStyles_class_to_idx[row['lowerStyles']]
        for (color_index, color) in enumerate(lower_colors):
            color_val = row[color]
            if color_val != '':
                # filter NaN
                lower_group_samples[lowerStyles_index][color_index].append(index)
    return lower_group_samples

def split_by_ratio(val_sample, class_samples, num_of_thr) -> list:
    intersection = list(set(val_sample).intersection(set(class_samples)))
    if len(intersection) < num_of_thr:
        unintersection = list(set(class_samples) - set(intersection))
        num_of_split = num_of_thr - len(intersection)
        return unintersection[:num_of_split]
    else:
        return []

def split_samples(group_samples):
    val_dataset = []
    # split
    for i in range(3):
        for j in range(11):
            samples = group_samples[i][j]
            # do not split val dataset when samples less than 6
            if len(samples) <= 5:
                continue
            # shuffle samples
            np.random.shuffle(samples)
            num_of_val = math.ceil((len(samples) * split_val_ratio))
            split_samples = split_by_ratio(val_dataset, samples, num_of_val)
            val_dataset = val_dataset + split_samples
    return list(set(val_dataset))

def split_lower_rare_class() -> list:
    # split lowerOrange
    val_dataset = []
    rare_val_ratio = 0.2
    all_lowerOrange = []
    for i in range(len(all_labels_data_info)):
        row = all_labels_data_info[i]
        if row['lowerOrange'] != '':
            all_lowerOrange.append(i)
    # shuffle samples
    np.random.shuffle(all_lowerOrange)
    num_of_val = math.ceil((len(all_lowerOrange) * rare_val_ratio))
    val_dataset = val_dataset + all_lowerOrange[:num_of_val]
    return val_dataset

def generate_csv_file(dataset, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = labels_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in dataset:
            writer.writerow(all_labels_data_info[index])

def get_classes_counter(data_info, classes_list, class_to_idx_map):
    # count classes
    classes_counter = [0 for class_idx in range(len(classes_list))]
    for row in data_info:
        for _class in classes_list:
            _class_val = row[_class]
            if _class_val != '':
                class_idx = class_to_idx_map[_class]
                classes_counter[class_idx] = classes_counter[class_idx] + 1
    return classes_counter

def analyze_classes_blance(val_idxes_dataset, _classes):
    all_classes_list = _classes
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

def split_val_by_group_samples(group_samples, _classes, train_dataset_filepath, val_dataset_filepath):
    # split
    val_idxes_dataset = split_samples(group_samples)
    val_idxes_set = set(val_idxes_dataset)
    val_idxes_dataset = list(val_idxes_dataset)
    all_idxes_set = set(range(len(all_labels_data_info)))

    # split train dataset
    train_idxes_dataset = list(all_idxes_set - val_idxes_set)
    # sort
    list.sort(val_idxes_dataset)
    list.sort(train_idxes_dataset)
    # analyze classes
    analyze_classes_blance(val_idxes_dataset, _classes)
    print('total samples: ' + str(len(all_labels_data_info)) + ', train samples: ' + str(len(train_idxes_dataset)) + ', val samples: ' + str(len(val_idxes_dataset)))
    # generate file
    generate_csv_file(train_idxes_dataset, train_dataset_filepath)
    generate_csv_file(val_idxes_set, val_dataset_filepath)

if __name__ == "__main__":
    lower_group_samples = get_lower_group_samples()
    # split lower
    print('Split lower dataset start.')
    split_val_by_group_samples(lower_group_samples, lower_colors, lower_train_dataset_filepath, lower_val_dataset_filepath)
    print('Split lower dataset end.')