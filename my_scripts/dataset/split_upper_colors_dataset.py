import csv
import numpy as np

upper_labels_filepath = 'data/train1A_relabeled.csv'
train_dataset_filepath = 'data/upper_colors_train.csv'
val_dataset_filepath = 'data/upper_colors_val.csv'

# train 0.9 val 0.1
split_ratio = 0.9
# set random seed
np.random.seed(13)

# 'clothesStyles'
clothesStyles_classes = ['Solidcolor', 'multicolour', 'lattice']
# upper_colors
upper_colors = ['upperBlack',
'upperBrown', 'upperBlue', 'upperGreen', 'upperGray', 'upperOrange',
'upperPink', 'upperPurple', 'upperRed', 'upperWhite', 'upperYellow']

upper_colors_labels_data_info = []
upper_colors_labels_header = []
with open(upper_labels_filepath) as f:
    reader = csv.DictReader(f)
    upper_colors_labels_header = reader.fieldnames
    for (index, row) in enumerate(reader):
        upper_colors_labels_data_info.append(row)
            
def indexof(list, element) -> int:
    index = -1
    try:
        index = list.index(element)
    finally:
        return index

def get_group_samples() -> list:
    group_samples = []
    for i in range(3):
        clothesStyles_level = []
        for j in range(11):
            upper_colors_level = []
            clothesStyles_level.append(upper_colors_level)
        group_samples.append(clothesStyles_level)

    for (index, row) in enumerate(upper_colors_labels_data_info):
        clothesStyles_index = indexof(clothesStyles_classes, row['clothesStyles'])
        for (color_index, color) in enumerate(upper_colors):
            color_val = row[color]
            if color_val != '':
                # filter NaN
                group_samples[clothesStyles_index][color_index].append(index)
    return group_samples

def split_samples(group_samples):
    train_dataset = []
    val_dataset = []
    # rough split
    for i in range(3):
        for j in range(11):
            samples = group_samples[i][j]
            print(clothesStyles_classes[i] + ', ' + upper_colors[j] + ' samples len: ', len(samples))
            num_of_train = len(samples)
            # do not split val dataset when samples less than 6
            if len(samples) > 5:
                num_of_train = (int)(len(samples) * split_ratio)
            np.random.shuffle(samples)
            train_dataset = train_dataset + samples[:num_of_train]
            val_dataset = val_dataset + samples[num_of_train:]
    # duplicate elimination
    return split_intersection_of_list(train_dataset, val_dataset)

def split_intersection_of_list(a, b):
    intersection = list(set(a).intersection(set(b)))
    # remove intersection
    a = list(set(a) - set(intersection))
    b = list(set(b) - set(intersection))
    # split intersection elements to both list
    if len(intersection) > 0:
        half_len = (int)(len(intersection) / 2)
        a = a + intersection[:half_len]
        b = b + intersection[half_len:]
    return [a, b]

def generate_csv_file(dataset, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = upper_colors_labels_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in dataset:
            writer.writerow(upper_colors_labels_data_info[index])

if __name__ == "__main__":
    group_samples = get_group_samples()
    train_dataset, val_dataset = split_samples(group_samples)
    # sort
    list.sort(train_dataset)
    list.sort(val_dataset)
    # generate file
    generate_csv_file(train_dataset, train_dataset_filepath)
    generate_csv_file(val_dataset, val_dataset_filepath)
    