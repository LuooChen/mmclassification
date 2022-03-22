import csv
import numpy as np
import math

upper_labels_filepath = 'data/labels/train1A_relabeled.csv'
train_dataset_filepath = 'data/labels/upper_train.csv'
val_dataset_filepath = 'data/labels/upper_val.csv'

# train 0.9 val 0.1
split_ratio = 0.9
# rare samples split ratio
rare_split_ratio = 0.8
# set random seed
np.random.seed(13)

# 'upperLength' exclude 'NoSleeve'
upperLength_classes = ['LongSleeve', 'ShortSleeve']
# 'clothesStyles'
clothesStyles_classes = ['Solidcolor', 'multicolour', 'lattice']
# 'hairStyles' exclude 'Bald'
hairStyles_classes = ['Long', 'middle', 'Short']

upper_labels_data_info = []
upper_labels_header = []
NoSleeve_dataset = []
Bald_dataset = []
with open(upper_labels_filepath) as f:
    reader = csv.DictReader(f)
    upper_labels_header = reader.fieldnames
    print('upper_labels_header: ', upper_labels_header)
    for (index, row) in enumerate(reader):
        upper_labels_data_info.append(row)
        if row['upperLength'] == 'NoSleeve':
            NoSleeve_dataset.append(index)
        if row['hairStyles'] == 'Bald':
            Bald_dataset.append(index)
            
def indexof(list, element) -> int:
    index = -1
    try:
        index = list.index(element)
    finally:
        return index

def get_group_samples() -> list:
    group_samples = []
    for i in range(2):
        upperLength_level = []
        for j in range(3):
            clothesStyles_level = []
            for k in range(3):
                hairStyles_level = []
                clothesStyles_level.append(hairStyles_level)
            upperLength_level.append(clothesStyles_level)
        group_samples.append(upperLength_level)

    for (index, row) in enumerate(upper_labels_data_info):
        upperLength_index = indexof(upperLength_classes, row['upperLength'])
        clothesStyles_index = indexof(clothesStyles_classes, row['clothesStyles'])
        hairStyles_index = indexof(hairStyles_classes, row['hairStyles'])
        if upperLength_index == -1 or hairStyles_index == -1:
            continue
        group_samples[upperLength_index][clothesStyles_index][hairStyles_index].append(index)
    return group_samples

def split_samples(group_samples):
    train_dataset = []
    val_dataset = []
    for i in range(2):
        for j in range(3):
            for k in range(3):
                samples = group_samples[i][j][k]
                print(upperLength_classes[i] + ', ' + clothesStyles_classes[j] + ', ' + hairStyles_classes[k] + ' samples len: ', len(samples))
                num_of_train = (int)(len(samples) * split_ratio)
                np.random.shuffle(samples)
                train_dataset = train_dataset + samples[:num_of_train]
                val_dataset = val_dataset + samples[num_of_train:]
    return [train_dataset, val_dataset]

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

def split_NoSleeve_Bald_dataset():
    train_ds = []
    val_ds = []
    num_of_val = math.ceil(len(NoSleeve_dataset) * (1-rare_split_ratio))
    np.random.shuffle(NoSleeve_dataset)
    val_ds = val_ds + NoSleeve_dataset[:num_of_val]
    train_ds = train_ds + NoSleeve_dataset[num_of_val:]
    
    num_of_val = math.ceil(len(Bald_dataset) * (1-rare_split_ratio))
    np.random.shuffle(Bald_dataset)
    val_ds = val_ds + Bald_dataset[:num_of_val]
    train_ds = train_ds + Bald_dataset[num_of_val:]
    return split_intersection_of_list(train_ds, val_ds)

def generate_csv_file(dataset, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = upper_labels_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in dataset:
            writer.writerow(upper_labels_data_info[index])

if __name__ == "__main__":
    group_samples = get_group_samples()
    train_dataset, val_dataset = split_samples(group_samples)
    # add NoSleeve and Bald data
    train_part_ds, val_part_ds = split_NoSleeve_Bald_dataset()
    train_dataset = train_dataset + train_part_ds
    val_dataset = val_dataset + val_part_ds
    # sort
    list.sort(train_dataset)
    list.sort(val_dataset)
    # generate file
    generate_csv_file(train_dataset, train_dataset_filepath)
    generate_csv_file(val_dataset, val_dataset_filepath)
    