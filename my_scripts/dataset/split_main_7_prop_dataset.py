# import csv
# import numpy as np
# import math

# upper_labels_filepath = 'data/labels/train22.csv'
# train_dataset_filepath = 'data/labels/main_7_prop_train.csv'
# val_dataset_filepath = 'data/labels/main_7_prop_val.csv'

# # train 0.9 val 0.1
# split_ratio = 0.9
# # # rare samples split ratio
# # rare_split_ratio = 0.8
# # set random seed
# np.random.seed(13)

# # 7 main props
# # 'upperLength', 'clothesStyles', 'hairStyles', 'lowerLength',
# # 'lowerStyles', 'shoesStyles', 'towards'
# main_7_props = ['upperLength', 'clothesStyles', 'hairStyles', 'lowerLength', 'lowerStyles', 'shoesStyles', 'towards']

# all_labels_data_info = []
# normal_labels_indexes = []
# labels_header = []

# # exclude rare classes
# exclude_props = ['upperLength', 'hairStyles', 'shoesStyles', 'clothesStyles', 'lowerStyles']
# upperLength_exclude = {'NoSleeve'}
# hairStyles_exclude ={'Bald'}
# shoesStyles_exclude = {'else'}
# clothesStyles_exclude = {'lattice'}
# lowerStyles_exclude = {'lattice'}
# exclude_classes = [upperLength_exclude, hairStyles_exclude
#                  , shoesStyles_exclude, clothesStyles_exclude
#                  , lowerStyles_exclude]
# exclude_classes_indexes = [[], [], [], [], []]

# with open(upper_labels_filepath) as f:
#     reader = csv.DictReader(f)
#     upper_labels_header = reader.fieldnames
#     print('upper_labels_header: ', upper_labels_header)
#     for (i, row) in enumerate(reader):
#         all_labels_data_info.append(row)
#         is_normal = True
#         for (j, prop) in exclude_props:
#             if row[prop] in exclude_classes[j]:
#                 exclude_classes_indexes[j].append(i)
#                 is_normal = False
#         if is_normal:
#             normal_labels_indexes.append(i)
            
# def indexof(list, element) -> int:
#     index = -1
#     try:
#         index = list.index(element)
#     finally:
#         return index

# def split_samples():
#     val_dataset_indexes = []
#     all_labels_data_info

#     return []

# def split_intersection_of_list(a, b):
#     intersection = list(set(a).intersection(set(b)))
#     # remove intersection
#     a = list(set(a) - set(intersection))
#     b = list(set(b) - set(intersection))
#     # split intersection elements to both list
#     if len(intersection) > 0:
#         half_len = (int)(len(intersection) / 2)
#         a = a + intersection[:half_len]
#         b = b + intersection[half_len:]
#     return [a, b]

# def split_NoSleeve_Bald_dataset():
#     train_ds = []
#     val_ds = []
#     num_of_val = math.ceil(len(NoSleeve_dataset) * (1-rare_split_ratio))
#     np.random.shuffle(NoSleeve_dataset)
#     val_ds = val_ds + NoSleeve_dataset[:num_of_val]
#     train_ds = train_ds + NoSleeve_dataset[num_of_val:]
    
#     num_of_val = math.ceil(len(Bald_dataset) * (1-rare_split_ratio))
#     np.random.shuffle(Bald_dataset)
#     val_ds = val_ds + Bald_dataset[:num_of_val]
#     train_ds = train_ds + Bald_dataset[num_of_val:]
#     return split_intersection_of_list(train_ds, val_ds)

# def generate_csv_file(dataset, filepath):
#     with open(filepath, 'w', newline='') as csvfile:
#         fieldnames = upper_labels_header
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for index in dataset:
#             writer.writerow(upper_labels_data_info[index])

# if __name__ == "__main__":
#     group_samples = get_group_samples()
#     train_dataset, val_dataset = split_samples(group_samples)
#     # add NoSleeve and Bald data
#     train_part_ds, val_part_ds = split_NoSleeve_Bald_dataset()
#     train_dataset = train_dataset + train_part_ds
#     val_dataset = val_dataset + val_part_ds
#     # sort
#     list.sort(train_dataset)
#     list.sort(val_dataset)
#     # generate file
#     generate_csv_file(train_dataset, train_dataset_filepath)
#     generate_csv_file(val_dataset, val_dataset_filepath)
    