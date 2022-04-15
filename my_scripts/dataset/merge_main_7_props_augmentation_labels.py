import csv
import numpy as np

train_labels_filepath = 'data/labels/main_7_props/train22_relabeled_rewrite.csv'
augmentation_labels_filepath = 'data/labels/main_7_props/main_7_props_augmentation.csv'
labels_header = None
all_labels_data_info = []
merge_labels = "data/labels/main_7_props/train22_main_7_props_lables.csv"
# read
with open(train_labels_filepath) as f:
    reader = csv.DictReader(f)
    labels_header = reader.fieldnames
    for row in reader:
        all_labels_data_info.append(row)

with open(augmentation_labels_filepath) as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_labels_data_info.append(row)

# rewrite
with open(merge_labels, 'w', newline='') as csvfile:
    fieldnames = labels_header
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_labels_data_info:
        writer.writerow(row)
print("merge successfully.")