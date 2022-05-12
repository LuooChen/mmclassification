import csv
import numpy as np

train_labels_filepath = 'data/labels/train22_relabeled_rewrite.csv'
augmentation_labels_filepath = 'data/labels/segmentation_labels.csv'
labels_header = None
all_labels_data_info = []
merge_labels = "data/labels/train22_and_seg_relabeled_rewrite_labels.csv"
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