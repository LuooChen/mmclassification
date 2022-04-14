import csv
import numpy as np

train_labels_filepath = 'data/labels/train22_relabeled.csv'
labels_header = None
all_labels_data_info = []
image_prefix = "train2_new/"
rewrite_train_labels_filepath = "data/labels/train22_relabeled_rewrite.csv"
# read
with open(train_labels_filepath) as f:
    reader = csv.DictReader(f)
    labels_header = reader.fieldnames
    for row in reader:
        row['name'] = image_prefix + row['name']
        all_labels_data_info.append(row)

# rewrite
with open(rewrite_train_labels_filepath, 'w', newline='') as csvfile:
    fieldnames = labels_header
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_labels_data_info:
        writer.writerow(row)
print("rewrite successfully.")