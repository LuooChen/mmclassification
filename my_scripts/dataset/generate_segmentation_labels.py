import os
import csv
import copy

rewrite_train_labels_filepath = "data/labels/train22_relabeled_rewrite.csv"
data_root = 'data/train22'
seg_colors_image_prefix = "segmentation_colors/"
seg_main_image_prefix = "segmentation_main/"
# train_and_seg_labels_filepath = "data/labels/train22_and_seg_relabeled_rewrite.csv"
train_and_seg_labels_filepath = "data/labels/segmentation_labels.csv"

all_labels = {}
# read
with open(rewrite_train_labels_filepath) as f:
    reader = csv.DictReader(f)
    labels_header = reader.fieldnames
    for row in reader:
        img_name = row['name'].split('/')[1]
        all_labels[img_name] = row

def get_segmentation_list(data_root, seg_dir) -> list:
    seg_data_list = []
    for seg_img_name in os.listdir(os.path.join(data_root, seg_dir)):
        img_prefix_name = seg_img_name.split('_seg')[0]
        img_suffix_name = seg_img_name.split('_seg')[1]
        img_name = img_prefix_name + img_suffix_name
        if img_name in all_labels:
            seg_row = copy.deepcopy(all_labels[img_name])
            seg_row['name'] = os.path.join(seg_dir, seg_img_name)
            seg_data_list.append(seg_row)
    return seg_data_list

def generate_csv_file(data_info, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = labels_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_info:
            writer.writerow(row)

if __name__ == '__main__':
    # get segmentation data info
    seg_colors_list = get_segmentation_list(data_root, seg_colors_image_prefix)
    seg_main_list = get_segmentation_list(data_root, seg_main_image_prefix)
    segmentation_list = seg_colors_list + seg_main_list
    # generate csv
    generate_csv_file(segmentation_list, train_and_seg_labels_filepath)
    print("generate successfully.")