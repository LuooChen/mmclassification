import cv2
import csv
from pathlib import Path
import os
import copy

all_labels = "data/labels/upper_colors/train22_relabeled_rewrite.csv"
src_image_path = "data/train22/"
upper_colors_augmentation_image_path = "data/train22/upper_colors_augmentation/"
upper_colors_augmentation_image_prefix = "upper_colors_augmentation/"
upper_colors_augmentation_labels = "data/labels/upper_colors/upper_colors_augmentation.csv"
# check if dir exist
Path(upper_colors_augmentation_image_path).mkdir(parents=True, exist_ok=True)
allow_props = ['upperOrange', 'upperPurple', 'upperBrown', 'upperGreen']
rotation_angles = {
    'upperOrange': [15],
    'upperPurple': [],
    'upperBrown': [],
    'upperGreen': []
}
allow_augmentation_data_info = {}
for _prop in allow_props:
    allow_augmentation_data_info[_prop] = []
# labels header
labels_header = None
# all augmentation labels
all_augmentation_data_info = []
# read
with open(all_labels) as f:
    reader = csv.DictReader(f)
    labels_header = reader.fieldnames
    for row in reader:
        for _prop in allow_props:
            if row[_prop] == '':
                continue
            allow_augmentation_data_info[_prop].append(row)
            break

def horizontal_flip(img):
    return cv2.flip(img, 1)

def rotation(img, angle):
    # angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def join_image_path(img_name, aug_name, suffix):
    return upper_colors_augmentation_image_prefix + img_name + '_' + aug_name + '.' + suffix

def img_aumentation(img_src, img_full_name, prop) -> list:
    # img_full_name = img_src.split('/')[1]
    img_name = img_full_name.split('.')[0]
    img_suffix = img_full_name.split('.')[1]
    img = cv2.imread(img_src)
    img_flip = horizontal_flip(img)
    augmentation_images = {}
    augmentation_images[join_image_path(img_name, 'flip', img_suffix)] = img_flip
    for angle in rotation_angles[prop]:
        # src rotation
        img_rotation = rotation(img, angle)
        augmentation_images[join_image_path(img_name, str(angle), img_suffix)] = img_rotation
        img_rotation_reverse = rotation(img, -angle)
        augmentation_images[join_image_path(img_name, str(angle)+'reverse', img_suffix)] = img_rotation_reverse

        # flip rotation
        img_rotation = rotation(img_flip, angle)
        augmentation_images[join_image_path(img_name, 'flip' + str(angle), img_suffix)] = img_rotation
        img_rotation_reverse = rotation(img_flip, -angle)
        augmentation_images[join_image_path(img_name, 'flip' + str(angle) + 'reverse', img_suffix)] = img_rotation_reverse
    return augmentation_images

def save_augmentation_imgs(org_img_info, augmentation_images) -> list:
    augmentation_img_infos = []
    for img_path_name in augmentation_images:
        aug_img_info = copy.deepcopy(org_img_info)
        aug_img_info['name'] = img_path_name
        # save image
        cv2.imwrite(os.path.join(src_image_path, img_path_name), augmentation_images[img_path_name])
        # cv2.imwrite(image_dst, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # add label
        augmentation_img_infos.append(aug_img_info)
    return augmentation_img_infos

def generate_csv_file(data_info, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = labels_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_info:
            writer.writerow(row)

if __name__ == "__main__":
    # augmentation
    for _prop in allow_augmentation_data_info:
        for _row in allow_augmentation_data_info[_prop]:
            img_path_name = _row['name']
            img_full_name = img_path_name.split('/')[1]
            augmentation_images = img_aumentation(os.path.join(src_image_path, img_path_name), img_full_name, _prop)
            augmentation_img_infos = save_augmentation_imgs(_row, augmentation_images)
            all_augmentation_data_info = all_augmentation_data_info + augmentation_img_infos
    # write labels
    generate_csv_file(all_augmentation_data_info, upper_colors_augmentation_labels)
    print("rewrite successfully.")