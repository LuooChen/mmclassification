import cv2
import csv
from pathlib import Path
import os
import copy
import numpy as np
import random

all_labels = "data/labels/main_7_props/train22_relabeled_rewrite.csv"
src_image_path = "data/train22/"
main_7_props_augmentation_image_path = "data/train22/main_7_props_augmentation/"
main_7_props_augmentation_image_prefix = "main_7_props_augmentation/"
main_7_props_augmentation_labels = "data/labels/main_7_props/main_7_props_augmentation.csv"
# check if dir exist
Path(main_7_props_augmentation_image_path).mkdir(parents=True, exist_ok=True)

allow_props = ['NoSleeve', 'Bald', 'else', 'Shorts', 'clothesStyles_lattice', 'lowerStyles_lattice']
rotation_angles = {
    'NoSleeve': [10],
    'Bald': [10]
}
allow_rotation = {'NoSleeve', 'Bald'}
color_jitters = {
    'NoSleeve': ['b', 's', 'c'],
    'Bald': ['b', 's', 'c'],
    'else': ['b', 's', 'c'],
    'Shorts': ['b', 's', 'c'],
    'clothesStyles_lattice': ['b'],
    'lowerStyles_lattice': ['b']
}
allow_color_jitters = {'NoSleeve', 'Bald', 'else', 'Shorts', 'clothesStyles_lattice', 'lowerStyles_lattice'}
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
        if row['upperLength'] == 'NoSleeve':
            allow_augmentation_data_info['NoSleeve'].append(row)
            continue
        if row['hairStyles'] == 'Bald':
            allow_augmentation_data_info['Bald'].append(row)
            continue
        if row['shoesStyles'] == 'else':
            allow_augmentation_data_info['Shorts'].append(row)
            continue
        if row['lowerLength'] == 'Shorts':
            allow_augmentation_data_info['Shorts'].append(row)
            continue
        if row['clothesStyles'] == 'lattice':
            allow_augmentation_data_info['clothesStyles_lattice'].append(row)
            continue
        if row['lowerStyles'] == 'lattice':
            allow_augmentation_data_info['lowerStyles_lattice'].append(row)
            continue

def horizontal_flip(img):
    return cv2.flip(img, 1)

def rotation(img, angle):
    # angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def convert_to_gray(img):
    # Converting color image to grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def colorjitter(img, cj_type="b"):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "s":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img

def noisy(img, noise_type="gauss"):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    if noise_type == "gauss":
        image=img.copy()
        mean=0
        st=0.7
        gauss = np.random.normal(mean,st,image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image,gauss)
        return image

    elif noise_type == "sp":
        image=img.copy()
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image

def join_image_path(img_name, aug_name, suffix):
    return main_7_props_augmentation_image_prefix + img_name + '_' + aug_name + '.' + suffix

def save_image(org_img_info, img_path_name, _img, is_flip=False):
    aug_img_info = copy.deepcopy(org_img_info)
    aug_img_info['name'] = img_path_name
    if is_flip:
        if aug_img_info['towards'] == 'left':
            aug_img_info['towards'] = 'right'
        elif aug_img_info['towards'] == 'right':
            aug_img_info['towards'] = 'left'
    # save image
    cv2.imwrite(os.path.join(src_image_path, img_path_name), _img)
    # cv2.imwrite(image_dst, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return aug_img_info

def img_aumentation(img_src, src_img_full_name, prop, _row):
    img_name = src_img_full_name.split('.')[0]
    img_suffix = src_img_full_name.split('.')[1]
    img = cv2.imread(img_src)
    imgs_obj = {
        src_img_full_name: [img, _row]
    }
    img_flip = horizontal_flip(img)
    flip_img_full_name = join_image_path(img_name, 'flip', img_suffix)
    aug_img_info_flip = save_image(_row, flip_img_full_name, img_flip, True)
    imgs_obj[flip_img_full_name] = [img_flip, aug_img_info_flip]
    # rotation
    if prop in allow_rotation:
        temp_image_full_name_list = [x for x in imgs_obj.keys()]
        for angle in rotation_angles[prop]:
            index_count = 0
            for _img_full_name in temp_image_full_name_list:
                _img, _img_info = imgs_obj[_img_full_name]
                rotation_img = rotation(_img, angle)
                rotation_img_full_name = join_image_path(img_name, str(angle) + 'angle' + str(index_count), img_suffix)
                rotation_img_info = save_image(_img_info, rotation_img_full_name, rotation_img)
                imgs_obj[rotation_img_full_name] = [rotation_img, rotation_img_info]

                rotation_reverse_img = rotation(_img, -angle)
                rotation_reverse_img_full_name = join_image_path(img_name, str(angle)  + 'angle' + 'reverse' + str(index_count), img_suffix)
                rotation_reverse_info = save_image(_img_info, rotation_reverse_img_full_name, rotation_reverse_img)
                imgs_obj[rotation_reverse_img_full_name] = [rotation_reverse_img, rotation_reverse_info]
                index_count = index_count+1
    # color jitter
    if prop in allow_color_jitters:
        temp_image_full_name_list = [x for x in imgs_obj.keys()]
        for _type in color_jitters[prop]:
            color_count = 0
            for _img_full_name in temp_image_full_name_list:
                _img, _img_info = imgs_obj[_img_full_name]
                color_jitter_img = colorjitter(_img, _type)
                color_jitter_img_full_name = join_image_path(img_name, 'colorjitter'+_type+str(color_count), img_suffix)
                color_jitter_img_info = save_image(_img_info, color_jitter_img_full_name, color_jitter_img)
                imgs_obj[color_jitter_img_full_name] = [color_jitter_img, color_jitter_img_info]
                color_count = color_count+1
    augmentation_img_infos = []
    for _img_full_name in imgs_obj:
        if _img_full_name != src_img_full_name:
            augmentation_img_infos.append(imgs_obj[_img_full_name][1])
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
            src_img_full_name = img_path_name.split('/')[1]
            augmentation_img_infos = img_aumentation(os.path.join(src_image_path, img_path_name), src_img_full_name, _prop, _row)
            all_augmentation_data_info = all_augmentation_data_info + augmentation_img_infos
    # write labels
    generate_csv_file(all_augmentation_data_info, main_7_props_augmentation_labels)
    print("augmentation successfully.")