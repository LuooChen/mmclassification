import os
import csv
import copy

data_root = 'data/train22'
seg_colors_image_prefix = "segmentation_colors/"
seg_main_image_prefix = "segmentation_main/"

def rename_dir_imgs(data_root, seg_dir):
    seg_dir_path = os.path.join(data_root, seg_dir)
    for old_name in os.listdir(seg_dir_path):
        img_prefix_name = old_name.split('.')[0]
        img_suffix_name = old_name.split('.')[1]
        new_name = img_prefix_name + '_seg.' + img_suffix_name
        # Renaming the file
        os.rename(os.path.join(seg_dir_path, old_name), os.path.join(seg_dir_path, new_name))

if __name__ == '__main__':
    rename_dir_imgs(data_root, seg_colors_image_prefix)
    rename_dir_imgs(data_root, seg_main_image_prefix)
    print("rename successfully.")