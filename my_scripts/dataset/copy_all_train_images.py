import csv
import os
import shutil
from pathlib import Path

trainA_img_root_path = 'data/train1A/'
testA_img_root_path = 'data/testA/'
testB_img_root_path = 'data/testB/'

all_img_root_path = 'data/all_train_imgs/'
# check if dir exist
Path(all_img_root_path).mkdir(parents=True, exist_ok=True)

def copy_imgs(org_file_path, target_file_path):
    # all imgs
    for file_name in os.listdir(org_file_path):
        org_dir = os.path.join(org_file_path, file_name)
        target_dir = os.path.join(target_file_path, file_name)
        # copy
        shutil.copyfile(org_dir, target_dir)
    print(org_file_path + ' accomplished.')

if __name__ == "__main__":
    # trainA
    copy_imgs(trainA_img_root_path, all_img_root_path)
    # testA
    copy_imgs(testA_img_root_path, all_img_root_path)
    # testB
    copy_imgs(testB_img_root_path, all_img_root_path)
    print('copy img successfullly.')