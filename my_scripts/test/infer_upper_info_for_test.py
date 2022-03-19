# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import csv

from mmcls.apis import infer_upper_info, inference_multi_label_model, init_model, show_result_pyplot

upper_config = "work_dirs/my_vgg16_b16_upper_7/my_vgg16_b16_upper_7.py"
upper_checkpoint = "work_dirs/my_vgg16_b16_upper_7/epoch_37.pth"
upper_colors_config = "work_dirs/my_vgg16_b16_upper_colors_7/my_vgg16_b16_upper_colors_7.py"
upper_colors_checkpoint = "work_dirs/my_vgg16_b16_upper_colors_7/epoch_36.pth"
# device = "cpu"
device = "cuda:0"
# data_root = 'data/testA'
data_root = 'data/testB'
save_csv_path = 'data/submitB/result_7_best.csv'
# img = "img_qh_train1A_00359008148.jpg"
submit_csv_header = ['name', 'upperLength', 'clothesStyles', 'hairStyles', 'upperBlack', 'upperBrown', 'upperBlue', 'upperGreen', 'upperGray', 'upperOrange', 'upperPink', 'upperPurple', 'upperRed', 'upperWhite', 'upperYellow']
# init models
model_upper = init_model(upper_config, upper_checkpoint, device=device)
model_upper_colors = init_model(upper_colors_config, upper_colors_checkpoint, device=device)
    
def test_upper_info(img):
    # test a single image
    result = infer_upper_info(model_upper, model_upper_colors, img)
    return result

def generate_csv_file(data, csv_header, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = csv_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def convert_to_csv_row(result):
    row = {}
    # fill cols
    for col in submit_csv_header:
        row[col] = ''
    row['name'] = result['name']
    # 'upperLength', 'clothesStyles', 'hairStyles'
    row['upperLength'] = result['upperLength']['pred_class']
    row['clothesStyles'] = result['clothesStyles']['pred_class']
    row['hairStyles'] = result['hairStyles']['pred_class']
    for color_result in result['upper_colors']:
        row[color_result['pred_class']] = 1
    return row

if __name__ == '__main__':
    result_info = []
    for img in os.listdir(data_root):
        result = test_upper_info(os.path.join(data_root, img))
        result['name'] = img
        row = convert_to_csv_row(result)
        result_info.append(row)
    # write csv
    generate_csv_file(result_info, submit_csv_header, save_csv_path)