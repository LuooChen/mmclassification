# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import csv

from mmcls.apis import infer_pedestrian_info, init_model, show_result_pyplot

main_7_props_config = "work_dirs/my_vgg16_b16_upper_9/my_vgg16_b16_upper_9.py"
main_7_props_checkpoint = "work_dirs/my_vgg16_b16_upper_9/epoch_38.pth"
upper_colors_config = "work_dirs/my_vgg16_b16_upper_colors_9/my_vgg16_b16_upper_colors_9.py"
upper_colors_checkpoint = "work_dirs/my_vgg16_b16_upper_colors_9/epoch_21.pth"
lower_colors_config = "work_dirs/my_vgg16_b16_upper_colors_9/my_vgg16_b16_upper_colors_9.py"
lower_colors_checkpoint = "work_dirs/my_vgg16_b16_upper_colors_9/epoch_21.pth"
# device = "cpu"
device = "cuda:0"
data_root = 'data/test22A'
save_csv_path = 'data/submitPhase2/result_9_best.csv'

submit_csv_header = ['name', 'upperLength', 'clothesStyles', 'hairStyles', 'lowerLength',
'lowerStyles', 'shoesStyles', 'towards', 'upperBlack', 'upperBrown',
'upperBlue', 'upperGreen', 'upperGray', 'upperOrange', 'upperPink',
'upperPurple', 'upperRed', 'upperWhite', 'upperYellow', 'lowerBlack',
'lowerBrown', 'lowerBlue', 'lowerGreen', 'lowerGray', 'lowerOrange',
'lowerPink', 'lowerPurple', 'lowerRed', 'lowerWhite', 'lowerYellow']
# init models
model_main_7_props = init_model(main_7_props_config, main_7_props_checkpoint, device=device)
model_upper_colors = init_model(upper_colors_config, upper_colors_checkpoint, device=device)
model_lower_colors = init_model(lower_colors_config, lower_colors_checkpoint, device=device)

def test_pedestrian_info(img):
    # test a single image
    result = infer_pedestrian_info(model_main_7_props, model_upper_colors, model_lower_colors, img)
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
    main_7_props = ['upperLength', 'clothesStyles', 'hairStyles', 'lowerLength',
                    'lowerStyles', 'shoesStyles', 'towards']
    # main 7 props
    for _prop in main_7_props:
        row[_prop] = result[_prop]['pred_class']
    # upper colors
    for color_result in result['upper_colors']:
        row[color_result['pred_class']] = 1
    # lower colors
    for color_result in result['lower_colors']:
        row[color_result['pred_class']] = 1
    return row

if __name__ == '__main__':
    result_info = []
    for img in os.listdir(data_root):
        result = test_pedestrian_info(os.path.join(data_root, img))
        result['name'] = img
        row = convert_to_csv_row(result)
        result_info.append(row)
    # write csv
    generate_csv_file(result_info, submit_csv_header, save_csv_path)