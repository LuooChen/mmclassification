# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os

from mmcls.apis import infer_pedestrian_info, init_model, show_result_pyplot
config_root = "work_dirs"
main_7_props_config_name = "my_vgg16_b16_main_7_props_11"
main_7_props_epoch_name = "epoch_37.pth"
upper_colors_config_name = "my_vgg16_b16_upper_colors_11"
upper_colors_epoch_name = "epoch_34.pth"
lower_colors_config_name = "my_vgg16_b16_lower_colors_11"
lower_colors_epoch_name = "epoch_39.pth"

# device = "cpu"
device = "cuda:0"

main_7_props_config = os.path.join(config_root, main_7_props_config_name, main_7_props_config_name+".py")
main_7_props_checkpoint = os.path.join(config_root, main_7_props_config_name, main_7_props_epoch_name)
upper_colors_config = os.path.join(config_root, upper_colors_config_name, upper_colors_config_name+".py")
upper_colors_checkpoint = os.path.join(config_root, upper_colors_config_name, upper_colors_epoch_name)
lower_colors_config = os.path.join(config_root, lower_colors_config_name, lower_colors_config_name+".py")
lower_colors_checkpoint = os.path.join(config_root, lower_colors_config_name, lower_colors_epoch_name)

# init models
model_main_7_props = init_model(main_7_props_config, main_7_props_checkpoint, device=device)
model_upper_colors = init_model(upper_colors_config, upper_colors_checkpoint, device=device)
model_lower_colors = init_model(lower_colors_config, lower_colors_checkpoint, device=device)

def test_pedestrian_info(img):
    # test a single image
    result = infer_pedestrian_info(model_main_7_props, model_upper_colors, model_lower_colors, img)
    return result

data_root = 'data/train22/train2_new'
img = "img_qh_train2_00000006345.jpg"
# build the model from a config file and a checkpoint file
model_result = test_pedestrian_info(os.path.join(data_root, img))
print("result: ", model_result)