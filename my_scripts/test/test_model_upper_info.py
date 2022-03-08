# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os

from mmcls.apis import infer_upper_info, inference_multi_label_model, init_model, show_result_pyplot

upper_config = "work_dirs/my_vgg16_b16_upper_1/my_vgg16_b16_upper_1.py"
upper_checkpoint = "work_dirs/my_vgg16_b16_upper_1/epoch_40.pth"
upper_colors_config = "work_dirs/my_vgg16_b16_upper_colors_1/my_vgg16_b16_upper_colors_1.py"
upper_colors_checkpoint = "work_dirs/my_vgg16_b16_upper_colors_1/epoch_40.pth"
# device = "cpu"
device = "cuda:0"

data_root = 'data/train1A/'
img = "img_qh_train1A_00359008148.jpg"
# img = "img_qh_train1A_00018000974.jpg"
# img = "img_qh_train1A_00359008148.jpg"
# img = "img_qh_train1A_00412005924.jpg"
# img = "img_qh_train1A_00000000076.jpg"#train
# img = "img_qh_train1A_00001004361.jpg"#train
# build the model from a config file and a checkpoint file
model_upper = init_model(upper_config, upper_checkpoint, device=device)
model_upper_colors = init_model(upper_colors_config, upper_colors_checkpoint, device=device)
# test a single image
result = infer_upper_info(model_upper, model_upper_colors, os.path.join(data_root, img))
print("result: ", result)
# show the results
# show_result_pyplot(model, args.img, result)