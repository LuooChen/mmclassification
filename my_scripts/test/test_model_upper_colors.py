# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os

from mmcls.apis import infer_upper_colors_top3, inference_multi_label_model, init_model, show_result_pyplot

config = "work_dirs/my_vgg16_b16_upper_colors_1/my_vgg16_b16_upper_colors_1.py"
checkpoint = "work_dirs/my_vgg16_b16_upper_colors_1/epoch_40.pth"
# device = "cpu"
device = "cuda:0"

data_root = 'data/train1A/'
# img = "img_qh_train1A_00005004956.jpg"
# img = "img_qh_train1A_00018000974.jpg"
# img = "img_qh_train1A_00359008148.jpg"
# img = "img_qh_train1A_00412005924.jpg"
# img = "img_qh_train1A_00000000076.jpg"#train
img = "img_qh_train1A_00001004361.jpg"#train
# build the model from a config file and a checkpoint file
model = init_model(config, checkpoint, device=device)
# test a single image
result = infer_upper_colors_top3(model, os.path.join(data_root, img))
print("result: ", result)
# show the results
# show_result_pyplot(model, args.img, result)