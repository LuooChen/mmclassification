# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os

from mmcls.apis import infer_upper, inference_multi_label_model, init_model, show_result_pyplot

config = "work_dirs/my_vgg16_b16_upper_2/my_vgg16_b16_upper_2.py"
checkpoint = "work_dirs/my_vgg16_b16_upper_2/epoch_40.pth"
# device = "cpu"
device = "cuda:0"

data_root = 'data/train1A/'
# img = "img_qh_train1A_00000000076.jpg"
# img = "img_qh_train1A_00004001877.jpg"
# img = "img_qh_train1A_00007006813.jpg"
img = "img_qh_train1A_00009003190.jpg"
# img = "img_qh_train1A_00012003448.jpg"
# img = "img_qh_train1A_00248006874.jpg"
# img = "img_qh_train1A_00277007915.jpg"
# img = "img_qh_train1A_00567003517.jpg"
# img = "img_qh_train1A_00001004361.jpg"#train data
# img = "img_qh_train1A_00036008927.jpg"#train data
# build the model from a config file and a checkpoint file
model = init_model(config, checkpoint, device=device)
# test a single image
result = infer_upper(model, os.path.join(data_root, img))
result['name'] = img
print("result: ", result)
# show the results
# show_result_pyplot(model, args.img, result)