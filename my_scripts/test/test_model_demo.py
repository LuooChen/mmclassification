# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os

from mmcls.apis import inference_model, init_model, show_result_pyplot

config = "configs/resnet/resnet50_8xb32_in1k.py"
checkpoint = "checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
device = "cpu"
# device = "cuda:0"

img = "demo/demo.JPEG"
model = init_model(config, checkpoint, device=device)
# test a single image
result = inference_model(model, img)
print("result: ", result)
# show the results
# show_result_pyplot(model, args.img, result)