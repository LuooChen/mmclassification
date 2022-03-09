from val_ds_analyze import get_val_ds
from mmcls.apis import infer_upper_colors_top3, inference_multi_label_model, init_model, show_result_pyplot
import os
import json
import numpy as np

lattice_json_path = 'data/json/upper_colors_2_lattice_result_e25.json'
multicolour_json_path = 'data/json/upper_colors_2_multicolour_result_e25.json'

config = "work_dirs/my_vgg16_b16_upper_colors_2/my_vgg16_b16_upper_colors_2.py"
checkpoint = "work_dirs/my_vgg16_b16_upper_colors_2/epoch_25.pth"
# device = "cpu"
device = "cuda:0"
data_root = 'data/train1A/'
# build the model from a config file and a checkpoint file
model = init_model(config, checkpoint, device=device)

def infer_ds(ds):
    infer_result_json = {
        'two': {},
        'three': {}
    }
    for img in ds['two']:
        # test a single image
        result = infer_upper_colors_top3(model, os.path.join(data_root, img))
        for row in result:
            row['pred_scores'] = str(row['pred_scores'])
        infer_result_json['two'][img] = result
        
    for img in ds['three']:
        # test a single image
        result = infer_upper_colors_top3(model, os.path.join(data_root, img))
        for row in result:
            row['pred_scores'] = str(row['pred_scores'])
        infer_result_json['three'][img] = result
    return infer_result_json

if __name__ == '__main__':
    lattice_ds, multicolour_ds = get_val_ds()
    lattice_infer_result_json = infer_ds(lattice_ds)
    multicolour_infer_result_json = infer_ds(multicolour_ds)
    # print(lattice_infer_result_json)
    with open(lattice_json_path, "w") as f:
        json.dump(lattice_infer_result_json, f)
        
    with open(multicolour_json_path, "w") as f:
        json.dump(multicolour_infer_result_json, f)