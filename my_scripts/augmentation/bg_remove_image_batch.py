from cv2 import mixChannels
import bg_remove_image
import os

import pandas as pd
csv_file = "/home/superdisk/pedestrian-fine-recognition/data/train22/train2_new.csv"
data = pd.read_csv(csv_file)
print(data.keys)
exit()
data_list = data.values.tolist()

# 类别极不平衡类别名
badcases = ["NoSleeve", "lattice", "Bald", "Shorts", "l"]

badcase = list()

for item_list in data.index.values:
    item_list = item_list.to_dict()
    exit()
#     # print(item_list)
#     for item in item_list:
#         if item in badcases:
#             badcase.append(item_list)
#             break
    
# print(len(badcase))
exit()
# 目标目录
src_dir = "F:/2018-09-03/python/script/"
# 生成图片目录
dst_dir = "F:/2018-09-03/python/script/"
# api url
url = 'http://10.191.20.112:3000/api/imageRemoveBg'
image_lists = os.listdir(src_dir)
image_lists = ['img_36.jpg']
print(f"[INFO] image_lists = {len(image_lists)}")
for _ in image_lists:
    image_path = f"{src_dir}{_}"
    new_image = mixup_image.get_mixup_image(image_path, url)
    dst_path = f"{dst_dir}mix_{_}"
    mixup_image.im_write(dst_path, new_image, ".jpg")
print("[INFO] end-----------------")
