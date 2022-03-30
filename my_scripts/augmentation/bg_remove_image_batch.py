from base64 import encode
import bg_remove_image
import os
from csv import DictReader
import csv



def csv_to_dict(filename):
    try:
        with open(filename, 'r') as read_obj:
            dict_reader = DictReader(read_obj)
            list_of_dict = list(dict_reader)
        return list_of_dict
    except IOError as err:
        print("I/O error({0})".format(err))




# 目标目录
src_dir = "/home/superdisk/pedestrian-fine-recognition/data/train22/train2_new"
src_csv_file_path = "/home/superdisk/pedestrian-fine-recognition/data/train22/train2_new.csv"
# 生成图片目录
dst_dir = "/home/superdisk/pedestrian-fine-recognition/data/augmentation/train2_new"
dst_csv_file_path = "/home/superdisk/pedestrian-fine-recognition/data/augmentation/train2_new.csv"

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
    
frames = csv_to_dict(src_csv_file_path)

# 类别极不平衡类别名
badcases = {
    "upperLength-NoSleeve":list(), 
    "clothesStyles-lattice":list(), 
    "hairStyles-Bald":list(), 
    "lowerLength-Shorts":list(), 
    "lowerStyles-lattice":list(), 
    "lowerStyles-multicolour":list(), 
    "shoesStyles-else":list()
}

for badcase_key, badcase_value in badcases.items():
    classhead, name = badcase_key.split("-")
    paths = badcase_value
    # print(classhead, name, paths)
    for frame in frames:
        for frame_key, frame_value in frame.items():
            if frame_key == classhead and frame_value == name:
                # print(frame)
                badcase_value.append(frame["name"])

# 把不平衡类别对应的路径收集完成，验证数量是否正确
for badcase_key, badcase_value in badcases.items():
    print(badcase_key, len(badcase_value))

# api url
url = 'http://10.191.20.112:3000/api/imageRemoveBg'

output_list = list()
output_list.append(("name", "path"))
# output_list.to_csv(dst_csv_file_path, encoding="utf-8")

for name, paths in badcases.items():
    for path in paths:
        src_path = os.path.join(src_dir, path)
        new_image = bg_remove_image.get_mixup_image(src_path, url)
        dst_path = os.path.join(dst_dir, path)
        bg_remove_image.im_write(dst_path, new_image, ".jpg")
        output_list.append((name, dst_path))
        print(name, src_path)
        exit()

with open(dst_csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_list)
# for _ in image_lists:
#     image_path = f"{src_dir}{_}"
#     new_image = mixup_image.get_mixup_image(image_path, url)
#     dst_path = f"{dst_dir}mix_{_}"
#     mixup_image.im_write(dst_path, new_image, ".jpg")
# print("[INFO] end-----------------")
