import random
import mixup
import mosiac
import yoco
import pandas as pd

# 图片路径(不包括图片)
image_dir = ""
# 原标签文件csv
csv_file = ""
data = pd.read_csv(csv_file)
datas = data.values.tolist()
final_lists = list()
# 图片名字+类型的字典
result_dict = {
    "A": list(),
    "B": list(),
    "C": list(),
    "D": list()
}
for _ in datas:
    temp_dict = {
        "id": _[0],
        "type": _[1]
    }
    # 往最终结果的list里面添加数据
    final_lists.append(temp_dict)
    result_dict[_[1]].append(temp_dict)

A_lists = result_dict["A"] 
B_lists = result_dict["B"] 
C_lists = result_dict["C"] 
D_lists = result_dict["D"]

# 根据1：1：1分别mosiac、mixup、yoco
def get_broaden_num(max_num, rate_tup):
    nums = rate_tup[0] + rate_tup[1] + rate_tup[2]
    min_num = max_num // nums
    new_list = list()
    new_list.append(rate_tup[0] * min_num)
    new_list.append(rate_tup[1] * min_num)
    new_list.append(max_num - rate_tup[0] * min_num - rate_tup[1] * min_num)
    return new_list

"""
    board_list：增广比率
    board_type: B, C, D
    image_list: 增广随机取的图片
"""
def broaden_data(board_list, board_type, image_list, final_lists):
    mosiac_num = board_list[0]
    mixup_num = board_list[1]
    yoco_num = board_list[2]
    # mosiac增广
    for _ in range(0, mosiac_num):
        temp_dict1 = random.choice(image_list)
        temp_dict2 = random.choice(image_list)
        img_nm1 = temp_dict1["id"]
        src_image_path = f"{image_dir}{img_nm1}"
        img_nm2 = temp_dict2["id"]
        dst_image_path = f"{image_dir}{img_nm2}"
        new_image_path = f"{image_dir}mosiac_{board_type}_{_}.jpg"
        mosiac.mosiac(src_image_path, dst_image_path, new_image_path)
        # 保存标签
        final_dict = {
            "id": f"mosiac_{board_type}_{_}.jpg",
            "type": board_type
        }
        final_lists.append(final_dict)
    # mixup增广
    for _ in range(0, mixup_num):
        temp_dict1 = random.choice(image_list)
        temp_dict2 = random.choice(image_list)
        img_nm1 = temp_dict1["id"]
        src_image_path = f"{image_dir}{img_nm1}"
        img_nm2 = temp_dict2["id"]
        dst_image_path = f"{image_dir}{img_nm2}"
        new_image_path = f"{image_dir}mixup_{board_type}_{_}.jpg"
        mixup.mixup(src_image_path, dst_image_path, new_image_path)
        # 保存标签
        final_dict = {
            "id": f"mixup_{board_type}_{_}.jpg",
            "type": board_type
        }
        final_lists.append(final_dict)
    # yoco增广
    """
        因为yoco是一张变两张,所以先取整
    """
    new_yoco_num = yoco_num // 2
    index_num = 0
    for _ in range(0, new_yoco_num):
        temp_dict1 = random.choice(image_list)
        img_nm1 = temp_dict1["id"]
        src_image_path = f"{image_dir}{img_nm1}"
        if _ != 0:
            index_num = index_num + 1
        new_image_path1 = f"{image_dir}yoco_{board_type}_{index_num}.jpg"
        # 保存标签
        final_dict1 = {
            "id": f"yoco_{board_type}_{index_num}.jpg",
            "type": board_type
        }
        final_lists.append(final_dict1)
        index_num = index_num + 1
        new_image_path2 = f"{image_dir}yoco_{board_type}_{index_num}.jpg"
        yoco.yoco(src_image_path, new_image_path1, new_image_path2)
        # 保存标签
        final_dict2 = {
            "id": f"yoco_{board_type}_{index_num}.jpg",
            "type": board_type
        }
        final_lists.append(final_dict2)

# 执行增广
# board rate
rate_tup = (1, 1, 1)
# board B data
B_board_list = get_broaden_num(929, rate_tup)
broaden_data(B_board_list, "B", B_lists, final_lists)
print("[INFO] boarden B---end")
# board C data
C_board_list = get_broaden_num(414, rate_tup)
broaden_data(C_board_list, "C", C_lists, final_lists)
print("[INFO] boarden C---end")
# board D data
D_board_list = get_broaden_num(909, rate_tup)
broaden_data(D_board_list, "D", D_lists, final_lists)
print("[INFO] boarden D---end")
label_dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}
# 生成一份txt全部标签
output_path = ""
with open(output_path,'w') as file:
    for _ in final_lists:
            id = _["id"]
            car_type = _["type"]
            label = label_dict[car_type]
            result_label = f"{id} {label}\n"
            file.write(result_label)
      
# 生成一份所有标签的csv
pf = pd.DataFrame(list(final_lists))
output_csv_path = ""
pf.to_csv(output_csv_path, encoding='utf-8', index=False)
print("[INFO] ---end")