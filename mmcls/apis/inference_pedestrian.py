from mmcls.apis import inference_multi_label_model

# 'upperLength'
upperLength_classes = ['LongSleeve', 'ShortSleeve', 'NoSleeve']
# 'clothesStyles'
clothesStyles_classes = ['Solidcolor', 'multicolour', 'lattice']
# 'hairStyles'
hairStyles_classes = ['Long', 'middle', 'Short', 'Bald']

def get_max_item_of_list(list) -> dict:
    return max(list, key=lambda item: item['pred_scores'])

def infer_upper(model, img):
    pred_result = inference_multi_label_model(model, img)
    pred_upperLength = pred_result[:3]
    pred_clothesStyles = pred_result[3:6]
    pred_hairStyles = pred_result[6:]
    return {
        'upperLength': get_max_item_of_list(pred_upperLength),
        'clothesStyles': get_max_item_of_list(pred_clothesStyles),
        'hairStyles': get_max_item_of_list(pred_hairStyles)
    }