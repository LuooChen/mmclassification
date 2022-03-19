from mmcls.apis import inference_multi_label_model

# 'upperLength'
upperLength_classes = ['LongSleeve', 'ShortSleeve', 'NoSleeve']
# 'clothesStyles'
clothesStyles_classes = ['Solidcolor', 'multicolour', 'lattice']
# 'hairStyles'
hairStyles_classes = ['Long', 'middle', 'Short', 'Bald']

# clothesStyles
clothesStyles_Solidcolor = 'Solidcolor'
clothesStyles_multicolour = 'multicolour'
clothesStyles_lattice = 'lattice'

# score_threshold
score_threshold = 0.3
# upper_colors filter threshold
filter_thr = 0.5

def get_max_item_of_list(list) -> dict:
    return max(list, key=lambda item: item['pred_scores'])

def sorted_by_pred_scores(list):
    list.sort(key=lambda x: x['pred_scores'], reverse=True)
    
def get_colors_result_for_top2_or_top3(colors_result) -> list:
    if colors_result[2]['pred_scores'] > score_threshold:
        # three colors
        return colors_result
    else:
        # two colors
        return colors_result[:2]

def infer_upper(model, img) -> dict:
    pred_result = inference_multi_label_model(model, img)
    pred_upperLength = pred_result[:3]
    pred_clothesStyles = pred_result[3:6]
    pred_hairStyles = pred_result[6:]
    return {
        'upperLength': get_max_item_of_list(pred_upperLength),
        'clothesStyles': get_max_item_of_list(pred_clothesStyles),
        'hairStyles': get_max_item_of_list(pred_hairStyles)
    }

def infer_upper_colors_top3(model, img) -> list:
    pred_result = inference_multi_label_model(model, img)
    sorted_by_pred_scores(pred_result)
    return pred_result[:3]

def get_upper_colors_result_by_clothesStyles(clothesStyles, upper_colors_result) -> list:
    if clothesStyles == clothesStyles_Solidcolor:
        return [get_max_item_of_list(upper_colors_result)]
    else:
        # multicolour or lattice
        return get_colors_result_for_top2_or_top3(upper_colors_result)

def get_upper_colors_result_by_thr(upper_colors_result) -> list:
    filtered_result = [x for x in upper_colors_result if x['pred_scores'] >= filter_thr]
    if len(filtered_result) >= 1:
        return filtered_result
    else:
        # at least return the top1
        return [upper_colors_result[0]]

def infer_upper_info(model_upper, model_upper_colors, img):
    upper_result = infer_upper(model_upper, img)
    upper_colors_result = infer_upper_colors_top3(model_upper_colors, img)
    upper_colors_result = get_upper_colors_result_by_clothesStyles(upper_result['clothesStyles']['pred_class'], upper_colors_result)
    # upper_colors_result = get_upper_colors_result_by_thr(upper_colors_result)
    upper_result['upper_colors'] = upper_colors_result
    return upper_result