from bitarray import test
import numpy as np
import shutil

def convert_stylesClassname_to_classname(stylesClassname) -> str:
    return stylesClassname.split('_')[1]

def convert_styles_result_list(styles_result_list) -> list:
    for result in styles_result_list:
        result['pred_class'] = convert_stylesClassname_to_classname(result['pred_class'])
    return styles_result_list

test = [{'pred_class': 'clothesStyles_Solidcolor'}, {'pred_class': 'clothesStyles_multicolour'}, {'pred_class': 'clothesStyles_lattice'}]

print(convert_styles_result_list(test))