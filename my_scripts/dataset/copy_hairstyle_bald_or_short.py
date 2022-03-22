import csv
import os
import shutil

testA_img_root_path = 'data/testA/'
testA_pred_result_filepath = 'data/submit/result_6_best.csv'

bald_or_short_root_path = 'data/manual_annotation/bald_or_short/'
bald_or_short_labels_filepath = 'data/manual_annotation/testA_pred_bald_or_short.csv'

upper_labels_data_info = []
with open(testA_pred_result_filepath) as f:
    reader = csv.DictReader(f)
    upper_labels_header = reader.fieldnames
    # print('upper_labels_header: ', upper_labels_header)
    for (index, row) in enumerate(reader):
        if row['hairStyles'] == 'Short' or row['hairStyles'] == 'Bald':
            upper_labels_data_info.append(row)
 
def generate_csv_file(datainfo, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = upper_labels_header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in datainfo:
            writer.writerow(row)

if __name__ == "__main__":
    # generate file
    generate_csv_file(upper_labels_data_info, bald_or_short_labels_filepath)
    print('generate csv successfully.')
    for row in upper_labels_data_info:
        shutil.copyfile(os.path.join(testA_img_root_path, row['name']), os.path.join(bald_or_short_root_path, row['name']))
    print('copy img successfullly.')