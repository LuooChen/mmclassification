import csv
import os
import shutil

testA_img_root_path = 'data/testA/'
testA_pred_result_filepath = 'data/submit/result_6_best.csv'

target_root_path = 'data/manual_annotation/nosleeve_or_shortsleeve/'
target_labels_filepath = 'data/manual_annotation/testA_pred_nosleeve_or_shortsleeve.csv'

upper_labels_data_info = []
with open(testA_pred_result_filepath) as f:
    reader = csv.DictReader(f)
    upper_labels_header = reader.fieldnames
    # print('upper_labels_header: ', upper_labels_header)
    for (index, row) in enumerate(reader):
        if row['upperLength'] == 'NoSleeve' or row['upperLength'] == 'ShortSleeve':
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
    generate_csv_file(upper_labels_data_info, target_labels_filepath)
    print('generate csv successfully.')
    for row in upper_labels_data_info:
        shutil.copyfile(os.path.join(testA_img_root_path, row['name']), os.path.join(target_root_path, row['name']))
    print('copy img successfullly.')