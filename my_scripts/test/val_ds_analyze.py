import pandas as pd

def apply_color_count(series):
    count = 0
    for i in colors:
        if series[i] > 0:
            count = count + 1
    return count

upper_colors_val = pd.read_csv('data/labels/upper_colors_val.csv') # reading the csv file
color_count_df = upper_colors_val.copy(deep=True)
# print(color_count_df.head())
colors = ['upperBlack',
       'upperBrown', 'upperBlue', 'upperGreen', 'upperGray', 'upperOrange',
       'upperPink', 'upperPurple', 'upperRed', 'upperWhite', 'upperYellow']

color_count_df["color_count"] = color_count_df.apply(apply_color_count, axis=1)

color_count_df = color_count_df.drop(['upperLength', 'hairStyles', 'upperBlack',
       'upperBrown', 'upperBlue', 'upperGreen', 'upperGray', 'upperOrange',
       'upperPink', 'upperPurple', 'upperRed', 'upperWhite', 'upperYellow'], axis=1)

def get_val_ds():
    lattice_ds = {
        'two': [],
        'three':[]
    }
    multicolour_ds = {
        'two': [],
        'three':[]
    }
    for index, row in color_count_df.iterrows():
        if row['clothesStyles'] == 'lattice':
            if row['color_count'] == 2:
                lattice_ds['two'].append(row['name'])
            else:
                lattice_ds['three'].append(row['name'])
        if row['clothesStyles'] == 'multicolour':
            if row['color_count'] == 2:
                multicolour_ds['two'].append(row['name'])
            else:
                multicolour_ds['three'].append(row['name'])
    return [lattice_ds, multicolour_ds]

if __name__ == "__main__":
    lattice_ds, multicolour_ds = get_val_ds()
    print('lattice_ds two samples len: ', len(lattice_ds['two']))
    print('lattice_ds three samples len: ', len(lattice_ds['three']))
    print('multicolour_ds two samples len: ', len(multicolour_ds['two']))
    print('multicolour_ds three samples len: ', len(multicolour_ds['three']))
