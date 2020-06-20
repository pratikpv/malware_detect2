import os
from tqdm import tqdm
import pandas as pd
from utils import *
import sys
import traceback


def count_dataset(root_data_dir, csv_filename):
    class_names = os.listdir(root_data_dir)
    class_count = len(class_names)

    csv_delimiter = ','
    count_columns = ['class_name', 'total_samples']
    count_csv_file = open(csv_filename, "w")

    count_csv_file.write(csv_delimiter.join(count_columns) + "\n")

    for class_id in tqdm(range(class_count), desc='Counting dataset'):
        files = os.listdir(os.path.join(root_data_dir, class_names[class_id]))
        count_csv_file.write(class_names[class_id] + csv_delimiter + str(len(files)) + "\n")

    count_csv_file.close()


def process_cf_for_latex(data_list):
    data_path = os.path.join('logs', 'logs_preserve', 'logs_only')
    cf_file = 'confusion_matrix.csv'
    cf_file_latex = 'confusion_matrix_latex.txt'

    for date_dir, expr_name in data_list:
        cf_filepath = os.path.join(data_path, date_dir, expr_name, cf_file)
        print_line()
        print(f'Processing {cf_filepath}', end='')
        try:
            df = pd.read_csv(cf_filepath, index_col=0)
            df = df.apply(lambda x: x / x.sum(), axis=1)  # normalize by row
            df_s = len(df.columns)

            cf_file_latexpath = os.path.join(data_path, date_dir, expr_name, cf_file_latex)
            with open(cf_file_latexpath, 'w+') as f:
                for r in range(df_s):
                    for c in range(df_s):
                        cell_val = round(df.iloc[r, c], 4)
                        msg = '{c} {r} {cell_val}\n'.format(c=c, r=r, cell_val=cell_val)
                        f.write(msg)
        except:
            print(traceback.print_exc())
            print_line()
            print(sys.exc_info()[0])
            print_line()
            print(f'\t--> failed')
        else:
            print(f'\t--> success')
            print(cf_file_latexpath)
        print_line()