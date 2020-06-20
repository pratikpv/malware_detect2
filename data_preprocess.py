import argparse
import sys
from config import *
from data_utils.extract_pe_features import *
from data_utils.bin_to_img import *
from data_utils.extract_opcode import *
from data_utils.misc import *
from data_utils.data_loaders import *


def main():
    max_files = 0  # set 0 to process all files or set a specific number

    if args.extract_pe_features:
        extract_pe_features(ORG_DATASET_PE_FEATURES_CSV, ORG_DATASET_COUNT_PE_FEATURES_CSV, ORG_DATASET_PATH,
                            max_files=max_files)

    if args.bin_to_img:
        list_of_widths = [0, 1, 64, 128, 256, 512, 1024]
        for width in list_of_widths:
            convert_bin_to_img(ORG_DATASET_PATH, width, max_files=max_files)

    if args.extract_opcodes:
        process_opcodes_bulk(ORG_DATASET_PATH, max_files=max_files)

    if args.count_samples:
        count_dataset(ORG_DATASET_PATH, ORG_DATASET_COUNT_CSV)
        count_dataset(ORG_DATASET_OPCODES_PATH, ORG_DATASET_COUNT_OPCODES_PATH)
        count_dataset(get_image_datapath(image_dim=256), ORG_DATASET_COUNT_IMAGES_CSV)

    if args.split_opcodes:
        list_of_opcode_lens = [10, 20, 50, 100, 500, 1000, 2000, 5000]
        for opcode_len in list_of_opcode_lens:
            process_split_opcodes(ORG_DATASET_OPCODES_PATH, opcode_len=opcode_len)

    if args.latex_format:
        # tuple -> log_date_dir , experiment
        data_list = [("25-May-2020_22_44_37", "experiment_14"),
                     ("13-Jun-2020_16_49_09", "experiment_29"),
                     ("09-Jun-2020_20_42_39", "conv1d_experiment_65"),
                     ("14-Jun-2020_09_03_12", "experiment_18"),
                     ("06-Jun-2020_22_13_17", "rnn_experiment_22"),
                     ("06-Jun-2020_22_13_17", "rnn_experiment_46"),
                     ("13-Jun-2020_21_04_18", "tl_experiment_1"),
                     ("12-Jun-2020_21_54_44", "XGB_experiment_1"),
                     ("12-Jun-2020_21_54_44", "Knn_experiment_1"),
                     ("12-Jun-2020_21_54_44", "RandomForest_experiment_1")
                     ]
        process_cf_for_latex(data_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the Malware data')

    parser.add_argument('--extract_pe_features', action='store_true', help='Extract features from PE format',
                        default=False)
    parser.add_argument('--bin_to_img', action='store_true', help='Generate image files from malware binaries',
                        default=False)
    parser.add_argument('--extract_opcodes', action='store_true', help='Extract opcodes from malware binaries',
                        default=False)
    parser.add_argument('--count_samples', action='store_true', help='Count all sample files for all experiments',
                        default=False)
    parser.add_argument('--split_opcodes', action='store_true', help='split opcodes into train-test for TorchText',
                        default=False)

    parser.add_argument('--latex_format', action='store_true', help='Normalize Conf. matrix and save for latex',
                        default=False)

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    main()
