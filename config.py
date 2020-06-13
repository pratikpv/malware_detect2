import os
import torch
import multiprocessing

# original dataset i.e. binary executable files
ORG_DATASET_ROOT_PATH = os.path.join('data', 'exec_files')
ORG_DATASET_DIR_NAME = 'org_dataset'
ORG_DATASET_PATH = os.path.join('data', 'exec_files', ORG_DATASET_DIR_NAME)
supported_image_dims = [0, 1, 64, 128, 256, 512, 1024]

# opcodes from binary executables
# -1: the entire opcode len of the binary, else the specific opcode len
supported_opcode_lens = [-1, 10, 20, 50, 100, 500, 1000, 2000, 5000]
ORG_DATASET_OPCODES_PATH = os.path.join('data', 'exec_files', 'org_dataset_opcodes')
# PE features from binary executables
ORG_DATASET_PE_FEATURES_CSV = os.path.join('data', 'org_malware_dataset_pe_features.csv')

# count files for ORG_DATASET_PATH
ORG_DATASET_COUNT_CSV = os.path.join('data', 'org_malware_dataset_count.csv')
# count files for ORG_DATASET_PATH_IMAGE*
ORG_DATASET_COUNT_IMAGES_CSV = os.path.join('data', 'org_malware_dataset_count_images.csv')
# count files for ORG_DATASET_OPCODES_PATH
ORG_DATASET_COUNT_PE_FEATURES_CSV = os.path.join('data', 'org_malware_dataset_count_pe_features.csv')
# count files for ORG_DATASET_PE_FEATURES_CSV
ORG_DATASET_COUNT_OPCODES_PATH = os.path.join('data', 'org_malware_dataset_count_opcodes.csv')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
LINE_LEN = 80
LOG_MASTER_DIR = 'logs'
MODEL_INFO_LOG = 'model_info_and_results.log'
MODEL_META_INFO_LOG = 'models_meta_info.log'
MODEL_LOSS_INFO_LOG = 'model_losses.log'
MODEL_ACC_INFO_LOG = 'model_acc.log'
MODEL_CONF_MATRIX_CSV = 'confusion_matrix.csv'
MODEL_CONF_MATRIX_PNG = 'confusion_matrix.png'
MODEL_CONF_MATRIX_NORMALIZED_CSV = 'confusion_matrix_normalized.csv'
MODEL_CONF_MATRIX_NORMALIZED_PNG = 'confusion_matrix_normalized.png'
MODEL_ACCURACY_PNG = 'model_accuracy.png'
MODEL_LOSS_PNG = 'model_loss.png'
EXPERIMENT_RESULTS = 'experiments_results.csv'
GRID_CV_EXPERIMENT_RESULTS = 'grid_cv_experiments_results.csv'
CPU_COUNT = multiprocessing.cpu_count()

DEEP_FF = 'deep_feed_forward'
DEEP_RNN = 'rnn'
SHALLOW_ML = 'shallow_ml'
FEATURE_TYPE_IMAGE = 'image'
FEATURE_TYPE_OPCODE = 'opcode'
