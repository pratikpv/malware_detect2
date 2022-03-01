import argparse
from datetime import datetime
from models.models_utils import *
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.Shallow_ML_models import *
import sklearn.metrics
import sys
import traceback


def setup():
    current_time_str = str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S"))
    LOG_DIR = os.path.join(LOG_MASTER_DIR, current_time_str)
    os.makedirs(LOG_DIR)
    return LOG_DIR


def execute_deep_feedforward_model(model_params, LOG_DIR):
    print(f'Model params: {model_params}')

    batch_size = model_params['batch_size']
    feature_type = model_params['feature_type']

    if feature_type == FEATURE_TYPE_IMAGE:
        image_dim = model_params['image_dim']
        conv1d_image_dim_w = -1
        data_path = get_image_datapath(image_dim)
        if image_dim == 0:
            # conv1d models
            conv1d_image_dim_w = model_params['conv1d_image_dim_w']

        print(f'Loading image data')
        train_loader, val_loader, dataset_len, class_names = get_image_data_loaders(data_path=data_path,
                                                                                    image_dim=image_dim,
                                                                                    batch_size=batch_size,
                                                                                    conv1d_image_dim_w=conv1d_image_dim_w)

    else:
        print(f'Loading opcode data')
        opcode_len = model_params['opcode_len']
        data_path = get_opcode_datapath(opcode_len)
        train_loader, val_loader, dataset_len, class_names, \
        text_vocal_len, label_vocab_len, pad_idx = get_opcode_data_loaders(data_path=data_path,
                                                                           opcode_len=opcode_len,
                                                                           batch_size=batch_size)
        model_params['input_dim'] = text_vocal_len
        model_params['output_dim'] = label_vocab_len

    train_set_len = len(train_loader) * batch_size
    val_set_len = len(val_loader) * batch_size
    num_of_classes = len(class_names)

    model_params['num_of_classes'] = num_of_classes
    model_params['class_names'] = class_names

    if feature_type == FEATURE_TYPE_IMAGE:
        model = create_deep_image_model(model_params).to(device)
    else:
        model = create_deep_opcode_model(model_params).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    model, train_losses, train_accuracy = train_ann_model(model=model, model_params=model_params, criterion=criterion,
                                                          train_loader=train_loader, log_dir=LOG_DIR)
    test_accuracy, predicted, ground_truth = test_ann_model(model=model, model_params=model_params, criterion=criterion,
                                                            val_loader=val_loader)

    model_params['train_accuracy'] = np.mean(train_accuracy)
    model_params['test_accuracy'] = np.mean(test_accuracy)

    print(f"Average Train accuracy: {model_params['train_accuracy']:7.4f}%")
    print(f"Average Test accuracy : {model_params['test_accuracy']:7.4f}%")

    save_model_results_to_log(model=model, model_params=model_params,
                              train_losses=train_losses, train_accuracy=train_accuracy,
                              predicted=predicted, ground_truth=ground_truth,
                              log_dir=LOG_DIR)


def execute_deep_rnn_model(model_params, LOG_DIR):
    print(f'Model params: {model_params}')

    batch_size = model_params['batch_size']
    opcode_len = model_params['opcode_len']
    data_path = get_opcode_datapath(opcode_len)

    print_line()
    print(f'Loading Opcode data')

    train_iterator, test_iterator, dataset_len, class_names, \
    text_vocal_len, label_vocab_len, pad_idx = get_opcode_data_loaders(data_path=data_path,
                                                                       opcode_len=opcode_len,
                                                                       batch_size=batch_size)
    num_of_classes = len(class_names)

    print(f'Total images available: {dataset_len}')
    print(f'Number of classes: {num_of_classes}')
    print_line()

    model_params['num_of_classes'] = num_of_classes
    model_params['class_names'] = class_names
    model_params['input_dim'] = text_vocal_len
    model_params['output_dim'] = label_vocab_len

    model = create_deep_opcode_model(model_params).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    model, train_losses, train_accuracy = train_rnn_model(model=model, model_params=model_params, criterion=criterion,
                                                          train_loader=train_iterator, log_dir=LOG_DIR)
    test_accuracy, predicted, ground_truth = test_rnn_model(model=model, model_params=model_params, criterion=criterion,
                                                            val_loader=test_iterator)

    model_params['train_accuracy'] = np.mean(train_accuracy)
    model_params['test_accuracy'] = np.mean(test_accuracy)

    print(f"Average Train accuracy: {model_params['train_accuracy']:7.4f}%")
    print(f"Average Test accuracy : {model_params['test_accuracy']:7.4f}%")

    save_model_results_to_log(model=model, model_params=model_params,
                              train_losses=train_losses, train_accuracy=train_accuracy,
                              predicted=predicted, ground_truth=ground_truth,
                              log_dir=LOG_DIR)


def execute_conv_tl_model(model_params, LOG_DIR):
    print(f'Model params: {model_params}')

    batch_size = model_params['batch_size']
    image_dim = model_params['image_dim']
    data_path = get_image_datapath(image_dim)
    # dataloader transforms input images of image_dim to what pre-trained model expects
    # pretrained_image_dim is what pre-trained model expects
    pretrained_image_dim = get_pretrained_image_dim(model_params['model_name'])
    train_loader, val_loader, dataset_len, class_names = get_image_data_loaders(data_path=data_path,
                                                                                image_dim=image_dim,
                                                                                batch_size=batch_size,
                                                                                convert_to_rgb=True,
                                                                                pretrained_image_dim=pretrained_image_dim)

    train_set_len = len(train_loader) * batch_size
    val_set_len = len(val_loader) * batch_size
    num_of_classes = len(class_names)

    model_params['num_of_classes'] = num_of_classes
    model_params['class_names'] = class_names

    model = create_conv_tl_model(model_params).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    model, train_losses, train_accuracy = train_ann_model(model=model, model_params=model_params, criterion=criterion,
                                                          train_loader=train_loader, log_dir=LOG_DIR)
    test_accuracy, predicted, ground_truth = test_ann_model(model=model, model_params=model_params, criterion=criterion,
                                                            val_loader=val_loader)

    model_params['train_accuracy'] = np.mean(train_accuracy)
    model_params['test_accuracy'] = np.mean(test_accuracy)

    print(f"Average Train accuracy: {model_params['train_accuracy']:7.4f}%")
    print(f"Average Test accuracy : {model_params['test_accuracy']:7.4f}%")

    save_model_results_to_log(model=model, model_params=model_params,
                              train_losses=train_losses, train_accuracy=train_accuracy,
                              predicted=predicted, ground_truth=ground_truth,
                              log_dir=LOG_DIR)


def process_deep_learning(experiment_types, LOG_DIR):
    for expr_type in experiment_types:
        malware_expr_list = get_malware_experiments_list(expr_type)
        print(malware_expr_list)
        total_expr = len(malware_expr_list)

        final_results = []
        for num, ml in enumerate(malware_expr_list):
            if 'num_layers' in ml.keys():
                num_layers = ml['num_layers']
                if num_layers == 1:
                    ml['dropout'] = 0

            print_line()
            print(f'Executing : {ml["experiment_name"]} ({num + 1}/{total_expr})')
            print_line()
            try:
                if expr_type == DEEP_FF:
                    execute_deep_feedforward_model(ml, LOG_DIR)
                if expr_type == DEEP_RNN:
                    execute_deep_rnn_model(ml, LOG_DIR)
            except Exception:
                temp_dict = {'experiment_name': ml['experiment_name'],
                             'train_accuracy': 'failed',
                             'test_accuracy': 'failed'}
                print_line()
                print("FAILED")
                print(traceback.print_exc())
                print_line()
                print(sys.exc_info()[0])
                print_line()
            else:
                temp_dict = {'experiment_name': ml['experiment_name'],
                             'train_accuracy': ml['train_accuracy'],
                             'test_accuracy': ml['test_accuracy']}

            final_results.append(temp_dict)

        exp_results_filename = os.path.join(LOG_DIR, expr_type + '_' + EXPERIMENT_RESULTS)
        df = pd.DataFrame(final_results)
        expr_name = df['experiment_name']
        df.drop(['experiment_name'], axis=1, inplace=True)
        df.set_index(expr_name, drop=True, inplace=True)
        df.to_csv(exp_results_filename)
        save_models_metadata_to_log(malware_expr_list, LOG_DIR)


def prepare_shallow_model(model_params, LOG_DIR):
    print(f'Model params: {model_params}')
    df = pd.read_csv(ORG_DATASET_PE_FEATURES_CSV)

    # sort class names and re-assign the new class IDs w.r.t. sored classes.
    df.sort_values(by=['Malware_ClassName'], inplace=True)
    malware_classes = df['Malware_ClassName'].values
    malware_classes = sorted(list(set(list(malware_classes))))
    new_class_ids = df.apply(lambda row: malware_classes.index(row['Malware_ClassName']), axis=1)
    df['Malware_ClassID'] = new_class_ids

    data = df.drop(['Name', 'md5', 'Malware_ClassName'], axis=1)
    x = data.drop(['Malware_ClassID'], axis=1)
    y = data['Malware_ClassID']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    model_params['num_of_classes'] = len(malware_classes)
    model_params['class_names'] = malware_classes

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model, gsc_model = create_shallow_model(model_params=model_params)

    x_pred, y_pred, best_estimator, best_params = execute_shallow_model(model=gsc_model, x_train=x_train,
                                                                        y_train=y_train, x_test=x_test,
                                                                        model_params=model_params)

    test_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    model_params['train_accuracy'] = gsc_model.cv_results_['mean_train_score'][gsc_model.best_index_]
    model_params['test_accuracy'] = test_accuracy
    print(f"Train accuracy: {model_params['train_accuracy']:7.4f}%")
    print(f"Test accuracy: {model_params['train_accuracy']:7.4f}%")

    save_model_results_to_log(model=gsc_model, model_params=model_params,
                              predicted=y_pred, ground_truth=y_test, best_params=best_params,
                              log_dir=LOG_DIR)


def process_shallow_learning(LOG_DIR):
    shallow_expr_list = get_shallow_expr_list()
    total_expr = len(shallow_expr_list)
    final_results = []
    for num, ml in enumerate(shallow_expr_list):
        print_line()
        print(f'Executing : {ml["experiment_name"]} ({num + 1}/{total_expr})')
        print_line()
        prepare_shallow_model(ml, LOG_DIR)
        temp_dict = {'experiment_name': ml['experiment_name'],
                     'train_accuracy': ml['train_accuracy'],
                     'test_accuracy': ml['test_accuracy']}
        final_results.append(temp_dict)

    exp_results_filename = os.path.join(LOG_DIR, 'shallow_' + EXPERIMENT_RESULTS)
    df = pd.DataFrame(final_results)
    expr_name = df['experiment_name']
    df.drop(['experiment_name'], axis=1, inplace=True)
    df.set_index(expr_name, drop=True, inplace=True)
    df.to_csv(exp_results_filename)
    save_models_metadata_to_log(shallow_expr_list, LOG_DIR)


def process_conv_transfer_learning(LOG_DIR):
    tl_expr_list = get_conv_transfer_learning_expr_list()
    total_expr = len(tl_expr_list)
    final_results = []
    for num, ml in enumerate(tl_expr_list):
        print_line()
        print(f'Executing : {ml["experiment_name"]} ({num + 1}/{total_expr})')
        print_line()
        try:
            execute_conv_tl_model(ml, LOG_DIR)
        except Exception:
            temp_dict = {'experiment_name': ml['experiment_name'],
                         'train_accuracy': 'failed',
                         'test_accuracy': 'failed'}
            print_line()
            print("FAILED")
            print(traceback.print_exc())
            print_line()
            print(sys.exc_info()[0])
            print_line()
        else:
            temp_dict = {'experiment_name': ml['experiment_name'],
                         'train_accuracy': ml['train_accuracy'],
                         'test_accuracy': ml['test_accuracy']}
        final_results.append(temp_dict)

    exp_results_filename = os.path.join(LOG_DIR, 'conv_tl_' + EXPERIMENT_RESULTS)
    df = pd.DataFrame(final_results)
    expr_name = df['experiment_name']
    df.drop(['experiment_name'], axis=1, inplace=True)
    df.set_index(expr_name, drop=True, inplace=True)
    df.to_csv(exp_results_filename)
    save_models_metadata_to_log(tl_expr_list, LOG_DIR)


def main(args, LOG_DIR):
    deep_learning_models = []
    if args.deep_feedforward:
        deep_learning_models.append(DEEP_FF)
    if args.deep_rnn:
        deep_learning_models.append(DEEP_RNN)

    if len(deep_learning_models) > 0:
        print_line()
        print(f'Starting Deep Learning Experiments to detect Malwares')
        print_line()
        process_deep_learning(deep_learning_models, LOG_DIR)
        print_line()

    if args.shallow_ml:
        print(f'Starting shallow Machine Learning Experiments to detect Malwares')
        print_line()
        process_shallow_learning(LOG_DIR)
        print_line()

    if args.transfer_conv_ml:
        print(f'Starting conv-based Transfer Learning Experiments to detect Malwares')
        print_line()
        process_conv_transfer_learning(LOG_DIR)
        print_line()


def print_banner(LOG_DIR):
    print_line()
    if use_cuda:
        print('Using GPU:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('Running on :', device)

    print(f'LOG_DIR = {LOG_DIR}')
    print_line()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Machine Learning models to detect and classify Malware')

    parser.add_argument('--deep_feedforward', action='store_true', help='Execute deep feedforward models',
                        default=False)
    parser.add_argument('--deep_rnn', action='store_true', help='Execute deep rnn models',
                        default=False)
    parser.add_argument('--shallow_ml', action='store_true', help='Execute shallow machine learning models',
                        default=False)
    parser.add_argument('--transfer_conv_ml', action='store_true', help='Transfer learning using conv-based models',
                        default=False)
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    LOG_DIR = setup()
    print_banner(LOG_DIR)
    main(args, LOG_DIR)
    print_banner(LOG_DIR)
