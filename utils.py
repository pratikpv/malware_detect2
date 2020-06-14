from config import *
import os
import pickle
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn as nn
import sys
from sklearn.model_selection import GridSearchCV
import json

def save_model_results_to_log(model=None, model_params=None,
                              train_losses=None, train_accuracy=None,
                              predicted=None, ground_truth=None, best_params=None,
                              misc_data=None, log_dir=None):
    print('Saving model results', end='')
    experiment_name = model_params['experiment_name']
    model_name = model_params['model_name']
    num_of_classes = model_params['num_of_classes']
    class_names = model_params['class_names']

    model_log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(model_log_dir, exist_ok=True)
    model_log_file = os.path.join(model_log_dir, MODEL_INFO_LOG)
    model_train_losses_log_file = os.path.join(model_log_dir, MODEL_LOSS_INFO_LOG)
    model_train_accuracy_log_file = os.path.join(model_log_dir, MODEL_ACC_INFO_LOG)
    model_save_path = os.path.join(model_log_dir, model_name + '.pt')
    model_conf_mat_csv = os.path.join(model_log_dir, MODEL_CONF_MATRIX_CSV)
    model_conf_mat_png = os.path.join(model_log_dir, MODEL_CONF_MATRIX_PNG)
    model_conf_mat_normalized_csv = os.path.join(model_log_dir, MODEL_CONF_MATRIX_NORMALIZED_CSV)
    model_conf_mat_normalized_png = os.path.join(model_log_dir, MODEL_CONF_MATRIX_NORMALIZED_PNG)

    model_loss_png = os.path.join(model_log_dir, MODEL_LOSS_PNG)
    model_accuracy_png = os.path.join(model_log_dir, MODEL_ACCURACY_PNG)

    grid_cv_filepath = os.path.join(model_log_dir,GRID_CV_EXPERIMENT_RESULTS)
    print('.', end='')
    # generate and save confusion matrix
    plot_x_label = "Predictions"
    plot_y_label = "Actual"
    cmap = plt.cm.Blues
    pred_class_indexes = sorted(np.unique(predicted))
    pred_num_classes = len(pred_class_indexes)
    target_class_names = [class_names[i] for i in pred_class_indexes]

    cm = metrics.confusion_matrix(ground_truth, predicted)

    print('.', end='')
    df_confusion = pd.DataFrame(cm)
    df_confusion.index = target_class_names
    df_confusion.columns = target_class_names
    df_confusion.round(2)
    df_confusion.to_csv(model_conf_mat_csv)
    fig = plt.figure(figsize=(20, 20))
    sns.heatmap(df_confusion, annot=True, cmap=cmap)
    plt.xlabel(plot_x_label)
    plt.ylabel(plot_y_label)
    plt.title('Confusion Matrix')
    plt.savefig(model_conf_mat_png)
    plt.close(fig)

    print('.', end='')
    cm = metrics.confusion_matrix(ground_truth, predicted, normalize='all')
    df_confusion = pd.DataFrame(cm)
    df_confusion.index = target_class_names
    df_confusion.columns = target_class_names
    df_confusion.round(2)
    df_confusion.to_csv(model_conf_mat_normalized_csv)
    fig = plt.figure(figsize=(20, 20))
    sns.heatmap(df_confusion, annot=True, cmap=cmap)
    plt.xlabel(plot_x_label)
    plt.ylabel(plot_y_label)
    plt.title('Normalized Confusion Matrix')
    plt.savefig(model_conf_mat_normalized_png)
    plt.close(fig)

    if train_losses is not None:
        print('.', end='')
        fig = plt.figure(figsize=(8, 8))
        plt.plot(train_losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(model_loss_png)
        plt.close(fig)

        print('.', end='')
        # save model training stats
        with open(model_train_losses_log_file, 'wb') as file:
            pickle.dump(train_losses, file)
            file.flush()

    if train_accuracy is not None:
        print('.', end='')
        fig = plt.figure(figsize=(8, 8))
        plt.plot(train_accuracy, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.savefig(model_accuracy_png)
        plt.close(fig)

        print('.', end='')
        with open(model_train_accuracy_log_file, 'wb') as file:
            pickle.dump(train_accuracy, file)
            file.flush()

    print('.', end='')
    report = metrics.classification_report(ground_truth, predicted, target_names=list(target_class_names))

    if not isinstance(model, nn.Module):
        cv_df = pd.DataFrame(model.cv_results_)
        cv_df.to_csv(grid_cv_filepath)

    # save model arch and params
    with open(model_log_file, 'a') as file:
        file.write('-' * LINE_LEN + '\n')
        file.write('model architecture' + '\n')
        file.write('-' * LINE_LEN + '\n')
        file.write(str(model) + '\n')
        file.write('-' * LINE_LEN + '\n')
        file.write('model params' + '\n')
        file.write('-' * LINE_LEN + '\n')
        file.write(str(model_params) + '\n')
        file.write('-' * LINE_LEN + '\n')
        if not isinstance(model, nn.Module):
            file.write('GridSearchCV results' + '\n')
            if isinstance(model, GridSearchCV):
                file.write(str(model.cv_results_) + '\n')
        file.write('-' * LINE_LEN + '\n')

        if misc_data:
            file.write('misc data: ' + misc_data + '\n')
            file.write('-' * LINE_LEN + '\n')

        if best_params is not None:
            file.write('best params of the grid search' + '\n')
            file.write('-' * LINE_LEN + '\n')
            file.write(str(best_params) + '\n')
            file.write('-' * LINE_LEN + '\n')

        file.write('classification report' + '\n')
        file.write('-' * LINE_LEN + '\n')
        file.write(report + '\n')
        file.write('-' * LINE_LEN + '\n')
        file.flush()

    print('.', end='')
    # save model as pytorch state dict
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), model_save_path)
    else:
        # save model to file
        pickle.dump(model, open(model_save_path, "wb"))

    print('Done')
    sys.stdout.flush()


def save_models_metadata_to_log(list_of_model_params, LOG_DIR, logfile=MODEL_META_INFO_LOG):
    logfile = os.path.join(LOG_DIR, logfile)
    with open(logfile, 'a') as file:
        file.write('-' * LINE_LEN + '\n')
        for i in list_of_model_params:
            file.write(str(i) + '\n')
        file.write('-' * LINE_LEN + '\n')
        file.flush()


def print_line(print_len=LINE_LEN):
    print('-' * print_len)
