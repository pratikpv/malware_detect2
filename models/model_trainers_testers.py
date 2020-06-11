import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from config import *
import numpy as np
from models.CnnModels import *
from data_utils.data_loaders import *
from utils import *
from tqdm import tqdm


def train_ann_model(model=None, model_params=None, criterion=None,
                    train_loader=None, log_dir=None):
    epochs = model_params['epochs']
    lr = model_params['lr']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracy = []

    tqdm_train_descr_format = "Training Feed-Forward model: Epoch Accuracy = {:02.4f}%, Loss = {:.8f}"
    tqdm_train_descr = tqdm_train_descr_format.format(0, float('inf'))
    tqdm_train_obj = tqdm(range(epochs), desc=tqdm_train_descr)

    model.train(True)

    for i in tqdm_train_obj:

        epoch_corr = 0
        epoch_loss = 0
        total_samples = 0

        for b, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            epoch_corr += batch_corr.item()
            epoch_loss += loss.item()
            total_samples += y_pred.shape[0]

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_accuracy = epoch_corr * 100 / total_samples
        epoch_loss = epoch_loss / total_samples

        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        tqdm_descr = tqdm_train_descr_format.format(epoch_accuracy, epoch_loss)
        tqdm_train_obj.set_description(tqdm_descr)

    return model, train_losses, train_accuracy


def test_ann_model(model=None, model_params=None, criterion=None,
                   val_loader=None):
    tqdm_test_descr_format = "Testing Feed-Forward model: Batch Accuracy = {:02.4f}%"
    tqdm_test_descr = tqdm_test_descr_format.format(0)
    tqdm_test_obj = tqdm(val_loader, desc=tqdm_test_descr)
    num_of_batches = len(val_loader)

    model.eval()

    total_test_loss = 0
    total_test_acc = 0
    predicted_all = torch.tensor([], dtype=torch.long, device=device)
    ground_truth_all = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(tqdm_test_obj):
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            predictions = model(X_test)
            loss = criterion(predictions, y_test)

            predicted = torch.max(predictions.data, 1)[1]
            batch_corr = (predicted == y_test).sum()
            batch_acc = batch_corr.item() * 100 / predictions.shape[0]
            total_test_acc += batch_acc
            total_test_loss += loss.item()

            predicted_all = torch.cat((predicted_all, predicted), 0)
            ground_truth_all = torch.cat((ground_truth_all, y_test), 0)

            tqdm_test_descr = tqdm_test_descr_format.format(batch_acc)
            tqdm_test_obj.set_description(tqdm_test_descr)

    predicted_all = predicted_all.cpu().numpy()
    ground_truth_all = ground_truth_all.cpu().numpy()
    total_test_acc = total_test_acc / num_of_batches

    return total_test_acc, predicted_all, ground_truth_all


def train_rnn_model(model=None, model_params=None, criterion=None,
                    train_loader=None, log_dir=None):
    epochs = model_params['epochs']
    lr = model_params['lr']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracy = []

    tqdm_train_descr_format = "Training RNN model: Epoch Accuracy = {:02.4f}%, Loss = {:.8f}"
    tqdm_train_descr = tqdm_train_descr_format.format(0, float('inf'))
    tqdm_train_obj = tqdm(range(epochs), desc=tqdm_train_descr)

    model.train(True)

    for i in tqdm_train_obj:

        epoch_corr = 0
        epoch_loss = 0
        total_samples = 0
        for b, batch in enumerate(train_loader):
            batch.text = batch.text.to(device)
            batch.label = batch.label.to(device)

            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)

            predicted = torch.max(predictions.data, 1)[1]
            batch_corr = (predicted == batch.label).sum()
            epoch_corr += batch_corr.item()
            epoch_loss += loss.item()
            total_samples += predictions.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_accuracy = epoch_corr * 100 / total_samples
        epoch_loss = epoch_loss / total_samples

        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        tqdm_descr = tqdm_train_descr_format.format(epoch_accuracy, epoch_loss)
        tqdm_train_obj.set_description(tqdm_descr)

    return model, train_losses, train_accuracy


def test_rnn_model(model=None, model_params=None, criterion=None,
                   val_loader=None):
    tqdm_test_descr_format = "Testing RNN model: Batch Accuracy = {:02.4f}%"
    tqdm_test_descr = tqdm_test_descr_format.format(0)
    tqdm_test_obj = tqdm(val_loader, desc=tqdm_test_descr)
    num_of_batches = len(val_loader)

    model.eval()

    total_test_loss = 0
    total_test_acc = 0
    predicted_all = torch.tensor([], dtype=torch.long, device=device)
    ground_truth_all = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        for b, batch in enumerate(tqdm_test_obj):
            batch.text = batch.text.to(device)
            batch.label = batch.label.to(device)

            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)

            predicted = torch.max(predictions.data, 1)[1]
            batch_corr = (predicted == batch.label).sum()
            batch_acc = batch_corr.item() * 100 / predictions.shape[0]
            total_test_acc += batch_acc
            total_test_loss += loss.item()

            predicted_all = torch.cat((predicted_all, predicted), 0)
            ground_truth_all = torch.cat((ground_truth_all, batch.label), 0)

            tqdm_test_descr = tqdm_test_descr_format.format(batch_acc)
            tqdm_test_obj.set_description(tqdm_test_descr)

    predicted_all = predicted_all.cpu().numpy()
    ground_truth_all = ground_truth_all.cpu().numpy()
    total_test_acc = total_test_acc / num_of_batches

    return total_test_acc, predicted_all, ground_truth_all
