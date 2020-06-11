import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from config import *
import numpy as np


class CNNMalware_Model1(nn.Module):
    def __init__(self, image_dim=32, num_of_classes=20):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes

        self.conv1_out_channel = 12
        self.conv1_kernel_size = 3

        self.conv2_out_channel = 16
        self.conv2_kernel_size = 3

        self.linear1_out_features = 120
        self.linear2_out_features = 90

        self.conv1 = nn.Conv2d(1, self.conv1_out_channel, self.conv1_kernel_size, stride=1,
                               padding=(2, 2))

        self.conv2 = nn.Conv2d(self.conv1_out_channel, self.conv2_out_channel, self.conv2_kernel_size, stride=1,
                               padding=(2, 2))

        self.temp = int((((self.image_dim + 2) / 2) + 2) / 2)

        self.fc1 = nn.Linear(self.temp * self.temp * self.conv2_out_channel, self.linear1_out_features)
        self.fc2 = nn.Linear(self.linear1_out_features, self.linear2_out_features)
        self.fc3 = nn.Linear(self.linear2_out_features, self.num_of_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.temp * self.temp * self.conv2_out_channel)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


class CNNMalware_Model2(nn.Module):
    def __init__(self, image_dim=32, num_of_classes=20):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes
        self.padding = 2
        self.conv1_out_channel = 15
        self.conv1_kernel_size = 15
        self.stride = 1
        self.conv2_out_channel = 16
        self.conv2_kernel_size = 3

        conv1_nurons = int((self.image_dim - self.conv1_kernel_size + 2 * self.padding) / self.stride + 1)
        maxpool2d_1_nurons = int(conv1_nurons / 2)
        conv2_nurons = ((maxpool2d_1_nurons - self.conv2_kernel_size + 2 * self.padding) / self.stride + 1)
        maxpool2d_2_nurons = int(conv2_nurons / 2)

        self.linear1_in_features = int(maxpool2d_2_nurons * maxpool2d_2_nurons * self.conv2_out_channel)

        # reduce the neurons by 20% i.e. take 80% in_features
        self.linear1_out_features = int(self.linear1_in_features * 0.80)
        # reduce the neurons by 40%
        self.linear2_out_features = int(self.linear1_out_features * 0.60)

        self.features = nn.Sequential(
            nn.Conv2d(1, self.conv1_out_channel, self.conv1_kernel_size,
                      stride=self.stride, padding=(self.padding, self.padding)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.conv1_out_channel, self.conv2_out_channel, self.conv2_kernel_size,
                      stride=self.stride, padding=(self.padding, self.padding)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.linear1_in_features, self.linear1_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear1_out_features, self.linear2_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear2_out_features, self.num_of_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class CNNMalware_Model3(nn.Module):
    def __init__(self, image_dim_h=1, image_dim_w=1024, num_of_classes=20):
        super().__init__()

        self.image_dim_h = image_dim_h
        self.image_dim_w = image_dim_w
        self.num_of_classes = num_of_classes

        self.conv1_in_channel = 1
        self.conv1_out_channel = 28
        self.conv1_kernel_size = 3

        self.conv2_out_channel = 16
        self.conv2_kernel_size = 3

        self.linear1_out_features = 120
        self.linear2_out_features = 90

        self.conv1 = nn.Conv1d(self.conv1_in_channel, self.conv1_out_channel, self.conv1_kernel_size, stride=1,
                               padding=(0))

        self.conv2 = nn.Conv1d(self.conv1_out_channel, self.conv2_out_channel, self.conv2_kernel_size, stride=1,
                               padding=(2))

        self.fc1_size = self.conv2_out_channel * self.image_dim_w

        self.fc1 = nn.Linear(self.fc1_size, self.linear1_out_features)
        self.fc2 = nn.Linear(self.linear1_out_features, self.linear2_out_features)
        self.fc3 = nn.Linear(self.linear2_out_features, self.num_of_classes)

    def forward(self, X):
        X = X.squeeze(dim=1)
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = X.view(-1, self.fc1_size)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


class CNNMalware_Model4(nn.Module):
    def __init__(self, image_dim_h=1, image_dim_w=1024, num_of_classes=20,
                 c1_out=32, c1_kernel=16, c1_padding=2, c1_stride=2,
                 c2_out=32, c2_kernel=8, c2_padding=2, c2_stride=2,
                 ):
        super().__init__()

        self.image_dim_h = image_dim_h
        self.image_dim_w = image_dim_w
        self.num_of_classes = num_of_classes
        self.dilation = 1

        self.conv1_in_channel = 1
        self.conv1_out_channel = c1_out
        self.conv1_kernel_size = c1_kernel
        self.conv1_padding = c1_padding
        self.conv1_stride = c1_stride

        self.conv2_out_channel = c2_out
        self.conv2_kernel_size = c2_kernel
        self.conv2_padding = c2_padding
        self.conv2_stride = c2_stride

        self.linear1_out_features = 128 * 4
        self.linear2_out_features = 128

        self.conv1 = nn.Conv1d(self.conv1_in_channel, self.conv1_out_channel, self.conv1_kernel_size,
                               stride=self.conv1_stride,
                               padding=(self.conv1_padding), dilation=self.dilation)

        self.conv1_out_dim = self.calc_conv1d_out(self.image_dim_w, self.conv1_padding,
                                                  self.dilation, self.conv1_kernel_size, self.conv1_stride)

        self.conv2 = nn.Conv1d(self.conv1_out_channel, self.conv2_out_channel, self.conv2_kernel_size,
                               stride=self.conv2_stride,
                               padding=(self.conv2_padding))

        self.conv2_out_dim = self.calc_conv1d_out(self.conv1_out_dim, self.conv2_padding,
                                                  self.dilation, self.conv2_kernel_size, self.conv2_stride)

        self.fc1 = nn.Linear(self.conv2_out_dim * self.conv2_out_channel, self.linear1_out_features)
        self.fc2 = nn.Linear(self.linear1_out_features, self.linear2_out_features)
        self.fc3 = nn.Linear(self.linear2_out_features, self.num_of_classes)

    def calc_conv1d_out(self, l_in, padding, dilation, kernel_size, stride):
        return int(((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    def forward(self, X):
        X = X.squeeze(dim=1)
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


class CNNMalware_Model5(nn.Module):
    def __init__(self, input_dim=1024, embedding_dim=128, n_filters=3, filter_sizes=[3,6], output_dim=128, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        cat2 = self.fc(cat)
        return F.log_softmax(cat2, dim=1)
