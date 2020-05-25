import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from config import *
import numpy as np


class ANNMalware_Model1(nn.Module):
    def __init__(self, image_dim=32, num_of_classes=20):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes

        self.linear1_in_features = int(self.image_dim * self.image_dim)
        # reduce the neurons by 20% i.e. take 80% in_features
        self.linear1_out_features = int(self.linear1_in_features * 0.80)
        # reduce the neurons by 40%
        self.linear2_out_features = int(self.linear1_out_features * 0.60)

        self.classifier = nn.Sequential(
            nn.Linear(self.linear1_in_features, self.linear1_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear1_out_features, self.linear2_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear2_out_features, self.num_of_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class ANNMalware_Model2(nn.Module):
    def __init__(self, image_dim=32, num_of_classes=20):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes

        self.linear1_in_features = int(self.image_dim * self.image_dim)
        # reduce the neurons by 20% i.e. take 80% in_features
        self.linear1_out_features = int(self.linear1_in_features * 0.80)
        # reduce the neurons by 40%
        self.linear2_out_features = int(self.linear1_out_features * 0.60)
        # reduce the neurons by 20%
        self.linear3_out_features = int(self.linear1_out_features * 0.40)
        # reduce the neurons by 20%
        self.linear4_out_features = int(self.linear1_out_features * 0.20)

        self.classifier = nn.Sequential(
            nn.Linear(self.linear1_in_features, self.linear1_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear1_out_features, self.linear2_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear2_out_features, self.linear3_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear3_out_features, self.linear4_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear4_out_features, self.num_of_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
