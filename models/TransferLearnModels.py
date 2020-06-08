import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Resnet152_wrapper(nn.Module):
    def __init__(self, model_params):
        super(Resnet152_wrapper, self).__init__()
        self.model_params = model_params
        self.num_of_classes = model_params['num_of_classes']
        self.resnet152 = models.resnet152(pretrained=True)
        self.params_to_update = []

        for param in self.resnet152.parameters():
            param.requires_grad = False

        for param in self.resnet152.layer4.parameters():
            self.params_to_update.append(param)
            param.requires_grad = True

        self.fc_malware1 = nn.Linear(1000, 500)
        self.fc_malware2 = nn.Linear(500, self.num_of_classes)

        for param in self.fc_malware1.parameters():
            self.params_to_update.append(param)
            param.requires_grad = True

        for param in self.fc_malware2.parameters():
            self.params_to_update.append(param)
            param.requires_grad = True

    def parameters(self):
        return self.params_to_update

    def forward(self, x):
        x = self.resnet152(x)
        x = self.fc_malware1(x)
        x = self.fc_malware2(x)
        return F.log_softmax(x, dim=1)


class VGG19_wrapper(nn.Module):
    def __init__(self, model_params):
        super(VGG19_wrapper, self).__init__()
        self.model_params = model_params
        self.num_of_classes = model_params['num_of_classes']
        self.vgg19 = models.vgg19(pretrained=True)
        self.params_to_update = []

        for param in self.vgg19.parameters():
            param.requires_grad = False

        list_of_features_layers = [34, 35, 36]
        for f in list_of_features_layers:
            for param in self.vgg19.features[f].parameters():
                self.params_to_update.append(param)
                param.requires_grad = True

        for param in self.vgg19.classifier.parameters():
            self.params_to_update.append(param)
            param.requires_grad = True

        self.fc_malware1 = nn.Linear(1000, 500)
        self.fc_malware2 = nn.Linear(500, self.num_of_classes)

        for param in self.fc_malware1.parameters():
            self.params_to_update.append(param)
            param.requires_grad = True

        for param in self.fc_malware2.parameters():
            self.params_to_update.append(param)
            param.requires_grad = True

    def parameters(self):
        return self.params_to_update

    def forward(self, x):
        x = self.vgg19(x)
        x = self.fc_malware1(x)
        x = self.fc_malware2(x)
        return F.log_softmax(x, dim=1)
