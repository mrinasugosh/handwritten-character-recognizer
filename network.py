# -*- coding: utf-8 -*-
# @Author: Mrinalini Sugosh
# @Date:   09-15-2021 13:26:46
# @Last Modified by:   Mrinalini Sugosh
# @Last Modified time: 09-15-2021 18:39:16
# Define the network used for training
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(28, 64, (5, 5), padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 2, padding=2)

        self.fc1 = nn.Linear(2048, 1024)

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)

        self.bn = nn.BatchNorm1d(1)

        self.fc3 = nn.Linear(512, 128)

        self.fc4 = nn.Linear(128, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv1_bn(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        x = x.view(-1, 1, 512)
        x = self.bn(x)

        x = x.view(-1, 512)
        x = self.fc3(x)
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)
        #return x