import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from math import floor
import numpy as np
from mlmodelwatermarking.marklearn import Trainer
from mlmodelwatermarking import TrainingWMArgs
from mlmodelwatermarking.verification import verify
from sklearn.base import clone

from sklearn.model_selection import train_test_split
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


class LeNet(nn.Module):
    """ MNIST model """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_MNIST():
    """ Load MNIST dataset
    Returns:
    trainloader (object): training dataloader
    testloader (object): test dataloader

    """
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset = torchvision.datasets.MNIST('/tmp/',
                                         train=True,
                                         download=True,
                                         transform=transformation)
    size_split = int(len(dataset) * 0.8)
    trainset, valset = torch.utils.data.random_split(
        dataset, [size_split, len(dataset) - size_split])

    testset = torchvision.datasets.MNIST('/tmp/',
                                         train=False,
                                         download=True,
                                         transform=transformation)

    return trainset, valset, testset
