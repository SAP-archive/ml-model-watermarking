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


def clean_train(epochs):

    """ Training watermark-free model

    Parameters:
    epochs (int): number of epochs

    Returns:
    model (object) : trained model

    """
    model = LeNet()
    correct = 0
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    trainset, testset = load_MNIST()
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(trainloader, 0):
            X, y = data
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        model.eval()
        for x, y in testloader:
            predictions = torch.argmax(model(x), 1).numpy()
            correct += len(np.where(predictions == y.numpy())[0])
    accuracy_clean_regular = (100 * correct) / len(testset)
    return model, accuracy_clean_regular


def test_watermark(X, y, base_model, metric='accuracy', trigger_size=100):
    """ Test the watermark functions

    Parameters:
    X (array): Input data
    y (array): Label data (0 / 1)
    base_model (Object): Model to be tested
    metric (string): Type of metri for WM verification
    trigger_size (int): Number of trigger inputs

    """
    X_train, _, y_train, _ = train_test_split(X,
                                              y,
                                              test_size=0.1,
                                              random_state=42)

    # Train a watermarked model
    print('Training watermarked model')
    wm_model = Trainer(clone(base_model),
                       encryption=False,
                       metric=metric,
                       trigger_size=trigger_size)
    ownership = wm_model.fit(X_train, y_train)
    WM_X = ownership['inputs']
    number_labels = len(np.unique([floor(k) for k in y_train]))

    # Train a non-watermarked model
    print('Training non-watermarked model')
    clean_model = clone(base_model)
    clean_model.fit(X_train, y_train)

    # Verification for non-stolen
    print('Clean model not detected as stolen...', end=' ')
    if metric == 'accuracy':
        verification = verify(
                            ownership['labels'],
                            clean_model.predict(WM_X),
                            number_labels=number_labels,
                            metric=metric)

    else:
        verification = verify(
                            ownership['labels'],
                            clean_model.predict(WM_X),
                            number_labels=ownership['selected_q'],
                            bounds=(min(y), max(y)),
                            metric=metric)
    assert not verification['is_stolen']
    print('Done!')

    # Verification for stolen
    print('Stolen watermarked model detected as stolen...', end=' ')
    if metric == 'accuracy':
        verification = verify(
                            ownership['labels'],
                            wm_model.predict(WM_X),
                            number_labels=number_labels,
                            metric=metric)
    else:
        verification = verify(
                            ownership['labels'],
                            wm_model.predict(WM_X),
                            number_labels=ownership['selected_q'],
                            bounds=(min(y), max(y)),
                            metric=metric)
    assert verification['is_stolen']
    print('Done!')
