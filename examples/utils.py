import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_MNIST
from models import MNIST


def clean_train(epochs):

    """ Training watermark-free model

    Parameters:
    epochs (int): number of epochs

    Returns:
    model (object) : trained model

    """
    model = MNIST()
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
