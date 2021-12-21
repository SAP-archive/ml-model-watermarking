import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from mlmodelwatermarking.marktorch.marktorch import MarkTorch
from mlmodelwatermarking.marktorch.utils import load_trigger


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


def MNIST_noise():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = LeNet()
    trainset, valset, testset = load_MNIST()

    trainer = MarkTorch(
                    model=model,
                    optimizer=optim.SGD(model.parameters(), lr=0.01),
                    criterion=nn.NLLLoss(),
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    nbr_classes=10)

    ownership = trainer.train(epochs=20)
    accuracy_wm_regular = trainer.test()
    is_stolen, _, _ = trainer.verify(ownership)
    assert is_stolen is True

    # CLEAN
    model = LeNet()
    trainer_clean = MarkTorch(
                    model=model,
                    optimizer=optim.SGD(model.parameters(), lr=0.01),
                    criterion=nn.NLLLoss(),
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    nbr_classes=10,
                    watermark=False)

    trainer_clean.train(epochs=20)
    accuracy_clean_regular = trainer_clean.test()
    accuracy_loss = round(accuracy_clean_regular - accuracy_wm_regular, 4)
    print(f'Accuracy loss: {accuracy_loss}')
    clean_model = trainer_clean.get_model()

    is_stolen, _, _ = trainer.verify(ownership, suspect=clean_model)
    assert is_stolen is False


def MNIST_selected():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = LeNet()
    trainset, valset, testset = load_MNIST()
    specialset = load_trigger('tests/marktorch/trigger_set', (1, 28, 28))

    trainer = MarkTorch(
                    model=model,
                    optimizer=optim.SGD(model.parameters(), lr=0.01),
                    criterion=nn.NLLLoss(),
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    specialset=specialset,
                    trigger_technique='selected',
                    nbr_classes=10)

    ownership = trainer.train(epochs=25)
    accuracy_wm_regular = trainer.test()
    is_stolen, _, _ = trainer.verify(ownership)
    assert is_stolen is True

    # CLEAN
    model = LeNet()
    trainer_clean = MarkTorch(
                    model=model,
                    optimizer=optim.SGD(model.parameters(), lr=0.01),
                    criterion=nn.NLLLoss(),
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    nbr_classes=10,
                    watermark=False)

    trainer_clean.train(epochs=25)
    accuracy_clean_regular = trainer_clean.test()
    accuracy_loss = round(accuracy_clean_regular - accuracy_wm_regular, 4)
    print(f'Accuracy loss: {accuracy_loss}')
    clean_model = trainer_clean.get_model()
    
    is_stolen, _, _ = trainer.verify(ownership, suspect=clean_model)
    assert is_stolen is False


def MNIST_patch():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = LeNet()
    trainset, valset, testset = load_MNIST()

    trainer = MarkTorch(
                    model=model,
                    optimizer=optim.SGD(model.parameters(), lr=0.01),
                    criterion=nn.NLLLoss(),
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    patch_args={'msg': 'ID42', 'target': 5},
                    trigger_technique='patch',
                    nbr_classes=10)

    ownership = trainer.train(epochs=20)
    accuracy_wm_regular = trainer.test()
    is_stolen, _, _ = trainer.verify(ownership)
    assert is_stolen is True

    # CLEAN
    model = LeNet()
    trainer_clean = MarkTorch(
                    model=model,
                    optimizer=optim.SGD(model.parameters(), lr=0.01),
                    criterion=nn.NLLLoss(),
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    nbr_classes=10,
                    watermark=False)

    trainer_clean.train(epochs=20)
    accuracy_clean_regular = trainer_clean.test()
    accuracy_loss = round(accuracy_clean_regular - accuracy_wm_regular, 4)
    print(f'Accuracy loss: {accuracy_loss}')
    clean_model = trainer_clean.get_model()
    
    is_stolen, _, _ = trainer.verify(ownership, suspect=clean_model)
    assert is_stolen is False


if __name__ == '__main__':
    #MNIST_noise()
    MNIST_selected()
    #MNIST_patch()
