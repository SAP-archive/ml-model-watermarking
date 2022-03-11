import string
import random
import torch
from warnings import simplefilter

from mlmodelwatermarking.marktorch import Trainer
from mlmodelwatermarking.verification import verify
from mlmodelwatermarking import TrainingWMArgs

from utils import LeNet, load_MNIST

simplefilter(action='ignore', category=UserWarning)


def default_key(length: int):
    elements = string.ascii_uppercase + string.digits
    return ''.join(random.choices(elements, k=length))


def MNIST_dawn():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = LeNet()

    trainset, valset, testset = load_MNIST()
    model = LeNet()
    args = TrainingWMArgs(
            trigger_technique='merrer',
            optimizer='SGD',
            lr=0.01,
            gpu=True,
            epochs=10,
            nbr_classes=10,
            batch_size=64,
            watermark=False)

    trainer_clean = Trainer(
                    model=model,
                    args=args,
                    trainset=trainset,
                    valset=valset,
                    testset=testset)
    trainer_clean.train()
    original_model = trainer_clean.get_model()

    args = TrainingWMArgs(
                nbr_classes=10,
                key_dawn=default_key(255),
                probability_dawn=1/100,
                trigger_technique='dawn',
                metric='accuracy')

    trainer = Trainer(
                model=original_model,
                trainset=trainset,
                args=args)

    ownership, wm_model = trainer.get_model()
    triggerloader = torch.utils.data.DataLoader(
                        ownership['inputs'],
                        batch_size=32,
                        shuffle=True)
    results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for _, data in enumerate(triggerloader):
        inputs = data
        pred = wm_model(inputs.to(device))
        results += list(torch.argmax(pred, 1).cpu().numpy())

    verification = verify(
                    ownership['labels'],
                    results,
                    number_labels=args.nbr_classes,
                    metric='accuracy',
                    dawn=True)

    assert verification['is_stolen'] is True

    results = []
    for _, data in enumerate(triggerloader):
        inputs = data
        pred = original_model(inputs.to(device))
        results += list(torch.argmax(pred, 1).cpu().numpy())

    verification = verify(
                    ownership['labels'],
                    results,
                    number_labels=args.nbr_classes,
                    metric='accuracy')

    assert verification['is_stolen'] is False


if __name__ == '__main__':
    MNIST_dawn()
