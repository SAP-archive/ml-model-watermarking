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

    args = TrainingWMArgs(
                nbr_classes=10,
                key_dawn=default_key(255),
                probability_dawn=1/100,
                trigger_technique='dawn',
                metric='accuracy')

    trainer = Trainer(
                model=model,
                trainset=trainset,
                args=args)

    ownership, wm_model = trainer.get_model()
    testloader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=32,
                        shuffle=True)
    triggerloader = torch.utils.data.DataLoader(
                        ownership['inputs'],
                        batch_size=32,
                        shuffle=True)
    results = []
    for _, data in enumerate(triggerloader):
        inputs = data
        results += list(torch.argmax(wm_model(inputs), 1).numpy())

    verification = verify(
                    ownership['labels'],
                    results,
                    number_labels=args.nbr_classes,
                    metric='accuracy',
                    dawn=True)

    print(verification)

if __name__ == '__main__':
    MNIST_dawn()
