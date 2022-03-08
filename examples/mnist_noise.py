from mlmodelwatermarking.marktorch import Trainer
from mlmodelwatermarking import TrainingWMArgs

from utils import LeNet, load_MNIST


def MNIST_noise():
    """Testing of watermarking for MNIST model."""

    trainset, valset, testset = load_MNIST()

    args = TrainingWMArgs(
                    optimizer='SGD',
                    lr=0.01,
                    gpu=True,
                    batch_size=64,
                    epochs=10,
                    nbr_classes=10)

    # CLEAN
    model = LeNet()
    trainer_clean = Trainer(
                    model=model,
                    args=args,
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    watermark=False)
    # WATERMARKED
    model = LeNet()
    trainer = Trainer(
                model=model,
                args=args,
                trainset=trainset,
                valset=valset,
                testset=testset)

    ownership = trainer.train()
    accuracy_wm_regular = trainer.test()
    verification = trainer.verify(ownership)
    assert verification['is_stolen'] is True

    trainer_clean.train()
    accuracy_clean_regular = trainer_clean.test()
    accuracy_loss = round(accuracy_clean_regular - accuracy_wm_regular, 4)
    print(f'Accuracy loss: {accuracy_loss}')

    verification = trainer.verify(ownership, suspect=trainer_clean.model)
    assert verification['is_stolen'] is False


if __name__ == '__main__':
    MNIST_noise()
