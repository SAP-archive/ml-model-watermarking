from mlmodelwatermarking.marktorch import Trainer
from mlmodelwatermarking import TrainingWMArgs
from mlmodelwatermarking.utils import load_trigger

from utils import LeNet, load_MNIST


def MNIST_selected():
    """Testing of watermarking for MNIST model."""

    # WATERMARKED
    model = LeNet()
    trainset, valset, testset = load_MNIST()
    specialset = load_trigger('tests/marktorch/trigger_set', (1, 28, 28))

    args = TrainingWMArgs(
                    optimizer='SGD',
                    lr=0.01,
                    gpu=True,
                    epochs=10,
                    nbr_classes=10,
                    batch_size=64,
                    trigger_technique='selected')

    trainer = Trainer(
                    model=model,
                    args=args,
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    specialset=specialset)

    ownership = trainer.train()
    accuracy_wm_regular = trainer.test()
    verification = trainer.verify(ownership)
    assert verification['is_stolen'] is True

    # CLEAN
    model = LeNet()
    trainer_clean = Trainer(
                    model=model,
                    args=args,
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    specialset=specialset,
                    watermark=False)

    trainer_clean.train()
    accuracy_clean_regular = trainer_clean.test()
    accuracy_loss = round(accuracy_clean_regular - accuracy_wm_regular, 4)
    print(f'Accuracy loss: {accuracy_loss}')
    clean_model = trainer_clean.model

    verification = trainer.verify(ownership, suspect=clean_model)
    assert verification['is_stolen'] is False


if __name__ == '__main__':
    MNIST_selected()
