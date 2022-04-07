from math import floor
import numpy as np
from tqdm import tqdm

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from mlmodelwatermarking.marklearn import Trainer
from mlmodelwatermarking import TrainingWMArgs
from mlmodelwatermarking.verification import verify

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def test_watermark_sklearn(X, y, base_model,
                           metric='accuracy', trigger_size=100):
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
    number_labels = len(np.unique([floor(k) for k in y_train]))
    args = TrainingWMArgs(
                    nbr_classes=number_labels,
                    trigger_size=trigger_size,
                    metric=metric)

    wm_model = Trainer(clone(base_model), args=args)
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
