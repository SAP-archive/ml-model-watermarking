from math import floor
import numpy as np
from mlmodelwatermarking.marklearn import Trainer
from mlmodelwatermarking.verification import verify
from sklearn import datasets
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def test_watermark(X, y, base_model, metric='accuracy'):
    """ Test the watermark functions

    Parameters:
    X (array): Input data
    y (array): Label data (0 / 1)
    base_model (Object): Model to be tested
    metric (string): Type of metri for WM verification

    """
    X_train, _, y_train, _ = train_test_split(X,
                                              y,
                                              test_size=0.1,
                                              random_state=42)

    # Train a watermarked model
    print('Training watermarked model')
    wm_model = Trainer(clone(base_model), encryption=False, metric=metric)
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
                            bounds=None,
                            number_labels=number_labels,
                            metric=metric)

    else:
        verification = verify(
                            ownership['labels'],
                            clean_model.predict(WM_X),
                            bounds=(min(y), max(y)),
                            number_labels=ownership['selected_q'],
                            metric=metric)
    assert not verification['is_stolen']
    print('Done!')

    # Verification for stolen
    print('Stolen watermarked model detected as stolen...', end=' ')
    if metric == 'accuracy':
        verification = verify(
                            ownership['labels'],
                            wm_model.predict(WM_X),
                            bounds=None,
                            number_labels=number_labels,
                            metric=metric)
    else:
        verification = verify(
                            ownership['labels'],
                            wm_model.predict(WM_X),
                            bounds=(min(y), max(y)),
                            number_labels=ownership['selected_q'],
                            metric=metric)
    assert verification['is_stolen']
    print('Done!')

    # Store encrypted triggers while training
    print('\nTraining watermarked with encrypted triggers')
    wm_model = Trainer(clone(base_model), encryption=True, metric=metric)
    encrypted_ownership = wm_model.fit(X_train, y_train)

    # Retrieve shape of the trigger inputs
    triggers = encrypted_ownership['triggers']
    keys = encrypted_ownership['keys']
    # Select the trigger block to use
    block_id = 0
    key = keys[block_id]
    # Decrypt the trigger
    print('Decrypt block')
    decrypted_trigger = wm_model.decrypt_trigger(
                                            triggers,
                                            block_id=block_id,
                                            key=key)

    # Verification for non-stolen
    print('Clean model detected as stolen for a given block...', end=' ')
    if metric == 'accuracy':
        verification = verify(
                            ownership['labels'],
                            clean_model.predict(decrypted_trigger),
                            bounds=None,
                            number_labels=number_labels,
                            metric=metric)
    else:
        verification = verify(
                            ownership['labels'],
                            clean_model.predict(decrypted_trigger),
                            bounds=(min(y), max(y)),
                            number_labels=ownership['selected_q'],
                            metric=metric)
    assert not verification['is_stolen']
    print('Done')

    # Verification for stolen
    print('Stolen watermarked detected for a given block...', end=' ')
    if metric == 'accuracy':
        verification = verify(
                            wm_model.predict(decrypted_trigger),
                            wm_model.predict(decrypted_trigger),
                            bounds=None,
                            number_labels=number_labels,
                            metric=metric)
    else:
        verification = verify(
                            wm_model.predict(decrypted_trigger),
                            wm_model.predict(decrypted_trigger),
                            bounds=(min(y), max(y)),
                            number_labels=ownership['selected_q'],
                            metric=metric)
    assert verification['is_stolen']
    print('Done')


if __name__ == '__main__':
    # Loading classification data
    wine = datasets.load_wine()
    X_wine, y_wine = wine.data, wine.target
    # Load regression data
    boston = datasets.load_boston()
    X_boston, y_boston = boston.data, boston.target

    # Random Forest Classifier
    print('RANDOM FOREST CLASSIFIER\n')
    base_model = RandomForestClassifier(max_depth=1000, random_state=42)
    test_watermark(X_wine, y_wine, base_model)

    # SVC
    print('\n\nSVC\n')
    base_model = SVC(gamma=10, C=1)
    test_watermark(X_wine, y_wine, base_model)

    # Random Forest Regressor
    print('\n\nRANDOM FOREST REGRESSOR\n')
    base_model = RandomForestRegressor(max_depth=1000, random_state=42)
    test_watermark(X_boston, y_boston, base_model, metric='RMSE')
    test_watermark(X_boston, y_boston, base_model, metric='MAPE')
