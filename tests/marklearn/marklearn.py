import numpy as np
from mlmodelwatermarking.marklearn.marklearn import MarkLearn
from mlmodelwatermarking.verification import verify
from sklearn import datasets
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def test_watermark(X, y, base_model, metric='accuracy'):
    """ Test the watermark functions

    Parameters:
    X (array): Input data
    y (array): Label data (0 / 1)
    base_model (Object): Model to be tested
    metric (string): Type of metri for WM verification

    """
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=42)

    # Train a watermarked model
    print('Training watermarked model')
    wm_model = MarkLearn(clone(base_model), encryption=False, metric=metric)
    ownership = wm_model.fit(X_train, y_train)
    WM_X = ownership['inputs']

    # Train a non-watermarked model
    print('Training non-watermarked model')
    clean_model = clone(base_model)
    clean_model.fit(X_train, y_train)

    # Verification for non-stolen
    print('Clean model not detected as stolen...', end=' ')
    is_stolen = verify(wm_model.predict(WM_X),
                       clean_model.predict(WM_X),
                       bounds=None,
                       number_labels=len(np.unique(y)))

    assert is_stolen is False
    print('Done!')

    # Verification for stolen
    print('Stolen watermarked detected as stolen...', end=' ')
    is_stolen = verify(wm_model.predict(WM_X),
                       wm_model.predict(WM_X),
                       bounds=None,
                       number_labels=len(np.unique(y)))

    assert is_stolen is True
    print('Done!')

    # Store encrypted triggers while training
    print('\nTraining watermarked with encrypted triggers')
    wm_model = MarkLearn(clone(base_model), encryption=True, metric=metric)
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
    is_stolen = verify(wm_model.predict(decrypted_trigger),
                       clean_model.predict(decrypted_trigger),
                       bounds=None,
                       number_labels=len(np.unique(y)))

    assert is_stolen is False
    print('Done')

    # Verification for stolen
    print('Stolen watermarked detected for a given block...', end=' ')
    is_stolen = verify(wm_model.predict(decrypted_trigger),
                       wm_model.predict(decrypted_trigger),
                       bounds=None,
                       number_labels=len(np.unique(y)))

    assert is_stolen is True
    print('Done')


if __name__ == '__main__':
    # Loading classification data
    wine = datasets.load_wine()
    X_wine, y_wine = wine.data, wine.target
    # Load regression data
    boston = datasets.load_boston()
    X_boston, y_boston = boston.data, boston.target

    # SGD Classifier
    print('\n\nSGD CLASSIFIER\n')
    base_model = SGDClassifier(loss='hinge',
                               penalty='l2',
                               max_iter=300,
                               random_state=42)
    test_watermark(X_wine, y_wine, base_model)

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
