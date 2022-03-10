import random
from math import floor

import numpy as np
from cryptography.fernet import Fernet
from sklearn import svm
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mlmodelwatermarking.verification import verify


class Trainer:

    def __init__(self,
                 model,
                 args):
        """Main wrapper class to watermark sk models.

        Args:
            model (Object): Model
            args (dict): args for watermarking

        """
        self.model = model
        self.args = args
        self.encryption = args.encryption
        self.nb_blocks = args.nb_blocks
        self.metric = args.metric
        self.trigger_size = args.trigger_size
        self.watermarked = False
        # Supported models
        self.Classifiers = [RandomForestClassifier,
                            svm.SVC,
                            RidgeClassifier,
                            LogisticRegression]
        self.Regressors = [RandomForestRegressor]

    def get_trigger_block(self, block_id):
        """Return a precise trigger block."""
        return self.triggers['block_' + str(block_id)], self.triggers['shape']

    def generate_trigger(self, X_train, y_train, mode='CLASSIFICATION'):
        """Generation random trigger set.

        Args:
            X_train (array): Input data
            y_train (array): Label data
            mode (str, optional): Classification or
                regression mode

        Returns:
            ownership (dict): Watermark triggers information
            X_train (array): Modified input data
            y_train (array): Modified label data

        """
        # Check if inputs are arrays
        if type(X_train) != np.ndarray and type(y_train) != np.ndarray:
            raise TypeError("Error: X and Y must be np.ndarray!")

        WM_X, WM_y = np.array([]), np.array([])
        ownership = {}
        _, shape_x = X_train.shape

        for k in range(self.trigger_size):
            # Generate triggers
            trigger_x = np.array([[random.random() for i in range(shape_x)]])
            if mode == 'CLASSIFICATION':
                trigger_y = np.array([random.choice(np.unique(y_train))])
            else:
                a, b = min(y_train), max(y_train)
                trigger_y = np.array([random.randint(a, b)])
            # Add instance to trigger set
            if k == 0:
                WM_X, WM_y = trigger_x, trigger_y
            else:
                WM_X = np.vstack((WM_X, trigger_x))
                WM_y = np.vstack((WM_y, trigger_y))

        # Add trigger to training data
        X_train = np.vstack((X_train, WM_X))
        y_train = np.vstack((y_train.reshape(-1, 1), WM_y)).ravel()
        ownership['inputs'] = WM_X
        ownership['labels'] = WM_y.ravel()
        ownership['bounds'] = (min(y_train), max(y_train))

        return ownership, X_train, y_train

    def train_step(self, ownership, X_train, y_train):
        """Train the model depending on model type.

        Args:
            ownership (dict): Ownership information about
                            triggers
            X_train (array): Input data
            y_train (array): Label data (0 / 1)

        Returns:
            None for classification tasks
            Quantification parameter q for regression

        """
        number_labels = len(np.unique([floor(k) for k in y_train]))
        # CLASSIFICATION MODELS ###

        # Random Forest Classifier
        if isinstance(self.model, RandomForestClassifier):
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(ownership['inputs'])
            if verify(
                    ownership['labels'],
                    predictions,
                    number_labels=number_labels,
                    metric=self.metric)['is_stolen']:
                self.watermarked = True
            return None

        # Logistic Regression
        if isinstance(self.model, LogisticRegression):
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(ownership['inputs'])
            if verify(
                    ownership['labels'],
                    predictions,
                    number_labels=number_labels,
                    metric=self.metric)['is_stolen']:
                self.watermarked = True
            return None

        # SVC
        elif isinstance(self.model, svm.SVC):
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(ownership['inputs'])
            if verify(ownership['labels'],
                      predictions,
                      number_labels=number_labels,
                      metric=self.metric)['is_stolen']:
                self.watermarked = True

            return None

        # Ridge
        elif isinstance(self.model, RidgeClassifier):
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(ownership['inputs'])
            if verify(ownership['labels'],
                      predictions,
                      number_labels=number_labels,
                      metric=self.metric)['is_stolen']:
                self.watermarked = True

            return None

        # REGRESSION MODELS

        # Random Forest Regressor
        elif isinstance(self.model, RandomForestRegressor):
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(ownership['inputs'])
            for q in [1, 0.5, 0.3, 0.25, 0.2, 0.1, 0.05, 0.02]:
                selected_q = number_labels * q
                if verify(
                        ownership['labels'],
                        predictions,
                        number_labels=selected_q,
                        bounds=ownership['bounds'],
                        metric=self.metric)['is_stolen']:

                    break

            return selected_q

        else:
            raise NotImplementedError()

    def predict(self, X_test):
        """Predict method.

        Args:
            X_test (array): Test dataset

        Returns:
            predictions (array): Predictions

        """
        return self.model.predict(X_test)

    def fit(self, X_train, y_train):
        """ Train the model on watermarked data

        Args:
            X_train (array): Input data
            y_train (array): Label data (0 / 1)

        Returns:
            ownership (dict): Watermark triggers information

        """
        # Check type of the model
        if type(self.model) in self.Classifiers:
            # Generate triggers
            ownership, X_train, y_train = self.generate_trigger(
                                                    X_train,
                                                    y_train,
                                                    mode='CLASSIFICATION')
        elif type(self.model) in self.Regressors:
            # Generate triggers
            ownership, X_train, y_train = self.generate_trigger(
                                                    X_train,
                                                    y_train,
                                                    mode='REGRESSION')
        else:
            raise NotImplementedError()
        # Train the model

        selected_q = self.train_step(ownership, X_train, y_train)
        ownership['selected_q'] = selected_q
        # Store encrypted triggers
        if self.encryption:
            return self.encrypt(ownership)
        else:
            return ownership

    def encrypt(self, ownership):
        """ Store the watermark in encrypted fashion.

        Args:
            ownership (dict): Watermark triggers information

        Returns:
            encrypted_trigger (dict): Encrypted triggers information

        """
        # Initiating the keys and the triggers(crypted)
        keys = []
        triggers = {}

        WM_X = np.array(ownership['inputs'])
        nb_blocks = min(self.nb_blocks, len(WM_X))

        step = floor(len(WM_X) / nb_blocks)
        # For each block
        for block in range(0, nb_blocks):
            # Generate the encryption key
            key = Fernet.generate_key()
            # Add the key to the key set
            keys.append(key)
            # Initiating the encryption object based on the key
            f = Fernet(key)
            # Encrypt / store
            to_encrypt = WM_X[block * step:(block + 1) * step].tobytes()
            encrypted_trigger = f.encrypt(to_encrypt)
            triggers['block_' + str(block)] = encrypted_trigger
        shape_x, shape_y = WM_X.shape
        triggers['shape'] = (int(shape_x / nb_blocks), shape_y)
        triggers['dtype'] = WM_X.dtype

        encrypted_trigger = {'triggers': triggers, 'keys': keys}

        return encrypted_trigger

    def decrypt_trigger(self, triggers, block_id, key):
        """ Decrypt trigger block

        Args:
            triggers (dict): Encrypted trigger information
            block_id (int): Trigger block to decrypt
            key (str): Decryption key

        Returns:
            clear_trigger (array): Decrypted trigger

        """
        shape = triggers['shape']
        dtype = triggers['dtype']
        f = Fernet(key)
        clear_trigger = f.decrypt(triggers['block_' + str(block_id)])
        clear_trigger = np.frombuffer(clear_trigger, dtype=dtype)
        return clear_trigger.reshape(shape)
