# Copyright 2020 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging


try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
except ImportError:
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True
    try:
        tf.logging.set_verbosity(tf.logging.ERROR)
    except AttributeError:
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

import scipy

from tensorflow.keras import callbacks

from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.models.classifiers.lstm_base import _get_optimizer
from asreview.utils import _set_class_weight


def _check_tensorflow():
    if not TF_AVAILABLE:
        raise ImportError(
            "Install tensorflow package (`pip install tensorflow`) to use"
            " 'EmbeddingIdf'.")


class OP_NN(BaseTrainClassifier):

    name = "OP_NN"

    def __init__(self,
                 dense_width=64,
                 optimizer='rmsprop',
                 learn_rate=1.0,
                 verbose=0,
                 epochs=50,
                 shuffle=False,
                 class_weight=30.0):
        """Initialize the 2-layer neural network model."""
        super(OP_NN, self).__init__()
        self.dense_width = int(dense_width)
        self.optimizer = optimizer
        self.learn_rate = learn_rate
        self.verbose = verbose
        self.epochs = int(epochs)
        self.shuffle = shuffle
        self.class_weight = class_weight

        self._model = None
        self.input_dim = None

    def fit(self, X, y):

        # check is tensorflow is available
        _check_tensorflow()

        if scipy.sparse.issparse(X):
            X = X.toarray()
        if self._model is None or X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
            keras_model = _create_dense_nn_model(
                self.input_dim, self.dense_width, self.optimizer,
                self.learn_rate, self.verbose)
            self._model = KerasClassifier(keras_model, verbose=self.verbose)

        callback = callbacks.EarlyStopping(monitor='acc', patience=10, restore_best_weights=True)
        

        self._model.fit(
            X,
            y,
            batch_size=24,
            epochs=self.epochs,
            shuffle=self.shuffle,
            callbacks=[callback],
            verbose=self.verbose,
            #class_weight=_set_class_weight(self.class_weight))
        )

    def predict_proba(self, X):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return super(OP_NN, self).predict_proba(X)


def _create_dense_nn_model(vector_size=40,
                           dense_width=64,
                           optimizer='rmsprop',
                           learn_rate_mult=1.0,
                           verbose=1):
    """Return callable lstm model.

    Returns
    -------
    callable:
        A function that return the Keras Sklearn model when
        called.

    """

    # check is tensorflow is available
    _check_tensorflow()

    def model_wrapper():
        model = Sequential()

        model.add(
            Dense(
                dense_width*2,
                input_dim=vector_size,
                activation='relu',
            ))

        # add Dense layer with relu activation
        model.add(
            Dense(
                dense_width,
                activation='relu',
            ))

        model.add(
            Dense(
                dense_width,
                activation='relu',
            ))

        # add Dense layer
        model.add(Dense(1, activation='sigmoid'))

        optimizer_fn = _get_optimizer(optimizer, learn_rate_mult)

        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer_fn,
            metrics=['acc'])

        if verbose >= 1:
            model.summary()

        return model

    return model_wrapper
