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

import scipy

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import callbacks
from tensorflow.keras import backend

from math import ceil

from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.utils import _set_class_weight


class POWER_LSTM(BaseTrainClassifier):

    name = "power_lstm"

    def __init__(self, verbose = 1):

        """Initialize the 2-layer neural network model."""
        super(POWER_LSTM, self).__init__()
        self._model = None
        self.verbose = verbose
    

    def fit(self, X, y):

 
        self._model = KerasClassifier(_create_dense_nn_model(X.shape[1]))
        
        print("\n Fitting New Iteration:\n")
        self._model.fit(
            _format(X),
            y,
            batch_size=ceil(X.shape[0]/20),
            epochs=50,
            shuffle=True,
            verbose=self.verbose,)

    def predict_proba(self, X):
        return self._model.predict_proba(_format(X))

def _format(X):
        return X.reshape((X.shape[0],X.shape[1],1))

def _create_dense_nn_model(_size):

    def model_wrapper():

        backend.clear_session()

        model = Sequential()
        
        model.add(layers.Conv1D(input_shape = (_size, 1), filters = 128, kernel_size = 5, padding = 'valid'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.LSTM(_size,return_sequences = True, implementation=2))
        model.add(layers.Dropout(0.7))
        model.add(layers.Activation('relu'))
        model.add(layers.LSTM(256,return_sequences = True))
        model.add(layers.Activation('relu'))
        model.add(layers.LSTM(128,return_sequences = True))
        model.add(layers.Activation('relu'))
        model.add(layers.LSTM(64,return_sequences = True))
        model.add(layers.Activation('relu'))
        model.add(layers.LSTM(32,return_sequences = True))
        model.add(layers.Activation('relu'))
        model.add(layers.LSTM(16))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(learning_rate=0.001),
            metrics=['acc'])

        return model

    return model_wrapper
