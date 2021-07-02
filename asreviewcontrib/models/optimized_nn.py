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
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import callbacks
from tensorflow.keras import backend

from math import ceil

from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.utils import _set_class_weight


class POWER_CNN(BaseTrainClassifier):

    name = "power_cnn"

    def __init__(self, verbose = 1, patience=15, min_delta = 0.025):

        """Initialize the conv neural network model."""
        super(POWER_CNN, self).__init__()
        self.patience = patience
        self._model = None
        self.min_delta = min_delta
        self.verbose = verbose

        print("""

██████╗  ██████╗ ██╗    ██╗███████╗██████╗ 
██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔══██╗
██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝
██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗
██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║
╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
        ██████╗███╗   ██╗███╗   ██╗        
       ██╔════╝████╗  ██║████╗  ██║        
       ██║     ██╔██╗ ██║██╔██╗ ██║        
       ██║     ██║╚██╗██║██║╚██╗██║        
       ╚██████╗██║ ╚████║██║ ╚████║        
        ╚═════╝╚═╝  ╚═══╝╚═╝  ╚═══╝        
 
 """)
    

    def fit(self, X, y):

 
        self._model = KerasClassifier(_create_dense_nn_model(X.shape[1]))
        callback = callbacks.EarlyStopping(min_delta = self.min_delta, monitor='loss', patience=self.patience, restore_best_weights=True)
        
        print("\n Fitting New Iteration:\n")
        self._model.fit(
            _format(X),
            y,
            batch_size=ceil(X.shape[0]/20),
            epochs=100,
            shuffle=True,
            callbacks=[callback],
            verbose=self.verbose,)

    def predict_proba(self, X):
        return self._model.predict_proba(_format(X))

def _format(X):
        return X.reshape((X.shape[0],X.shape[1],1))

def _create_dense_nn_model(_size):

    def model_wrapper():

        backend.clear_session()

        model = Sequential()

        model.add(layers.SeparableConv1D(input_shape = (_size, 1), filters = 256, kernel_size = 5, padding = 'valid'))
        model.add(layers.Activation(activations.relu))
        model.add(layers.SeparableConv1D(filters = 256, kernel_size = 5, padding = 'valid'))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling1D(pool_size = 2, padding = 'valid'))

        model.add(layers.SeparableConv1D(filters = 256, kernel_size = 3, padding = 'valid'))
        model.add(layers.Activation(activations.relu))
        model.add(layers.SeparableConv1D(filters = 256, kernel_size = 3, padding = 'valid'))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling1D(pool_size = 2, padding = 'valid'))
        model.add(layers.Dropout(rate=0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(128))
        model.add(layers.Activation(activations.relu))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(16))

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(learning_rate=0.001),
            metrics=['acc'])

        return model

    return model_wrapper
