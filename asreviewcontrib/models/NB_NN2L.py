from asreviewcontrib.models.model_switcher import base_switcher

from asreview.models.classifiers.nn_2_layer import NN2LayerClassifier
from asreviewcontrib.models.Op_NN import OP_NN
from asreview.models.classifiers.nb import NaiveBayesClassifier


class NB_NN2L_Model(base_switcher):

    name = "NB_NN2L"

    def __init__(self):
        super().__init__()
        self._model_1 = NaiveBayesClassifier()
        self._model_2 = OP_NN(dense_width=64, verbose=1, epochs=100)
        #self._model_2 = NN2LayerClassifier(verbose=1)