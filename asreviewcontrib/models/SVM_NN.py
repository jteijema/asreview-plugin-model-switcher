from asreviewcontrib.models.model_switcher import base_switcher

from asreview.models.classifiers.nn_2_layer import NN2LayerClassifier
from asreviewcontrib.models.optimized_nn import OP_NN
from asreview.models.classifiers.svm import SVMClassifier


class SVM_NN_Model(base_switcher):

    name = "SVM_NN"

    def __init__(self):
        super().__init__()
        self._model_1 = SVMClassifier()
        #self._model_2 = OP_NN(verbose=0)
        self._model_2 = NN2LayerClassifier()