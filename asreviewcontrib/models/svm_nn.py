from asreviewcontrib.models.model_switcher import base_switcher

from asreview.models.classifiers.nn_2_layer import NN2LayerClassifier
from asreview.models.classifiers.svm import SVMClassifier


class SVM_NN_Model(base_switcher):

    name = "svm_nn"

    def __init__(self):
        super().__init__()
        self._model_1 = SVMClassifier()
        self._model_2 = NN2LayerClassifier()
