from asreviewcontrib.models.model_switcher import base_switcher
from asreview.models.classifiers.svm import SVMClassifier
from asreview.models.classifiers.lstm_base import LSTMBaseClassifier


class SVM_LSTM_Model(base_switcher):

    name = "SVM_LSTM"

    def __init__(self):
        super().__init__()
        self._model_1 = SVMClassifier()
        self._model_2 = LSTMBaseClassifier()