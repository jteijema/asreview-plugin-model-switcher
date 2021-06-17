from sklearn.naive_bayes import MultinomialNB

from asreview.models.classifiers.base import BaseTrainClassifier

class SVM_LSTM_Model(BaseTrainClassifier):
    """Naive Bayes classifier

    The Naive Bayes classifier with the default SKLearn parameters.
    """

    name = "SVM_LSTM"

    def __init__(self):

        super().__init__()
        self._model = MultinomialNB()
