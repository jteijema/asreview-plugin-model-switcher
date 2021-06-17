from sklearn.naive_bayes import MultinomialNB

from asreview.models.classifiers.base import BaseTrainClassifier

class NB_NN2L_Model(BaseTrainClassifier):
    """Naive Bayes classifier

    The Naive Bayes classifier with the default SKLearn parameters.
    """

    name = "NB_NN2L"

    def __init__(self):

        super().__init__()
        self._model = MultinomialNB()
