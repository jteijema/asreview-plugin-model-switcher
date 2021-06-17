from sklearn.naive_bayes import MultinomialNB

from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.models.classifiers.nn_2_layer import NN2LayerClassifier

class NB_NN2L_Model(BaseTrainClassifier):
    """Naive Bayes classifier

    The Naive Bayes classifier with the default SKLearn parameters.
    """

    name = "NB_NN2L"

    def __init__(self):
        self.iteration = 0

        super().__init__()
        self.nb_model = MultinomialNB()
        self.nn_model = NN2LayerClassifier()

    def fit(self, X, y):
        self.iteration += 1

        if (self.iteration < 220) : 
            return self.nb_model.fit(X, y)

        else: 
            self.nn_model.fit()

    def predict_proba(self, X):
        
        if (self.iteration < 220) : 
            return self.nb_model.predict_proba(X)

        else: 
            return self.nn_model.predict_proba(X)