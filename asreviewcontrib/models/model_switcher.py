from asreview.models.classifiers.base import BaseTrainClassifier

class base_switcher(BaseTrainClassifier):

    name = "base-switcher"


    def __init__(self,
                switchpoint=220,
                model_1=None,
                model_2=None,
                ):

        self._iteration = 0
        
        self._switchpoint = switchpoint
        self._model_1 = model_1
        self._model_2 = model_2

    def fit(self, X, y):
        self._iteration += 1

        if (self._iteration < self._switchpoint) : 
            print("Iteration {}. Fitting {}".format(self._iteration, type(self._model_1)))
            return self._model_1.fit(X, y)

        else: 
            print("Iteration {}. Fitting {}".format(self._iteration, type(self._model_2)))
            self._model_2.fit(X, y)

    def predict_proba(self, X):

        if (self._iteration < self._switchpoint) : 
            return self._model_1.predict_proba(X)

        else: 
            print("Start NN prediction phase")
            return self._model_2.predict_proba(X)