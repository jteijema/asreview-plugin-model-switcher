from asreview.models.classifiers.base import BaseTrainClassifier

class base_switcher(BaseTrainClassifier):

    name = "base-switcher"


    def __init__(self,
                switchpoint=220,
                model_1=None,
                model_2=None,
                save_switch_point=False
                ):

        self._iteration = 0
        self._save_switch_point = save_switch_point

        self._switchpoint = switchpoint
        self._model_1 = model_1
        self._model_2 = model_2

    def fit(self, X, y):

        if(self._save_switch_point and self._iteration==self.switchpoint): 
            from numpy import savez
            savez('../output/data-' + str(self._iteration), X=X, y=y)
        
        self._iteration += 1

        if self._iteration == self._switchpoint : print("Switching Model")

        if (self._iteration < self._switchpoint): 
            self.log_message(type(self._model_1))
            return self._model_1.fit(X, y)

        else: 
            self.log_message(type(self._model_2))
            self._model_2.fit(X, y)

    def predict_proba(self, X):

        if (self._iteration < self._switchpoint): 
            return self._model_1.predict_proba(X)

        else: 
            return self._model_2.predict_proba(X)

    def log_message(self, 
                    modeltype = None
                    ):
        print("Iteration {}, fitting {}".format(self._iteration, modeltype))
