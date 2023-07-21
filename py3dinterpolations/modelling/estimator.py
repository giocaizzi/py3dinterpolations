"""cross validation module"""

from sklearn.model_selection import GridSearchCV

from pykrige.rk import Krige # scikit-learn wrapper for pykrige


class Estimator:
    """class for estimation of model parameters.

    Note:
        At the moment only pykrige.Krige is supported.

    Args:

    """

    def __init__(self, griddata, params):

        self.estimator = GridSearchCV(
            Krige(),
            params,
            verbose=3,
        )
        self.estimator.fit(
            y=griddata.numpy_data[:, 3], # value
            X=griddata.numpy_data[:, 0:3] # x, y, z
        )

    @property
    def best_params(self):
        return self.estimator.best_params_
    
    @property
    def best_score(self):
        return self.estimator.best_score_
    
    @property
    def cv_results(self):
        return self.estimator.cv_results_