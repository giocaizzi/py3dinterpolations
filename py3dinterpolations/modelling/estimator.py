"""cross validation module"""

from sklearn.model_selection import GridSearchCV

from pykrige.rk import Krige  # scikit-learn wrapper for pykrige


class Estimator:
    """class for estimation of model parameters.

    Note:
        At the moment only pykrige.Krige is supported.

    Args:

    """

    def __init__(self, griddata, params, scoring="neg_mean_absolute_error", verbose=3):
        self.estimator = GridSearchCV(
            Krige(),
            params,
            scoring=scoring,  # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            verbose=verbose,  # from 1 to 3 to higher level of verbosity
        )
        self.estimator.fit(
            y=griddata.numpy_data[:, 3],  # value
            X=griddata.numpy_data[:, 0:3],  # x, y, z
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
