"""cross validation module"""

from ..core.griddata import GridData

from sklearn.model_selection import GridSearchCV

from pykrige.rk import Krige  # scikit-learn wrapper for pykrige


class Estimator:
    """class for estimation of model parameters.

    Runs a parameter estimation looking for the best `scoring` method.
    The `scoring` attribute can be any of the scoring methods supported by
    scikit-learn.
    See https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    Get the best parameters with the `best_params` attribute.
    Get the best score with the `best_score` attribute.
    Get the cross validation results with the `cv_results` attribute.

    Verbose level can be set from 0 to 3 to higher level of verbosity.

    Note:
        At the moment only pykrige.Krige is supported.

    Args:
        griddata (GridData): griddata to interpolate
        params (dict): parameters to search
        scoring (str, optional): scoring method. Defaults to "neg_mean_absolute_error".
        verbose (int, optional): verbosity level. Defaults to 3.

    Attributes:
        estimator (GridSearchCV): estimator object
        best_params (dict): best parameters
        best_score (float): best score
        cv_results (dict): cross validation results,
            ready to be converted to pandas DataFrame

    Examples:
        >>> # parameters of 3d kriging, both ordinary and universal
        >>> params = {
        >>>     "method": ["ordinary3d","universal3d"],
        >>>     "variogram_model": ["linear", "power", "gaussian"],
        >>>     "nlags": [2, 4, 6, 8, 10],
        >>>     "weight": [True, False],
        >>> }
    """

    def __init__(
        self,
        griddata: GridData,
        params: dict,
        scoring: str = "neg_mean_absolute_error",
        verbose: int = 3,
    ):
        self.estimator = GridSearchCV(
            Krige(),
            params,
            scoring=scoring,
            verbose=verbose,
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
