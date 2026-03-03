"""Cross-validation parameter estimation for interpolation models."""

from pykrige.rk import Krige
from sklearn.model_selection import GridSearchCV

from ..core.griddata import GridData


class Estimator:
    """Parameter estimation via sklearn GridSearchCV.

    Currently supports pykrige's Krige wrapper for cross-validation.

    Args:
        griddata: Training data.
        params: Parameter grid for GridSearchCV.
        scoring: Scoring method. See sklearn docs.
        verbose: Verbosity level (0-3).
    """

    def __init__(
        self,
        griddata: GridData,
        params: dict[str, list[object]],
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
            y=griddata.numpy_data[:, 3],
            X=griddata.numpy_data[:, 0:3],
        )

    @property
    def best_params(self) -> dict[str, object]:
        return self.estimator.best_params_

    @property
    def best_score(self) -> float:
        return self.estimator.best_score_

    @property
    def cv_results(self) -> dict[str, object]:
        return self.estimator.cv_results_
