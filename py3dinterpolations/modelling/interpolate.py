"""Top-level interpolation function."""

import logging

from ..core.grid3d import create_grid
from ..core.griddata import GridData
from ..core.types import ModelType
from .estimator import Estimator
from .modeler import Modeler
from .models import get_model
from .preprocessor import PreprocessingKwargs, Preprocessor

logger = logging.getLogger(__name__)


def interpolate(
    griddata: GridData,
    model_type: ModelType | str,
    grid_resolution: float | dict[str, float],
    model_params: dict[str, object] | None = None,
    model_params_grid: dict[str, list[object]] | None = None,
    preprocessing: PreprocessingKwargs | None = None,
    **predict_kwargs: object,
) -> Modeler:
    """Interpolate GridData and return the Modeler with results.

    Args:
        griddata: Source data to interpolate.
        model_type: Which model to use (e.g. "ordinary_kriging", "idw").
        grid_resolution: Grid resolution. Float for regular, dict for irregular.
        model_params: Model constructor parameters.
        model_params_grid: Parameter grid for cross-validation search.
        preprocessing: Keyword args for Preprocessor
            (e.g. downsampling_res, normalize_xyz).
        **predict_kwargs: Extra kwargs passed to model.predict().

    Returns:
        Modeler instance with .result populated.

    Raises:
        ValueError: If neither or both model_params/model_params_grid are given.
        NotImplementedError: If parameter search is used for non-kriging models.
    """
    logger.info("Starting interpolation with model=%s", model_type)

    if model_params is None and model_params_grid is None:
        msg = "Either model_params or model_params_grid must be provided"
        raise ValueError(msg)
    if model_params is not None and model_params_grid is not None:
        msg = "Cannot provide both model_params and model_params_grid"
        raise ValueError(msg)

    # Build grid
    grid = create_grid(griddata, grid_resolution)

    # Preprocess if needed
    if preprocessing is not None:
        preprocessor = Preprocessor(griddata, **preprocessing)
        griddata = preprocessor.preprocess()

    # Parameter search via estimator
    if model_params is None:
        model_type_enum = ModelType(model_type)
        if model_type_enum != ModelType.ORDINARY_KRIGING:
            msg = "Parameter search is only supported for ordinary_kriging"
            raise NotImplementedError(msg)

        assert model_params_grid is not None
        est = Estimator(griddata, model_params_grid)
        model_params = dict(est.best_params)
        model_params.pop("method", None)

    # Build and fit model
    model = get_model(model_type, **model_params)
    modeler = Modeler(griddata=griddata, grid=grid, model=model)

    # Predict
    modeler.predict(**predict_kwargs)

    logger.info("Interpolation complete")
    return modeler
