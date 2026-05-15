"""Utility functions for modelling module."""

import pandas as pd

from ..core.types import NormalizationParams, StandardizationParams


def normalize(series: pd.Series) -> tuple[pd.Series, NormalizationParams]:
    """Normalize series to [0, 1] range.

    Args:
        series: Series to normalize.

    Returns:
        Tuple of (normalized series, normalization params).

    Note:
        NaN values are silently ignored when computing min/max (pandas
        ``skipna=True`` default). NaN inputs will remain NaN in the output.
    """
    series = series.copy()
    params = NormalizationParams(min=float(series.min()), max=float(series.max()))
    value_range = params.max - params.min
    if value_range == 0.0:
        series[:] = 0.0
        return series, params
    series = (series - params.min) / value_range
    return series, params


def standardize(series: pd.Series) -> tuple[pd.Series, StandardizationParams]:
    """Standardize series to mean=0, std=1.

    Args:
        series: Series to standardize.

    Returns:
        Tuple of (standardized series, standardization params).

    Note:
        NaN values are silently ignored when computing mean/std (pandas
        ``skipna=True`` default). NaN inputs will remain NaN in the output.
    """
    series = series.copy()
    params = StandardizationParams(mean=float(series.mean()), std=float(series.std()))
    if params.std == 0.0 or pd.isna(params.std):
        series[:] = 0.0
        return series, params
    series = (series - params.mean) / params.std
    return series, params
