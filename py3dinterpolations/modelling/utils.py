"""Utility functions for modelling module."""

import pandas as pd

from ..core.types import NormalizationParams, StandardizationParams


def normalize(series: pd.Series) -> tuple[pd.Series, NormalizationParams]:
    """Normalize series to [0, 1] range.

    Args:
        series: Series to normalize.

    Returns:
        Tuple of (normalized series, normalization params).
    """
    series = series.copy()
    params = NormalizationParams(min=float(series.min()), max=float(series.max()))
    series = (series - params.min) / (params.max - params.min)
    return series, params


def standardize(series: pd.Series) -> tuple[pd.Series, StandardizationParams]:
    """Standardize series to mean=0, std=1.

    Args:
        series: Series to standardize.

    Returns:
        Tuple of (standardized series, standardization params).
    """
    series = series.copy()
    params = StandardizationParams(mean=float(series.mean()), std=float(series.std()))
    series = (series - params.mean) / params.std
    return series, params
