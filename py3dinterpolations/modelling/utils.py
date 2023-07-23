"""utility functions for modelling module"""

import pandas as pd


def _normalize(series: pd.Series) -> tuple:
    """normalize series between 0 and 1

    Args:
        series (pd.Series): series to normalize

    Returns:
        tuple: normalized series and normalization parameters
    """
    series = series.copy()
    # save normalization parameters
    params = {
        "min": series.min(),
        "max": series.max(),
    }
    # normalize
    series = (series - params["min"]) / (params["max"] - params["min"])
    return series, params


def _standardize(series: pd.Series) -> tuple:
    """standardize series to have mean 0 and std 1

    Args:
        series (pd.Series): series to standardize

    Returns:
        tuple: standardized series and standardization parameters
    """
    series = series.copy()
    # save standardization parameters
    params = {
        "mean": series.mean(),
        "std": series.std(),
    }
    # standardize
    series = (series - params["mean"]) / (params["std"])
    return series, params
