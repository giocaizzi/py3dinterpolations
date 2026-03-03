"""Preprocessing pipeline for GridData."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypedDict, cast

import pandas as pd

from ..core.griddata import GridData
from ..core.types import (
    Axis,
    DownsamplingParams,
    DownsamplingStatistic,
    NormalizationParams,
    PreprocessingParams,
    StandardizationParams,
)
from .utils import normalize, standardize

logger = logging.getLogger(__name__)


class PreprocessingKwargs(TypedDict, total=False):
    """Type-safe kwargs for Preprocessor construction."""

    downsampling_res: float | None
    downsampling_method: DownsamplingStatistic | str | Callable[..., pd.DataFrame]
    normalize_xyz: bool
    standardize_v: bool


class Preprocessor:
    """Preprocess GridData before interpolation.

    Supports downsampling, normalization of XYZ, and standardization of V.
    Returns a new GridData with preprocessing params attached.

    Args:
        griddata: Source data to preprocess.
        downsampling_res: Block resolution for downsampling. None to skip.
        downsampling_method: Statistic for downsampling, or a custom callable.
        normalize_xyz: Whether to normalize XYZ to [0, 1].
        standardize_v: Whether to standardize V to mean=0, std=1.
    """

    def __init__(
        self,
        griddata: GridData,
        downsampling_res: float | None = None,
        downsampling_method: (
            DownsamplingStatistic | str | Callable[..., pd.DataFrame]
        ) = DownsamplingStatistic.MEAN,
        normalize_xyz: bool = True,
        standardize_v: bool = True,
    ):
        self.griddata = griddata
        self.downsampling_res = downsampling_res
        self.downsampling_method = downsampling_method
        self.normalize_xyz = normalize_xyz
        self.standardize_v = standardize_v

    def preprocess(self) -> GridData:
        """Execute the preprocessing pipeline.

        Returns:
            New GridData with preprocessed data and params attached.
        """
        logger.info("Starting preprocessing")
        data = self.griddata.data.copy().reset_index()[["ID", "X", "Y", "Z", "V"]]

        downsampling_params: DownsamplingParams | None = None
        normalization_params: dict[Axis, NormalizationParams] | None = None
        standardization_params: StandardizationParams | None = None

        if self.downsampling_res is not None:
            data = self._downsample_data(data, statistic=self.downsampling_method)
            downsampling_params = DownsamplingParams(resolution=self.downsampling_res)

        if self.normalize_xyz:
            data, normalization_params = self._normalize_xyz(data)

        if self.standardize_v:
            data, standardization_params = self._standardize_v(data)

        params = PreprocessingParams(
            downsampling=downsampling_params,
            normalization=normalization_params,
            standardization=standardization_params,
        )
        logger.info("Preprocessing complete: %s", params)
        return GridData(data, preprocessing_params=params)

    def _normalize_xyz(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[Axis, NormalizationParams]]:
        """Apply min-max normalization to X, Y, Z columns."""
        df = data.copy()
        axis_params: dict[Axis, NormalizationParams] = {}
        for axis in [Axis.X, Axis.Y, Axis.Z]:
            df[axis.value], params = normalize(df[axis.value])
            axis_params[axis] = params
        return df, axis_params

    def _standardize_v(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, StandardizationParams]:
        """Apply z-score standardization to V column."""
        df = data.copy()
        df["V"], params = standardize(df["V"])
        return df, params

    def _downsample_data(
        self,
        data: pd.DataFrame,
        statistic: (
            DownsamplingStatistic | str | Callable[..., pd.DataFrame]
        ) = DownsamplingStatistic.MEAN,
    ) -> pd.DataFrame:
        """Downsample data by averaging blocks of given resolution."""
        assert self.downsampling_res is not None
        res = self.downsampling_res
        idfs = []
        for id_val in self.griddata.data.index.get_level_values("ID").unique():
            idf = self.griddata.data.loc[
                self.griddata.data.index.get_level_values("ID") == id_val
            ].reset_index()

            x = idf["X"].iloc[0]
            y = idf["Y"].iloc[0]
            idf = idf[["Z", "V"]]

            grouped = idf.groupby(idf["Z"].apply(lambda z: res * round(z / res)))
            idf = grouped[["V"]].apply(_apply_downsampling, downsampling_func=statistic)

            idf["X"] = x
            idf["Y"] = y
            idf["ID"] = id_val
            idf = idf.reset_index()
            idfs.append(idf)

        return cast(pd.DataFrame, pd.concat(idfs))


def _apply_downsampling(
    grouped_df: pd.DataFrame,
    downsampling_func: DownsamplingStatistic | str | Callable[..., pd.DataFrame],
) -> pd.DataFrame | pd.Series[float]:
    """Apply a downsampling statistic to a grouped DataFrame."""
    if callable(downsampling_func) and not isinstance(downsampling_func, str):
        return downsampling_func(grouped_df)

    stat = DownsamplingStatistic(downsampling_func)
    match stat:
        case DownsamplingStatistic.MEAN:
            return grouped_df[["V"]].mean()
        case DownsamplingStatistic.MAX:
            return grouped_df[["V"]].max()
        case DownsamplingStatistic.MIN:
            return grouped_df[["V"]].min()
        case DownsamplingStatistic.MEDIAN:
            return grouped_df[["V"]].median()
        case DownsamplingStatistic.SUM:
            return grouped_df[["V"]].sum()
        case DownsamplingStatistic.QUANTILE75:
            return grouped_df[["V"]].quantile(0.75)


def reverse_preprocessing(griddata: GridData) -> GridData:
    """Reverse all reversible preprocessing transformations.

    Reverses normalization of XYZ and standardization of V.
    Downsampling cannot be reversed.

    Args:
        griddata: GridData with preprocessing_params set.

    Returns:
        New GridData with reversed transformations.

    Raises:
        ValueError: If no preprocessing params are present.
    """
    params = griddata.preprocessing_params
    if params is None:
        msg = "No preprocessing has been applied to the data"
        raise ValueError(msg)

    data = griddata.data.copy().reset_index()

    if params.normalization is not None:
        for axis in [Axis.X, Axis.Y, Axis.Z]:
            norm = params.normalization[axis]
            data[axis.value] = data[axis.value] * (norm.max - norm.min) + norm.min

    if params.standardization is not None:
        std = params.standardization
        data["V"] = data["V"] * std.std + std.mean

    return GridData(data)
