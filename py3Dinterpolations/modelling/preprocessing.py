"""preprocessing objects"""

import pandas as pd

from ..core import GridData
from typing import Union


class Preprocessing:
    """preprocessing class"""

    preprocessing_params = {}

    def __init__(
        self,
        griddata: GridData,
        downsampling_res: Union[float, None] = None,
        normalize_xyz: bool = True,
        standardize_v: bool = True,
    ):
        self.griddata = griddata
        self.downsampling_res = downsampling_res
        self.normalize_xyz = normalize_xyz
        self.standardize_v = standardize_v

    def preprocess(self) -> GridData:
        # get data
        data = self.griddata.data.copy().reset_index()[["ID", "X", "Y", "Z", "V"]]

        # first dowmsample
        if self.downsampling_res is not None:
            data = self._downsample_data(data)

        # normalize
        if self.normalize_xyz:
            data = self._normalize_xyz(data)

        # standardize
        if self.standardize_v:
            data = self._standardize_v(data)

        # return new object with preprocessed data and parameters
        return GridData(data, preprocessing_params=self.preprocessing_params)

    def _normalize_xyz(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        self.preprocessing_params["normalization"] = {}
        for axis in ["X", "Y", "Z"]:
            df[axis], axis_params = _normalize(df[axis])
            self.preprocessing_params["normalization"][axis] = axis_params
        return df

    def _standardize_v(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["V"], params = _standardize(df["V"])
        self.preprocessing_params["standardization"] = params
        return df

    def _downsample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """downsample data making the average by blocks of given resolution"""
        self.preprocessing_params["downsampling"] = {
            "resolution": self.downsampling_res
        }
        idfs = []
        # loop over unique ids
        for id in self.griddata.data.index.get_level_values("ID").unique():
            # filter by id
            idf = self.griddata.data.loc[
                self.griddata.data.index.get_level_values("ID") == id
            ].reset_index()
            # save x,y values
            x = idf["X"][0]
            y = idf["Y"][0]
            # extract z
            idf = idf[["Z", "V"]]
            # downsample by grouping in blocks of given resolution
            # and taking the mean
            idf = idf.groupby(
                idf["Z"].apply(
                    lambda x: self.preprocessing_params["downsampling"]["resolution"]
                    * round(x / self.preprocessing_params["downsampling"]["resolution"])
                )
            )[["V"]].mean()
            # new downsampled df
            idf["X"] = x
            idf["Y"] = y
            idf["ID"] = id
            idf.reset_index(inplace=True)  # reset index resulting from groupby
            idfs.append(idf)
        # return downsampled grid data

        return pd.concat(idfs)


def _standardize(series: pd.Series) -> tuple:
    series = series.copy()
    # save standardization parameters
    params = {
        "mean": series.mean(),
        "std": series.std(),
    }
    # standardize
    series = (series - params["mean"]) / (params["std"])
    return series, params


def _normalize(series: pd.Series) -> tuple:
    series = series.copy()
    # save normalization parameters
    params = {
        "min": series.min(),
        "max": series.max(),
    }
    # normalize
    series = (series - params["min"]) / (params["max"] - params["min"])
    return series, params


def reverse_preprocessing(griddata: GridData) -> GridData:
    """reverse preprocessing of whole GridData object

    reverse all reversible transformations that have been
    applied to the GridData object.

    This method reverses the operations of preprocessing of:
        - Normalization of X Y Z
        - Standardization of V

    Note:
        It cannot reverse downsampling!

    Returns:
        GridData: GridData object with the data with reversed preprocessing
    """
    if griddata.preprocessing_params is None:
        raise ValueError("No preprocessing has been applied to the data")
    else:
        # get data
        data = griddata.data.copy().reset_index()
        if "normalization" in griddata.preprocessing_params:
            # reverse normalization of X Y Z
            # original_value = normalized_value * (max_value - min_value) + min_value
            for axis in ["X", "Y", "Z"]:
                data[axis] = (
                    data[axis]
                    * (
                        griddata.preprocessing_params["normalization"][axis]["max"]
                        - griddata.preprocessing_params["normalization"][axis]["min"]
                    )
                    + griddata.preprocessing_params["normalization"][axis]["min"]
                )
        if "standardization" in griddata.preprocessing_params:
            # reverse standardization of V
            data["V"] = (
                data["V"] * griddata.preprocessing_params["standardization"]["std"]
                + griddata.preprocessing_params["standardization"]["mean"]
            )
            # return
        return GridData(data)
