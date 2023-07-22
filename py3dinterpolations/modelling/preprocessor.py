"""preprocessor module"""

import pandas as pd

from ..core.griddata import GridData
from typing import Union


class Preprocessor:
    """Preprocessor class

    Preprocess data before interpolation. Preprocessing includes:
        - Downsampling, by taking the mean of blocks of given resolution
        - Normalization of X Y Z
        - Standardization of V

    By calling the preprocess method, the data is preprocessed and a new
    GridData object is returned with the preprocessed data and the
    preprocessing parameters set.

    Args:
        griddata (GridData): GridData object to preprocess
        downsampling_res (Union[float, None]): resolution to downsample data
            by taking the mean of blocks of given resolution. If None, no
            downsampling is applied. Default is None.
        normalize_xyz (bool): whether to normalize X Y Z. Default is True.
        standardize_v (bool): whether to standardize V. Default is True.

    Attributes:
        griddata (GridData): GridData object to preprocess
        downsampling_res (Union[float, None]): resolution to downsample data
            by taking the mean of blocks of given resolution. If None, no
            downsampling is applied. Default is None.
        normalize_xyz (bool): whether to normalize X Y Z. Default is True.
        standardize_v (bool): whether to standardize V. Default is True.
        preprocessor_params (dict): dictionary with the parameters of the
            preprocessing. It is set after calling the preprocess method.::

                preprocessor_params = {
                    "downsampling": {
                        "resolution": downsampling_res,
                    },
                    "normalization": {
                        "X": {
                            "min": min_value_of_X,
                            "max": max_value_of_X,
                        },
                        "Y": {
                            "min": min_value_of_Y,
                            "max": max_value_of_Y,
                        },
                        "Z": {
                            "min": min_value_of_Z,
                            "max": max_value_of_Z,
                        },
                    },
                    "standardization": {
                        "mean": mean_value_of_V,
                        "std": std_value_of_V,
                    },

                }

    Examples:
        >>> # preprocess data
        >>> preprocessor = Preprocessor(
        >>>     griddata,
        >>>     downsampling_res=0.1,
        >>>     normalize_xyz=True,
        >>>     standardize_v=True,
        >>> )
        >>> griddata = preprocessor.preprocess()
    """

    preprocessor_params = {}

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
        """preprocess data

        Returns:
            GridData: new GridData object with the preprocessed data
                and prerpocessing parameters set

        """
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
        return GridData(data, preprocessor_params=self.preprocessor_params)

    def _normalize_xyz(self, data: pd.DataFrame) -> pd.DataFrame:
        """apply normalization to X Y Z

        Args:
            data (pd.DataFrame): data to normalize

        Returns:
            pd.DataFrame: normalized data
        """
        df = data.copy()
        self.preprocessor_params["normalization"] = {}
        for axis in ["X", "Y", "Z"]:
            df[axis], axis_params = _normalize(df[axis])
            self.preprocessor_params["normalization"][axis] = axis_params
        return df

    def _standardize_v(self, data: pd.DataFrame) -> pd.DataFrame:
        """apply standardization to V"""
        df = data.copy()
        df["V"], params = _standardize(df["V"])
        self.preprocessor_params["standardization"] = params
        return df

    def _downsample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """downsample data making the average by blocks of given resolution

        TODO: add option to downsample by selecting the statistic
            taking the median, or the max

        Args:
            data (pd.DataFrame): data to downsample

        Returns:
            pd.DataFrame: downsampled data
        """
        self.preprocessor_params["downsampling"] = {"resolution": self.downsampling_res}
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
                    lambda x: self.preprocessor_params["downsampling"]["resolution"]
                    * round(x / self.preprocessor_params["downsampling"]["resolution"])
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
    if griddata.preprocessor_params is None:
        raise ValueError("No preprocessing has been applied to the data")
    else:
        # get data
        data = griddata.data.copy().reset_index()
        if "normalization" in griddata.preprocessor_params:
            # reverse normalization of X Y Z
            # original_value = normalized_value * (max_value - min_value) + min_value
            for axis in ["X", "Y", "Z"]:
                data[axis] = (
                    data[axis]
                    * (
                        griddata.preprocessor_params["normalization"][axis]["max"]
                        - griddata.preprocessor_params["normalization"][axis]["min"]
                    )
                    + griddata.preprocessor_params["normalization"][axis]["min"]
                )
        if "standardization" in griddata.preprocessor_params:
            # reverse standardization of V
            data["V"] = (
                data["V"] * griddata.preprocessor_params["standardization"]["std"]
                + griddata.preprocessor_params["standardization"]["mean"]
            )
            # return
        return GridData(data)
