"""core classes"""

import pandas as pd
import numpy as np


class GridData:
    """GridData class

    Class for 3D grid data.

    As defaults, looks for X, Y, Z, V columns in the data.
    Otherwise, custom column names can be provided.

    This object keeps track of the transformations (preprocessing)
    applied to the data, by storing the relative parameters in the
    `preprocessor_params` attribute.

    Args:
        data (pd.DataFrame): source data
        ID (str): column name of the ID, default: "ID"
        X (str): column name of the X, default: "X"
        Y (str): column name of the Y, default: "Y"
        Z (str): column name of the Z, default: "Z"
        V (str): column name of the V, default: "V"
        preprocessor_params (dict): preprocessing parameters used
            by the Preprocessor object that has generated the initiated
            GridData object, default: {}

    Attributes:
        data (pd.DataFrame): data in GridData.data standard
        preprocessor_params (dict): preprocessing parameters used
            by the Preprocessor object that has generated the initiated
            GridData object, default: {}
        columns (dict): column names mapping

    Properties:
        numpy_data (np.ndarray): numpy data of X,Y,Z,V
        specs (GridDataSpecs): specs of the GridData
    """

    data = pd.DataFrame()
    preprocessor_params = {}
    columns = {}

    def __init__(
        self,
        data: pd.DataFrame,
        ID: str = "ID",
        X: str = "X",
        Y: str = "Y",
        Z: str = "Z",
        V: str = "V",
        preprocessor_params: dict = {},
    ):
        # set params of used by the Preprocessor object
        # this is done with a new object with preprocessed data
        if isinstance(preprocessor_params, dict):
            self.preprocessor_params = preprocessor_params
        else:
            raise ValueError("preprocessor_params must be a dict")

        # set column names dictionary
        self.columns = {
            "ID": ID,
            "X": X,
            "Y": Y,
            "Z": Z,
            "V": V,
        }

        # set data to GridData.data standard
        self.data = self._set_data(data)

    def _set_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """set data to GridData.data standard

        - only equired columns
        - column names to default X, Y, Z, V
        - index to X, Y, ID, Z

        Returns:
            pd.DataFrame: data in GridData.data standard
        """
        # slice and get data
        df = data.copy()[list(self.columns.values())]
        # rename columns
        df.rename(
            columns={v: k for k, v in self.columns.items()},  # inverted
            inplace=True,
        )
        # sorted index
        df.set_index(["ID", "X", "Y", "Z"], inplace=True)
        df["V"] = df["V"].astype(float).copy()
        df.sort_index(inplace=True, ascending=False)

        return df

    @property
    def specs(self):
        """specs of the GridData

        Initializes a GridDataSpecs object
        with the current GridData object
        to get the specs of the GridData

        Returns:
            GridDataSpecs: specs of the GridData
        """
        if self.data is None:
            raise ValueError("Specs can't be calculated without data")
        else:
            return GridDataSpecs(self)

    @property
    def numpy_data(self) -> np.ndarray:
        """get numpy data

        Returns:
            np.ndarray: numpy data of X,Y,Z,V
        """
        return self.data.reset_index()[["X", "Y", "Z", "V"]].to_numpy()


class GridDataSpecs:
    """GridDataSpecs class

    Class for finding specs of data in GridData
    - xmin, xmax, ymin, ymax, zmin, zmax (coordinates)
    - vmin, vmax (values)

    Args:
        griddata (GridData): GridData object
    """

    def __init__(self, griddata: GridData):
        self.data = griddata.data.copy().reset_index()

    @property
    def xmin(self) -> float:
        """xmin"""
        return self.data["X"].min()

    @property
    def xmax(self) -> float:
        """xmax"""
        return self.data["X"].max()

    @property
    def ymin(self) -> float:
        """ymin"""
        return self.data["Y"].min()

    @property
    def ymax(self) -> float:
        """ymax"""
        return self.data["Y"].max()

    @property
    def zmin(self) -> float:
        """zmin"""
        return self.data["Z"].min()

    @property
    def zmax(self) -> float:
        """zmax"""
        return self.data["Z"].max()

    @property
    def vmin(self) -> float:
        """vmin"""
        return self.data["V"].min()

    @property
    def vmax(self) -> float:
        """vmax"""
        return self.data["V"].max()

    @property
    def xsize(self) -> float:
        """xsize"""
        return np.abs(self.xmax - self.xmin)

    @property
    def ysize(self) -> float:
        """ysize"""
        return np.abs(self.ymax - self.ymin)

    @property
    def zsize(self) -> float:
        """zsize"""
        return np.abs(self.zmax - self.zmin)
