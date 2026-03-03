"""Downsampling comparison plot."""

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.figure import Figure

from ..core.griddata import GridData
from ..modelling.preprocessor import reverse_preprocessing
from .utils import number_of_plots


def plot_downsampling(
    original_griddata: GridData,
    downsampled_griddata: GridData,
) -> Figure:
    """Plot original vs downsampled data per ID.

    Args:
        original_griddata: Original data before downsampling.
        downsampled_griddata: Data after downsampling
            (may have normalization/standardization).

    Returns:
        Matplotlib Figure with comparison plots.
    """
    params = downsampled_griddata.preprocessing_params
    if params is not None and (
        params.normalization is not None or params.standardization is not None
    ):
        downsampled_griddata = reverse_preprocessing(downsampled_griddata)

    df = original_griddata.data.copy().reset_index()
    resampled = downsampled_griddata.data.copy().reset_index()
    unique_ids = df["ID"].unique().tolist()

    num_rows, num_cols = number_of_plots(len(unique_ids))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10), dpi=300)
    fig.subplots_adjust(wspace=0.3, hspace=0.7)

    for idx, id_to_plot in enumerate(unique_ids):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        x_values = df[df["ID"] == id_to_plot]["V"]
        y_values = df[df["ID"] == id_to_plot]["Z"]
        ax.plot(x_values, y_values, "-", linewidth=0.5, zorder=1)

        x_values = resampled[resampled["ID"] == id_to_plot]["V"]
        y_values = resampled[resampled["ID"] == id_to_plot]["Z"]
        ax.plot(x_values, y_values, "o", markersize=1, zorder=10)

        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.tick_params(axis="x", which="major", rotation=15)
        ax.tick_params(axis="both", which="major", labelsize=3)
        ax.set_title(f"{id_to_plot}", fontsize=5)

        if len(unique_ids) < num_rows * num_cols:
            for i in range(len(unique_ids), num_rows * num_cols):
                r = i // num_cols
                c = i % num_cols
                axes[r, c].set_visible(False)

        ax.set_xlim(
            xmin=0,
            xmax=original_griddata.specs.vmax
            + (original_griddata.specs.vmax / 100) * 10,
            auto=False,
        )

    fig.suptitle(f"{next(iter(original_griddata.columns))}", fontsize=10)
    plt.close(fig)
    return fig
