"""plotting methods"""
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import plotly.graph_objs as go

import numpy as np

from ..core.griddata import GridData
from ..modelling.modeler import Modeler
from ..modelling.preprocessor import reverse_preprocessing


def _number_of_plots(n: int, n_cols: int = 4) -> tuple:
    """Determine number of rows and columns
    for plotting with respect to remainder of division."""
    n_rows = (n + n_cols - 1) // n_cols
    return n_rows, n_cols


def plot_downsampling(
    original_griddata: GridData,
    downsampled_griddata: GridData,
) -> Figure:
    if (
        "normalization" in downsampled_griddata.preprocessor_params
        or "standardization" in downsampled_griddata.preprocessor_params
    ):
        downsampled_griddata = reverse_preprocessing(downsampled_griddata)
    # get data
    df = original_griddata.data.copy().reset_index()
    resampled = downsampled_griddata.data.copy().reset_index()

    # Get unique ID values
    unique_ids = df["ID"].unique().tolist()

    # determine number of axes
    num_rows, num_cols = _number_of_plots(len(unique_ids))
    # Create the figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10), dpi=300)
    fig.subplots_adjust(wspace=0.3, hspace=0.7)

    # Loop over each ID
    for idx, id_to_plot in enumerate(unique_ids):
        # Determine the current row and column
        row = idx // num_cols
        col = idx % num_cols

        # Get the current axis
        ax = axes[row, col]

        # Extract the X and Y values
        x_values = df[df["ID"] == id_to_plot]["V"]
        y_values = df[df["ID"] == id_to_plot]["Z"]
        ax.plot(x_values, y_values, "-", linewidth=0.5, zorder=1)

        # Create the sampled plot
        x_values = resampled[resampled["ID"] == id_to_plot]["V"]
        y_values = resampled[resampled["ID"] == id_to_plot]["Z"]

        # Create the sampled plot
        ax.plot(x_values, y_values, "o", markersize=1, zorder=10)

        # style
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.tick_params(axis="x", which="major", rotation=15)
        ax.tick_params(axis="both", which="major", labelsize=3)
        ax.set_title(f"{id_to_plot}", fontsize=5)

        # Hide empty subplots
        if len(unique_ids) < num_rows * num_cols:
            for i in range(len(unique_ids), num_rows * num_cols):
                row = i // num_cols
                col = i % num_cols
                axes[row, col].set_visible(False)

        # xlims to the extends of the original griddata
        ax.set_xlim(
            xmin=0,
            xmax=original_griddata.specs.vmax
            + (original_griddata.specs.vmax / 100) * 10,
            auto=False,
        )
    # add title
    fig.suptitle(f"{list(original_griddata.columns)[0]}", fontsize=10)
    # avoid printing the plot when calling within jupyter notebook
    plt.close(fig)
    return fig


def plot_3d_model(
    modeler: Modeler,
    plot_points: bool = False,
    scale_points=1.0,
    volume_kwargs={},
) -> go.Figure:
    """plot 3d model"""

    gd_reversed = reverse_preprocessing(modeler.griddata)

    # pyplot convention of indexing
    # zyx -> xyz
    values = np.einsum("ZXY->XYZ", modeler.results["interpolated"])

    data = [
        go.Volume(
            x=modeler.grid3d.mesh["X"].flatten(),
            y=modeler.grid3d.mesh["Y"].flatten(),
            z=modeler.grid3d.mesh["Z"].flatten(),
            value=values.flatten(),
            opacityscale=[(0, 0), (1, 1)],
            cmin=gd_reversed.specs.vmin,
            cmax=gd_reversed.specs.vmax,
            **volume_kwargs,
        ),
    ]

    if plot_points:
        # get correct points, if preprocessing was applied
        if modeler.griddata.preprocessor_params:
            points = gd_reversed.data.copy().reset_index()
        else:
            points = modeler.griddata.data.copy().reset_index()

        data.append(
            go.Scatter3d(
                x=points["X"],
                y=points["Y"],
                z=points["Z"],
                mode="markers",
                marker=dict(
                    size=points["V"].to_list(),
                    sizemode="area",
                    sizeref=2.0 * max(points["V"]) / (scale_points**2),
                    color=points["V"],
                    sizemin=1,
                ),
            )
        )

    # plot figure
    fig = go.Figure(data=data)
    fig.update_scenes(aspectmode="data")
    return fig


SLICING_AXIS = {
    "X": {
        "X'": "Y",
        "Y'": "Z",
    },
    "Y": {
        "X'": "X",
        "Y'": "Z",
    },
    "Z": {
        "X'": "X",
        "Y'": "Y",
    },
}


def plot_2d_model(
    modeler: Modeler,
    axis: str = "Z",
    plot_points: bool = False,
    annotate_points: bool = False,
    figure_width: float = 8,
):
    """plot 2d along axis

    Args:
        modeler (Modeler): modeler object
        axis (str, optional): axis along which to plot. Defaults to "Z".
        plot_points (bool, optional): plot points. Defaults to False.
        figure_width (float, optional): figure width in inches. Less than
            8 results in weird looks. Defaults to 8.

    Returns:
        Figure: matplotlib figure

    Example:
        >>> fig2 = plot_2d_along_axis(
        >>>    model,
        >>>    plot_points=True,
        >>> )
    """
    axis_data = modeler.grid3d.grid[axis]

    # determine number of axes
    num_rows, num_cols = _number_of_plots(len(axis_data), n_cols=2)

    # figure with gridspec
    figure_height_ratio = 1.25
    fig = plt.figure(
        dpi=300, figsize=(figure_width, figure_width * figure_height_ratio)
    )
    gs = gridspec.GridSpec(
        num_rows,
        num_cols + 1,
        width_ratios=[1] * num_cols + [0.1],
    )
    # Create regular subplots in the first n_cols columns
    axes = []
    for row in range(num_rows):
        for col in range(num_cols):
            axes.append(plt.subplot(gs[row, col]))
    # Create the single subplot in the rightmost column
    colorbar_ax = plt.subplot(gs[:, -1])
    # disable visibility of spines and ticks
    colorbar_ax.spines["top"].set_visible(False)
    colorbar_ax.spines["bottom"].set_visible(False)
    colorbar_ax.spines["left"].set_visible(False)
    colorbar_ax.spines["right"].set_visible(False)
    colorbar_ax.set_xticks([])
    colorbar_ax.set_yticks([])
    colorbar_ax.set_xticklabels([])
    colorbar_ax.set_yticklabels([])
    # inset axes
    colorbar_inset_ax = inset_axes(
        colorbar_ax, width="100%", height="50%", loc="center"
    )

    # scale colors on overall vmin and vmax, before preprocessing
    gd_reversed = reverse_preprocessing(modeler.griddata)
    norm = plt.Normalize(gd_reversed.specs.vmin, gd_reversed.specs.vmax)

    # loop over axes
    for ax, i in zip(axes, range(len(axis_data))):
        # results is respahed #18
        # numpy convetion of indexing
        # pykrige output zyx
        if axis == "Z":
            matrix = modeler.results["interpolated"][i, :, :]
        elif axis == "Y":
            matrix = modeler.results["interpolated"][:, i, :]
        elif axis == "X":
            matrix = modeler.results["interpolated"][:, :, i]
        else:
            raise NotImplementedError(
                f"axis {axis} not implemented. Choose from {SLICING_AXIS.keys()}"
            )
        # plot interpolated
        img = ax.imshow(
            matrix.squeeze(),  # remove singleton dimensions
            origin="lower",
            extent=[
                modeler.grid3d.get_axis(SLICING_AXIS[axis]["X'"]).min,  # X' min
                modeler.grid3d.get_axis(SLICING_AXIS[axis]["X'"]).max,  # X' max
                modeler.grid3d.get_axis(SLICING_AXIS[axis]["Y'"]).min,  # Y' min
                modeler.grid3d.get_axis(SLICING_AXIS[axis]["Y'"]).max,  # Y' max
            ],
            cmap="plasma",
            norm=norm,
        )
        # slice griddata
        from_value = modeler.grid3d.grid[axis][i]
        # works only for regular grids
        to_value = from_value + modeler.grid3d.gridres

        # plot points
        if plot_points:
            points_df = gd_reversed.data.copy().reset_index()

            # get points conatined in each grid cell
            points = points_df[
                (points_df[axis] >= from_value) & (points_df[axis] < to_value)
            ].copy()

            # sort by value to plot highest values on top
            points.sort_values(by=["V"], inplace=True)
            ax.scatter(
                points[SLICING_AXIS[axis]["X'"]],
                points[SLICING_AXIS[axis]["Y'"]],
                c=points["V"],
                cmap="plasma",
                norm=norm,
                s=figure_width / 2,
            )
            if annotate_points:
                # annotate points
                for idx, row in points.iterrows():
                    ax.annotate(
                        "{:.0f}".format(row["V"]),
                        xy=(
                            row[SLICING_AXIS[axis]["X'"]],
                            row[SLICING_AXIS[axis]["Y'"]],
                        ),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=figure_width / 2,
                    )
        # subplot label
        ax.set_title(f"{axis} = {from_value}÷{to_value} m")
    # suptitle
    fig.suptitle(f"Along {axis} axis")

    # if write:
    #     ## write grid
    #     write_asc_grid(
    #         gridx,
    #         gridy,
    #         interpolated_original[i, :, :],
    #         filename=f"{foldername}/Z_{i+0.5}.asc",
    #     )

    # colorbar
    plt.colorbar(
        img,
        cax=colorbar_inset_ax,
        format="%.0f",
        fraction=0.1,
    )

    # Hide empty subplots
    if len(axis_data) < num_rows * num_cols:
        for i in range(len(axis_data), num_rows * num_cols):
            axes[i].set_visible(False)
    return fig
