"""plotting methods"""
from matplotlib import ticker
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import plotly.graph_objs as go

from ..core.griddata import GridData
from ..modelling.modeler import Modeler
from ..modelling.preprocessor import reverse_preprocessing

def number_of_plots(n: int,n_cols:int=4) -> tuple:
    """determine number of rows and columns for plotting"""
    n_rows = int(n / n_cols) + 1
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
    num_rows, num_cols = number_of_plots(len(unique_ids))
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


def plot_model_epsilon(model) -> None:
    """plot model epsilon native pykrige method"""
    model.plot_epsilon_residuals()


def plot_3d_model(
    modeler: Modeler, plot_points: bool = False, scale_points=1.0, **kwargs
) -> go.Figure:
    """plot 3d model"""
    data = [
        go.Volume(
            x=modeler.grid3d.mesh["X"].flatten(),
            y=modeler.grid3d.mesh["Y"].flatten(),
            z=modeler.grid3d.mesh["Z"].flatten(),
            value=modeler.results["interpolated"].flatten(),
            opacityscale=[(0, 0), (1, 1)],
            **kwargs,
        ),
    ]

    if plot_points:
        # get correct points, if preprocessing was applied
        if modeler.griddata.preprocessor_params:
            gd_reversed = reverse_preprocessing(modeler.griddata)
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


def plot_2d_along_axis(
    modeler: Modeler,
    axis: str = "z",
):
    if axis != "z":
        raise NotImplementedError("Only z-axis is implemented")
    else:
            # determine number of axes
        num_rows, num_cols = number_of_plots(len(modeler.grid3d.grid["Z"]))
        fig, axes = plt.subplots(num_rows, num_cols figsize=(120, 120))
        # norm = plt.Normalize(
        #     modeler.griddata.specs.vmin, modeler.griddata.specs.vmax
        # )
        for ax, i in zip(axes, range(len(axes))):
            img = ax.imshow(
                modeler.results["interpolated"][i, :, :],
                origin="lower",
                extent=[xmin, xmax, ymin, ymax],
                cmap="jet",
                norm=norm,
            )
            ax.set_title(f"Z = {i+0.5} da p.c.")
            points = griddata[
                (griddata["Z"] >= i) & (griddata["Z"] < i + 1)
            ].copy()
            points.sort_values(by=[colname], inplace=True)
            ax.scatter(
                points["X"], points["Y"], c=points[colname], cmap="jet", norm=norm
            )
            for idx, row in points.iterrows():
                ax.annotate(
                    "{:.0f}".format(row[colname]),
                    xy=(row["X"], row["Y"]),
                    xytext=(2, 2),
                    textcoords="offset points",
                )
            fig.colorbar(img, ax=ax, format="%.0f")
            # if write:
            #     ## write grid
            #     write_asc_grid(
            #         gridx,
            #         gridy,
            #         interpolated_original[i, :, :],
            #         filename=f"{foldername}/Z_{i+0.5}.asc",
            #     )
            return fig
