"""2D slice plots along a given axis."""

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..modelling.modeler import Modeler
from ..modelling.preprocessor import reverse_preprocessing
from .utils import SLICING_AXIS, number_of_plots


def plot_2d_model(
    modeler: Modeler,
    axis: str = "Z",
    plot_points: bool = False,
    annotate_points: bool = False,
    figure_width: float = 8,
) -> Figure:
    """Plot 2D slices of a 3D interpolation along an axis.

    Args:
        modeler: Modeler with prediction results.
        axis: Axis to slice along ("X", "Y", or "Z").
        plot_points: Whether to overlay training points on slices.
        annotate_points: Whether to annotate point values.
        figure_width: Figure width in inches.

    Returns:
        Matplotlib Figure.
    """
    assert modeler.result is not None
    axis_data = modeler.grid.grid[axis]

    num_rows, num_cols = number_of_plots(len(axis_data), n_cols=2)

    figure_height_ratio = 1.25
    fig = plt.figure(
        dpi=300, figsize=(figure_width, figure_width * figure_height_ratio)
    )
    gs = gridspec.GridSpec(
        num_rows,
        num_cols + 1,
        width_ratios=[1] * num_cols + [0.1],
    )

    axes = []
    for row in range(num_rows):
        for col in range(num_cols):
            axes.append(plt.subplot(gs[row, col]))

    colorbar_ax = plt.subplot(gs[:, -1])
    colorbar_ax.spines["top"].set_visible(False)
    colorbar_ax.spines["bottom"].set_visible(False)
    colorbar_ax.spines["left"].set_visible(False)
    colorbar_ax.spines["right"].set_visible(False)
    colorbar_ax.set_xticks([])
    colorbar_ax.set_yticks([])
    colorbar_ax.set_xticklabels([])
    colorbar_ax.set_yticklabels([])

    colorbar_inset_ax = inset_axes(
        colorbar_ax, width="100%", height="50%", loc="center"
    )

    gd_reversed = reverse_preprocessing(modeler.griddata)
    norm = plt.Normalize(gd_reversed.specs.vmin, gd_reversed.specs.vmax)

    img = None
    for ax, i in zip(axes, range(len(axis_data)), strict=False):
        if axis == "Z":
            matrix = modeler.result.interpolated[i, :, :]
        elif axis == "Y":
            matrix = modeler.result.interpolated[:, i, :]
        elif axis == "X":
            matrix = modeler.result.interpolated[:, :, i]
        else:
            keys = list(SLICING_AXIS.keys())
            msg = f"axis {axis} not implemented. Choose from {keys}"
            raise NotImplementedError(msg)

        img = ax.imshow(
            matrix.squeeze(),
            origin="lower",
            extent=[
                modeler.grid.get_axis(SLICING_AXIS[axis]["X'"]).min,
                modeler.grid.get_axis(SLICING_AXIS[axis]["X'"]).max,
                modeler.grid.get_axis(SLICING_AXIS[axis]["Y'"]).min,
                modeler.grid.get_axis(SLICING_AXIS[axis]["Y'"]).max,
            ],
            cmap="plasma",
            norm=norm,
        )

        from_value = modeler.grid.grid[axis][i]
        to_value = from_value + modeler.grid.gridres

        if plot_points:
            points_df = gd_reversed.data.copy().reset_index()
            points = points_df[
                (points_df[axis] >= from_value) & (points_df[axis] < to_value)
            ].copy()
            points = points.sort_values(by=["V"])
            ax.scatter(
                points[SLICING_AXIS[axis]["X'"]],
                points[SLICING_AXIS[axis]["Y'"]],
                c=points["V"],
                cmap="plasma",
                norm=norm,
                s=figure_width / 2,
            )
            if annotate_points:
                for _idx, row in points.iterrows():
                    ax.annotate(
                        f"{row['V']:.0f}",
                        xy=(
                            row[SLICING_AXIS[axis]["X'"]],
                            row[SLICING_AXIS[axis]["Y'"]],
                        ),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=figure_width / 2,
                    )

        ax.set_title(f"{axis} = {from_value}\u00f7{to_value} m")

    fig.suptitle(f"Along {axis} axis")

    if img is not None:
        plt.colorbar(
            img,
            cax=colorbar_inset_ax,
            format="%.0f",
            fraction=0.1,
        )

    if len(axis_data) < num_rows * num_cols:
        for i in range(len(axis_data), num_rows * num_cols):
            axes[i].set_visible(False)

    return fig
