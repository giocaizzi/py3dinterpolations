"""3D volume plot using plotly."""

import numpy as np
import plotly.graph_objs as go

from ..modelling.modeler import Modeler
from ..modelling.preprocessor import reverse_preprocessing


def plot_3d_model(
    modeler: Modeler,
    plot_points: bool = False,
    scale_points: float = 1.0,
    volume_kwargs: dict[str, object] | None = None,
) -> go.Figure:
    """Plot 3D interpolation result as a plotly Volume.

    Args:
        modeler: Modeler with prediction results.
        plot_points: Whether to overlay training points.
        scale_points: Scale factor for point marker sizes.
        volume_kwargs: Extra kwargs for go.Volume.

    Returns:
        Plotly Figure.
    """
    if volume_kwargs is None:
        volume_kwargs = {}

    gd_reversed = reverse_preprocessing(modeler.griddata)

    assert modeler.result is not None
    # ZYX -> XYZ
    values = np.einsum("ZXY->XYZ", modeler.result.interpolated)

    data: list[go.Volume | go.Scatter3d] = [
        go.Volume(
            x=modeler.grid.mesh["X"].flatten(),
            y=modeler.grid.mesh["Y"].flatten(),
            z=modeler.grid.mesh["Z"].flatten(),
            value=values.flatten(),
            opacityscale=[(0, 0), (1, 1)],
            cmin=gd_reversed.specs.vmin,
            cmax=gd_reversed.specs.vmax,
            **volume_kwargs,
        ),
    ]

    if plot_points:
        params = modeler.griddata.preprocessing_params
        if params is not None:
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

    fig = go.Figure(data=data)
    fig.update_scenes(aspectmode="data")
    return fig
