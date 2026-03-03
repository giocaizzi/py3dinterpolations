# Visualization

py3dinterpolations provides built-in visualization for interpolation results
using matplotlib (2D slices) and plotly (3D interactive volumes).

## 3D interactive volume

[`plot_3d_model`][py3dinterpolations.plotting.plot_3d.plot_3d_model] creates an
interactive 3D volume plot using plotly:

```python
from py3dinterpolations import plot_3d_model

fig = plot_3d_model(
    modeler,
    plot_points=True,
    scale_points=10,
    volume_kwargs={"surface_count": 10},
)
fig.show()
```

<iframe src="../assets/3d_plot.html" width="100%" height="500" frameborder="0"></iframe>

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `plot_points` | `False` | Overlay training data points on the volume |
| `scale_points` | `1.0` | Scale factor for point marker sizes |
| `volume_kwargs` | `None` | Extra kwargs passed to `plotly.graph_objs.Volume` |

The returned `go.Figure` can be customized further using the plotly API, saved
as HTML, or displayed inline in Jupyter notebooks.

## 2D slices

[`plot_2d_model`][py3dinterpolations.plotting.plot_2d.plot_2d_model] creates a
grid of 2D slice plots along a specified axis:

```python
from py3dinterpolations import plot_2d_model

fig = plot_2d_model(
    modeler,
    axis="Z",
    plot_points=True,
    annotate_points=True,
)
fig.savefig("slices.png", dpi=300, bbox_inches="tight")
```

Each subplot shows one slice along the chosen axis, with optional data point
overlay and value annotations.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `axis` | `"Z"` | Axis to slice along: `"X"`, `"Y"`, or `"Z"` |
| `plot_points` | `False` | Overlay training points on each slice |
| `annotate_points` | `False` | Annotate point values (requires `plot_points=True`) |
| `figure_width` | `8` | Figure width in inches |

## Downsampling comparison

[`plot_downsampling`][py3dinterpolations.plotting.downsampling.plot_downsampling]
compares original and downsampled data per borehole ID:

```python
from py3dinterpolations.plotting import plot_downsampling
from py3dinterpolations.modelling.preprocessor import Preprocessor

preprocessed = Preprocessor(griddata, downsampling_res=5.0).preprocess()
fig = plot_downsampling(griddata, preprocessed)
```

This produces a grid of line plots showing the original data (line) and
downsampled values (dots) for each borehole, helping assess how downsampling
affects the data.

## Preprocessing reversal

Both `plot_2d_model` and `plot_3d_model` automatically reverse preprocessing
transformations (normalization, standardization) for display. The plots always
show values and coordinates in the original data space.
