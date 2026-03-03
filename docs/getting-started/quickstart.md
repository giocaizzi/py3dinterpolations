# Quick Start

This guide walks through the core workflow: loading data, interpolating, and visualizing results.

## Load data

py3dinterpolations works with tabular data containing spatial coordinates and a value column.
The expected columns are **ID**, **X**, **Y**, **Z**, **V** (customizable via keyword arguments).

```python
import pandas as pd
from py3dinterpolations import GridData

df = pd.read_csv("measurements.csv")
griddata = GridData(df)

print(griddata)
# GridData(points=283, ids=31)
```

The `GridData` object standardizes the DataFrame into a canonical format with a
`MultiIndex(ID, X, Y, Z)` and a single column `V`.

!!! tip "Custom column names"
    If your columns differ from the defaults, pass them explicitly:

    ```python
    griddata = GridData(df, ID="site", X="easting", Y="northing", Z="depth", V="concentration")
    ```

## Interpolate

The [`interpolate`][py3dinterpolations.interpolate] function is the main entry point.
It builds the 3D grid, runs optional preprocessing, fits the model, and returns a
[`Modeler`][py3dinterpolations.modelling.modeler.Modeler] with results attached.

```python
from py3dinterpolations import interpolate

modeler = interpolate(
    griddata=griddata,
    model_type="ordinary_kriging",
    grid_resolution=5.0,
    model_params={
        "variogram_model": "spherical",
        "nlags": 15,
        "weight": True,
    },
    preprocessing={
        "normalize_xyz": True,
        "standardize_v": True,
    },
)
```

### Access results

```python
result = modeler.result       # InterpolationResult
grid = result.interpolated    # np.ndarray — the interpolated 3D volume
variance = result.variance    # np.ndarray or None — kriging variance
```

## Visualize

### 2D slices with matplotlib

```python
from py3dinterpolations import plot_2d_model

fig = plot_2d_model(
    modeler,
    axis="Z",
    plot_points=True,
    annotate_points=True,
)
fig.savefig("slices.png")
```

### 3D interactive volume with plotly

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

## Next steps

- [Preprocessing](../guide/preprocessing.md) — downsampling, normalization, standardization
- [Models](../guide/models.md) — Kriging vs IDW, parameters, cross-validation
- [Visualization](../guide/visualization.md) — 2D and 3D plotting options
