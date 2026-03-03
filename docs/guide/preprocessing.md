# Preprocessing

py3dinterpolations includes a [`Preprocessor`][py3dinterpolations.modelling.preprocessor.Preprocessor]
that transforms data before interpolation. Three operations are available:

1. **Downsampling** — reduce data density by averaging blocks
2. **XYZ normalization** — scale coordinates to [0, 1]
3. **V standardization** — center values to mean=0, std=1

All preprocessing is reversible (except downsampling, which is lossy) — the
[`Modeler`][py3dinterpolations.modelling.modeler.Modeler] automatically reverses
normalization and standardization when making predictions.

## Using preprocessing with `interpolate`

Pass a `preprocessing` dict to the [`interpolate`][py3dinterpolations.interpolate] function:

```python
from py3dinterpolations import GridData, interpolate

griddata = GridData(df)
modeler = interpolate(
    griddata=griddata,
    model_type="ordinary_kriging",
    grid_resolution=5.0,
    model_params={"variogram_model": "linear", "nlags": 6, "weight": True},
    preprocessing={
        "downsampling_res": 5.0,
        "normalize_xyz": True,
        "standardize_v": True,
    },
)
```

## Downsampling

Downsampling groups depth profiles into blocks of a given resolution and applies
a statistic (default: mean) to each block. This reduces data density while
preserving general trends.

```python
from py3dinterpolations.modelling.preprocessor import Preprocessor

preprocessor = Preprocessor(
    griddata,
    downsampling_res=5.0,
    downsampling_method="mean",  # or "max", "min", "median", "sum", "quantile75"
)
preprocessed = preprocessor.preprocess()

print(len(griddata))       # 283
print(len(preprocessed))   # 127 (reduced)
```

You can also pass a custom callable for `downsampling_method`.

### Visualizing downsampling

Use [`plot_downsampling`][py3dinterpolations.plotting.downsampling.plot_downsampling]
to compare original and downsampled data per borehole:

```python
from py3dinterpolations import plot_downsampling

fig = plot_downsampling(griddata, preprocessed)
```

## XYZ normalization

Min-max normalization scales each coordinate to [0, 1]:

$$
X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

This prevents coordinates with larger ranges from dominating the interpolation,
especially when X, Y, Z have different scales or units.

## V standardization

Z-score standardization centers values around 0 with unit variance:

$$
V_{\text{std}} = \frac{V - \mu_V}{\sigma_V}
$$

This ensures the interpolation algorithm is not influenced by the absolute
magnitude of the values. The `Modeler` automatically reverses this transformation
in the output.

## Preprocessing parameters

After preprocessing, the resulting `GridData` carries a
[`PreprocessingParams`][py3dinterpolations.core.types.PreprocessingParams]
object with the transformation parameters:

```python
preprocessed.preprocessing_params
# PreprocessingParams(
#     downsampling=DownsamplingParams(resolution=5.0),
#     normalization={X: NormalizationParams(...), Y: ..., Z: ...},
#     standardization=StandardizationParams(mean=..., std=...),
# )
```

These parameters are used by the `Modeler` to reverse transformations on predictions.
