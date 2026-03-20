# Senior Code Review: py3dinterpolations

**Reviewer**: Claude (Senior Software Engineer)
**Date**: 2026-03-18
**Scope**: Full codebase review — architecture, correctness, performance, testing, API design

---

## Executive Summary

`py3dinterpolations` is a well-structured 3D spatial interpolation library built on top of pykrige and scikit-learn. The codebase demonstrates strong software engineering fundamentals: clean layered architecture, comprehensive type annotations, proper use of protocols and abstract classes, and a well-designed preprocessing pipeline. The code is production-quality for a v1.0.0 release.

That said, there are **15 actionable findings** across correctness, robustness, performance, and API design that would elevate this codebase to a higher level of maturity.

**Overall Rating**: **B+** — Solid foundation with targeted improvements needed.

---

## 1. CRITICAL — Bugs & Correctness Issues

### 1.1 Assertions used as runtime validation (will vanish with `python -O`)

**Files**: `modelling/interpolate.py:68`, `modelling/models/idw.py:45-46`, `modelling/preprocessor.py:122`, `plotting/plot_2d.py:33`, `plotting/plot_3d.py:35`

```python
# interpolate.py:68
assert model_params_grid is not None  # Stripped by -O flag

# idw.py:45-46
assert self._points is not None
assert self._values is not None
```

**Problem**: `assert` statements are stripped when Python runs with the `-O` (optimize) flag. These are being used for runtime validation, not development-time invariant checks. In production, this could lead to `NoneType` attribute errors instead of clean error messages.

**Fix**: Replace with explicit `if ... raise RuntimeError(...)` checks that match the pattern already used elsewhere in the codebase (e.g., `kriging.py:42-43`).

---

### 1.2 `_apply_downsampling` may return inconsistent types

**File**: `modelling/preprocessor.py:146-168`

The function signature says `pd.DataFrame | pd.Series[float]`, but the `QUANTILE75` branch calls `.quantile(0.75)` on a DataFrame group, which returns a `Series` (or scalar for single columns). The other branches (`.mean()`, `.max()`, etc.) also return Series when called on `grouped_df[["V"]]`.

However, this function is called via `grouped[["V"]].apply(...)`. The behavior of `.apply()` depends on what the function returns — if some branches return DataFrame and others Series, pandas may produce inconsistent results across versions.

**Fix**: Ensure all branches return the same type. Since all branches operate on `grouped_df[["V"]]`, they'll all return Series. Make the type annotation `pd.Series[float]` and verify behavior.

---

### 1.3 `plot_3d_model` uses wrong einsum axes

**File**: `plotting/plot_3d.py:37`

```python
values = np.einsum("ZXY->XYZ", modeler.result.interpolated)
```

The interpolation result shape is documented as `(Z, Y, X)` (pykrige convention), but the einsum treats the input as `(Z, X, Y)`. The axis labels in einsum are just variable names, so this technically works (it transposes 0,1,2 → 1,2,0), but the labeling is misleading and suggests the developer may have confused the actual axis ordering.

Compare with `idw.py:106` which correctly documents the `(Z, Y, X)` output:
```python
interpolated = np.einsum("xyz->zyx", interpolated)
```

**Recommendation**: Verify the actual axis ordering and fix the einsum label to `"ZYX->XYZ"` if the intent is to reverse the pykrige convention.

---

### 1.4 Grid resolution accepts zero or negative values

**File**: `core/types.py:100-112`, `core/grid3d.py:32`

```python
# GridAxis.grid property
return np.arange(self.min, self.max, self.res)
```

If `res <= 0`, `np.arange` returns an empty array (if `res == 0`, it infinite loops). No validation exists in `GridResolution.from_input()`, `GridAxis`, or `create_grid()`.

**Fix**: Add validation in `GridResolution.__post_init__` or `from_input()`:
```python
if any(r <= 0 for r in (self.x, self.y, self.z)):
    raise ValueError("Resolution must be positive")
```

---

## 2. HIGH — Robustness Issues

### 2.1 `GridDataSpecs.from_dataframe` doesn't handle empty DataFrames

**File**: `core/griddata.py:94-107`

If an empty DataFrame is passed, `df["X"].min()` returns `nan`, and `float(nan)` produces `nan`. Downstream, `np.arange(nan, nan, res)` returns an empty array, and the pipeline silently produces empty results.

**Fix**: Add an early guard:
```python
if df.empty:
    raise ValueError("Cannot compute specs from an empty DataFrame")
```

---

### 2.2 `normalize()` and `standardize()` don't handle NaN values

**File**: `modelling/utils.py:8-42`

Real-world spatial data commonly contains NaN values. Both functions compute `min()`, `max()`, `mean()`, `std()` without `skipna` consideration (pandas defaults to `skipna=True`, so the params will be correct, but the normalized output will propagate NaNs silently).

**Recommendation**: Document the NaN handling behavior, or add an explicit `dropna()` before computing params if NaN propagation is not desired.

---

### 2.3 Convex hull filtering uses Python loop over shapely — very slow for large grids

**File**: `core/grid3d.py:228-232`

```python
mask = np.array([self._hull.contains(Point(p[0], p[1])) for p in points])
```

This creates a Python `Point` object and calls `contains()` per grid point. For a 100×100×50 grid (500K points), this takes minutes.

**Fix**: Use shapely's vectorized `contains` via `shapely.vectorized.contains` or `shapely.contains_xy` (shapely ≥2.0):
```python
from shapely import contains_xy
mask = contains_xy(self._hull, points[:, 0], points[:, 1])
```

This is 100-1000x faster.

---

### 2.4 `Modeler` fits model in `__init__` — violates separation of construction and behavior

**File**: `modelling/modeler.py:37-40`

The constructor immediately calls `self._model.fit(...)`, making it impossible to:
- Inspect the Modeler before fitting
- Retry fitting with different data
- Serialize/deserialize a Modeler without triggering a fit

This is a design smell. Construction should set up state; methods should trigger behavior.

**Recommendation**: Move the `fit()` call to `interpolate()` or add an explicit `.fit()` method. For backward compatibility, you could keep the current behavior but document it clearly.

---

## 3. MEDIUM — Performance Issues

### 3.1 `grid`, `normalized_grid`, `mesh`, `normalized_mesh` properties recompute on every access

**File**: `core/grid3d.py:91-136`

Every call to `.mesh` creates a new `np.meshgrid`. Every call to `.grid` creates new `np.arange` arrays. These are called multiple times during prediction and plotting.

**Fix**: Cache with `functools.cached_property` or compute once in `__init__`:
```python
from functools import cached_property

@cached_property
def mesh(self) -> dict[str, np.ndarray]:
    ...
```

Since `GridAxis` is frozen, these values never change — caching is safe.

---

### 3.2 Excessive DataFrame copies in `Preprocessor.preprocess()`

**File**: `modelling/preprocessor.py:71-93`

The preprocessing pipeline creates up to 4 copies of the DataFrame:
1. `self.griddata.data.copy()` (line 71)
2. `data.copy()` in `_normalize_xyz` (line 99)
3. `data.copy()` in `_standardize_v` (line 110)
4. Internal copies during downsampling

For large datasets, this is significant memory overhead.

**Fix**: Operate on a single copy. The initial `.copy()` at line 71 is sufficient — remove the copies in `_normalize_xyz` and `_standardize_v` since they receive the already-copied DataFrame.

---

### 3.3 `plot_2d_model` copies training data inside the loop

**File**: `plotting/plot_2d.py:104`

```python
for ax, i in zip(axes, range(len(axis_data)), strict=False):
    ...
    points_df = gd_reversed.data.copy().reset_index()  # Full copy per slice!
```

The full DataFrame is copied on every iteration of the loop.

**Fix**: Move the copy outside the loop.

---

## 4. LOW — API Design & Code Quality

### 4.1 `SklearnModel` not in `MODEL_REGISTRY` — inconsistent discovery

**File**: `modelling/models/__init__.py:9-12`

`SklearnModel` is exported in `__all__` but not in `MODEL_REGISTRY`. Users must know to instantiate it directly rather than using `get_model()`. This is a "hidden" API.

**Recommendation**: Either add a `ModelType.SKLEARN` variant and register it, or document that `SklearnModel` is intended for direct instantiation only.

---

### 4.2 Hardcoded column names in plotting functions

**Files**: `plotting/plot_2d.py:75-80`, `plotting/plot_3d.py:37-68`

Plotting functions hardcode `"X"`, `"Y"`, `"Z"`, `"V"` column names. While `GridData._set_data()` renames columns to canonical names internally, this creates a hidden coupling. If the internal representation ever changes, all plotting code breaks.

**Recommendation**: Use constants from a central location or access through `GridData` properties.

---

### 4.3 `plot_downsampling` hides empty-subplot logic inside the per-ID loop

**File**: `plotting/downsampling.py:58-62`

```python
for idx, id_to_plot in enumerate(unique_ids):
    ...
    if len(unique_ids) < num_rows * num_cols:   # This is loop-invariant!
        for i in range(len(unique_ids), num_rows * num_cols):
            ...
```

The visibility toggling of empty subplots runs on every iteration but only needs to run once after the loop.

**Fix**: Move the empty-subplot hiding block after the loop.

---

### 4.4 `plot_downsampling` crashes if only 1 unique ID (scalar axes)

**File**: `plotting/downsampling.py:37-43`

```python
fig, axes = plt.subplots(num_rows, num_cols, ...)
...
ax = axes[row, col]  # Fails if num_rows=1 and num_cols=1 (axes is a single Axes)
```

When there's only 1 ID, `plt.subplots(1, 4)` returns a 1D array, and `axes[row, col]` fails with an IndexError. With `(1, 1)`, `axes` is a single `Axes` object, not an array.

**Fix**: Use `fig, axes = plt.subplots(..., squeeze=False)` to always get a 2D array.

---

### 4.5 `interpolate()` requires exactly one of `model_params` or `model_params_grid`

**File**: `modelling/interpolate.py:46-51`

This means users cannot use IDW or sklearn models with default parameters. They must always pass `model_params={}` even when no custom params are needed.

**Recommendation**: Allow `model_params` to default to `{}` when neither argument is provided, or make both optional with a sensible default.

---

## 5. TESTING

### Strengths
- 71 test cases covering all major modules
- Good use of parametrization (8-way preprocessing combinations)
- Strategic mocking of expensive operations (GridSearchCV, pykrige)
- Proper error-case testing with `pytest.raises`

### Gaps
| Area | Missing Coverage |
|------|-----------------|
| Edge cases | Empty DataFrames, NaN values, duplicate coordinates, single-point datasets |
| Numerical | Precision of normalization round-trip, IDW with equidistant points |
| Grid | Zero/negative resolution, very large resolutions, min==max bounds |
| Models | Kriging variance numerical accuracy, IDW power=0 behavior |
| Plotting | Single-ID downsampling plot, empty result arrays |
| Integration | Full pipeline from CSV to plot with all preprocessing combinations |
| Performance | No benchmarks or regression tests for vectorized IDW |

---

## 6. ARCHITECTURE — Positive Observations

These are things done **well** that should be preserved:

1. **Clean layered architecture**: `core` → `modelling` → `plotting` with proper dependency direction
2. **Type system**: Protocols for sklearn compatibility, StrEnum for finite sets, frozen dataclasses for immutable params
3. **Preprocessing pipeline**: Chainable operations with parameter tracking and reversal — this is well-designed
4. **Model abstraction**: `BaseModel` ABC with consistent `fit/predict` interface
5. **Factory pattern**: `create_grid()` and `get_model()` encapsulate construction logic cleanly
6. **IDW vectorization**: The batched numpy broadcasting approach is well-implemented and memory-safe
7. **Variance handling**: Correct scaling of variance during standardization reversal (`variance * std²`)
8. **CI/CD**: Multi-version testing (3.11-3.13), ruff + mypy strict, Codecov integration, automated version checks

---

## Summary of Findings

| # | Severity | Issue | File(s) |
|---|----------|-------|---------|
| 1.1 | **CRITICAL** | `assert` used for runtime validation | interpolate.py, idw.py, preprocessor.py, plot_2d.py, plot_3d.py |
| 1.2 | **CRITICAL** | Inconsistent return type in downsampling | preprocessor.py |
| 1.3 | **CRITICAL** | Misleading einsum axis labels in 3D plot | plot_3d.py |
| 1.4 | **CRITICAL** | No validation for zero/negative grid resolution | types.py, grid3d.py |
| 2.1 | HIGH | Empty DataFrame not handled | griddata.py |
| 2.2 | HIGH | NaN handling undocumented | utils.py |
| 2.3 | HIGH | O(N) Python loop for hull filtering | grid3d.py |
| 2.4 | HIGH | Model fit in constructor | modeler.py |
| 3.1 | MEDIUM | Grid properties recomputed on every access | grid3d.py |
| 3.2 | MEDIUM | Excessive DataFrame copies in preprocessing | preprocessor.py |
| 3.3 | MEDIUM | DataFrame copy inside plot loop | plot_2d.py |
| 4.1 | LOW | SklearnModel not in registry | models/__init__.py |
| 4.2 | LOW | Hardcoded column names in plotting | plot_2d.py, plot_3d.py |
| 4.3 | LOW | Loop-invariant code inside loop | downsampling.py |
| 4.4 | LOW | Single-ID crash in downsampling plot | downsampling.py |
| 4.5 | LOW | Must pass empty dict for default model params | interpolate.py |

---

## Recommended Priority

1. **Immediate** (before next release): Fix 1.1, 1.4, 2.1, 4.4
2. **Next sprint**: Fix 1.2, 1.3, 2.3, 3.1, 3.3, 4.5
3. **Backlog**: Fix 2.2, 2.4, 3.2, 4.1, 4.2, 4.3
