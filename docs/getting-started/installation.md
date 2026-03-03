# Installation

## From PyPI

```bash
pip install py3dinterpolations
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add py3dinterpolations
```

## Dependencies

py3dinterpolations requires Python 3.11+ and installs the following dependencies automatically:

| Package | Purpose |
|---------|---------|
| pandas | Data handling |
| numpy | Numerical computation |
| matplotlib | 2D plotting |
| plotly | 3D interactive visualization |
| pykrige | Kriging models |
| scikit-learn | Cross-validation, IDW |
| shapely | Convex hull for irregular grids |

## Development setup

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/giocaizzi/py3dinterpolations.git
cd py3dinterpolations
uv sync --extra dev
```

To build docs locally:

```bash
uv sync --extra docs
uv run mkdocs serve
```
