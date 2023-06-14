# py3Dinterpolations

|  |  |
| --- | --- |
| Distribution | ![PyPI](https://img.shields.io/pypi/v/py3Dinterpolations?color=blue) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py3Dinterpolations) |
| Testing | [![Tests](https://github.com/giocaizzi/py3Dinterpolations/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/giocaizzi/py3Dinterpolations/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/giocaizzi/py3Dinterpolations/branch/main/graph/badge.svg?token=8COIITUR2I)](https://codecov.io/gh/giocaizzi/py3Dinterpolations) |
| Documentation | [![Documentation](https://github.com/giocaizzi/py3Dinterpolations/actions/workflows/documentation.yml/badge.svg?branch=gh-pages)](https://github.com/giocaizzi/py3Dinterpolations/actions/workflows/documentation.yml) |


This is a python package to compute **quick 3D interpolations of spatial data**.

Supports the **following interpolation** methods:
- *Ordinary 3D Kriging* : [`pykrige`](https://github.com/GeoStat-Framework/PyKrige)

Supports **preprocessing** of data:
- *Downsampling*
- *Normalization* of X,Y,Z coordinates
- *Standardization* of signal 

**Visualizations**
- 2D with [`matplotlib`](https://matplotlib.org/stable/)
- 3D with [`plotly`](https://plotly.com/).

## Installation

```
pip install py3Dinterpolations
```

## Documentation

Documentation with working examples can be found [here](https://giocaizzi.github.io/py3Dinterpolations/).
