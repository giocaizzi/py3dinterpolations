# py3Dinterpolations
![PyPI](https://img.shields.io/pypi/v/py3Dinterpolations)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py3Dinterpolations)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/giocaizzi/py3Dinterpolations/tests.yml)
[![codecov](https://codecov.io/gh/giocaizzi/py3Dinterpolations/branch/main/graph/badge.svg?token=8COIITUR2I)](https://codecov.io/gh/giocaizzi/py3Dinterpolations)


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

# Documentation

Documentation can be found [here](https://giocaizzi.github.io/py3Dinterpolations/).