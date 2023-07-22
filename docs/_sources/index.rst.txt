===============================================
Welcome to py3dinterpolations's documentation!
===============================================

This is a python package to compute quick 3D interpolations of spatial data.

Supports **deterministic and interpolation** methods:

- *Ordinary 3D Kriging* : `pykrige <https://github.com/GeoStat-Framework/PyKrige>`_
- *Inverse distance weighting (IDW)*

Features **parameters estimation**:

- GridSearchCV for Kriging : execute a exahustive search over specified parameter values for an estimator.
  See `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_

Supports **preprocessing** of data:

- *Downsampling* - reduce the number of points by using statistical methods by blocks
- *Normalization* of X,Y,Z coordinates reducing effect of magnitude of coordinates. 
- *Standardization* of signal - standard distribution of signal, reducing effect of magnitude of signal.

**Visualizations** in 2D and 3D:

- 2D with `matplotlib <https://matplotlib.org/stable/>`_
- 3D with `plotly <https://plotly.com/)>`_


.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/quickstart.ipynb
   examples/preprocessing.ipynb
   examples/estimator.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Models

   models/kriging.ipynb
   models/idw.ipynb



Code reference
==============

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :caption: Code reference
   :recursive:

   py3dinterpolations



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
