===============================================
Welcome to py3Dinterpolations's documentation!
===============================================

This is a python package to compute quick 3D interpolations of spatial data.

Supports the **following interpolation** methods:
- *Ordinary 3D Kriging* : `pykrige <https://github.com/GeoStat-Framework/PyKrige>`_

Supports **preprocessing** of data:
- *Downsampling*
- *Normalization* of X,Y,Z coordinates
- *Standardization* of signal 

**Visualizations**
- 2D with `matplotlib <https://matplotlib.org/stable/>`_
- 3D with `plotly <https://plotly.com/)>`_

The :obj:`py3Dinterpolations` features:

Different interpolation methods:

- Kriging (wrapping :obj:`pykrige`)
- Inverse distance weighting 

Preprocessing tools:
- *Downsampling* of data
- *Normalization* of coordinates
- *Standardization* of data values


.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/quickstart.ipynb
   examples/preprocessing.ipynb

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

   py3Dinterpolations



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
