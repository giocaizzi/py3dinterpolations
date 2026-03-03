"""Plotting module for 3D interpolation visualization."""

from .downsampling import plot_downsampling
from .plot_2d import plot_2d_model
from .plot_3d import plot_3d_model

__all__ = ["plot_2d_model", "plot_3d_model", "plot_downsampling"]
