#! /usr/bin/env python
"""Dependencies for RM utilities"""
import pkg_resources

__all__ = [
    "mpfit",
    "normalize",
    "util_FITS",
    "util_misc",
    "util_plotFITS",
    "util_plotTk",
    "util_rec",
    "util_RM",
]

__version__ = pkg_resources.get_distribution("RM-Tools").version
