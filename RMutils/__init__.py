#! /usr/bin/env python
"""Dependencies for RM utilities"""

from importlib.metadata import version

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


__version__ = version("RM-Tools")
