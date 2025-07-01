#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import setup

NAME = "RM-Tools"
DESCRIPTION = "RM-synthesis, RM-clean and QU-fitting on polarised radio spectra"
URL = "https://github.com/CIRADA-Tools/RM-Tools"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = "1.4.9"
DOWNLOAD_URL = (
    "https://github.com/CIRADA-Tools/RM-Tools/archive/v" + VERSION + ".tar.gz"
)

REQUIRED = [
    "numpy<2",
    "numpy>1.22;python_version=='3.8'",
    "scipy",
    "matplotlib>=3.4.0",
    "astropy",
    "tdqm",
    "deprecation",
    "finufft",
]

# Using AT's fork for now - includes tiny bug fix for bilby
extras_require = {
    "QUfitting": ["bilby>=1.1.5", "emcee", "nestle", "corner"],
    "parallel": ["schwimmbad"],
    "dev": ["pre-commit", "black", "isort", "pytest"],
}

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=REQUIRES_PYTHON,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=["RMtools_1D", "RMtools_3D", "RMutils"],
    entry_points={
        "console_scripts": [
            "rmsynth3d=RMtools_3D.do_RMsynth_3D:main",
            "rmclean3d=RMtools_3D.do_RMclean_3D:main",
            "rmsynth1d=RMtools_1D.do_RMsynth_1D:main",
            "rmclean1d=RMtools_1D.do_RMclean_1D:main",
            "rmsynth1dFITS=RMtools_1D.do_RMsynth_1D_fromFITS:main",
            "qufit=RMtools_1D.do_QUfit_1D_mnest:main",
            "rmtools_freqfile=RMtools_3D.make_freq_file:save_freq_file",
            "rmtools_calcRMSF=RMtools_1D.calculate_RMSF:main",
            "rmtools_testdata1D=RMtools_1D.mk_test_ascii_data:main",
            "rmtools_createchunks=RMtools_3D.create_chunks:main",
            "rmtools_assemblechunks=RMtools_3D.assemble_chunks:main",
            "rmtools_fitIcube=RMtools_3D.do_fitIcube:main",
            "rmtools_peakfitcube=RMtools_3D.RMpeakfit_3D:main",
            "rmtools_testdata3D=RMtools_3D.mk_test_cube_data:main",
            "rmtools_extractregion=RMtools_3D.extract_region:main",
            "rmtools_bwdepol=RMtools_1D.rmtools_bwdepol:main",
            "rmtools_bwpredict=RMtools_1D.rmtools_bwpredict:main",
            "rmtools_3DIrescale=RMtools_3D.rescale_I_model_3D:command_line",
        ],
    },
    install_requires=REQUIRED,
    extras_require=extras_require,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    maintainer="Cameron Van Eck",
    maintainer_email="cameron.vaneck@anu.edu.au",
    test_suite="tests",
)
