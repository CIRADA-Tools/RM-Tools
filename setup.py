#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = 'RM-tools'
DESCRIPTION = 'RM-synthesis, RM-clean and QU-fitting on polarised radio spectra'
URL = 'https://github.com/CIRADA-Tools/RM'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = '0.1.0'

REQUIRED = [
    'numpy', 'scipy', 'matplotlib', 'astropy',
]

extras_require={'QUfitting': ['pymultinest'],'parallel':["schwimmbad"]}

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['RMtools_1D', 'RMtools_3D', 'RMutils'],
    entry_points={
        'console_scripts': ['rmsynth3d=RMtools_3D.do_RMsynth_3D:main',
                            'rmclean3d=RMtools_3D.do_RMclean_3D:main',
                            'rmsynth1d=RMtools_1D.do_RMsynth_1D:main',
                            'rmclean1d=RMtools_1D.do_RMclean_1D:main',
                            'rmsynth1dFITS=RMtools_1D.do_RMsynth_1D_fromFITS:main'],
    },
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    test_suite='tests',
)
