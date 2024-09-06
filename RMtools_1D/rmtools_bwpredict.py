#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================#
#                                                                             #
# NAME:     rmtools_bwpredict.py                                              #
#                                                                             #
# PURPOSE: Algorithm for finding polarized sources while accounting for       #
#          bandwidth depolarization.                                          #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2020 Canadian Initiative for Radio Astronomy Data Analysis    #                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
# =============================================================================#

import argparse
import math as m
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c as speed_of_light

from RMtools_1D.rmtools_bwdepol import (
    adjoint_theory,
    estimate_channel_bandwidth,
    plot_adjoint_info,
)

if sys.version_info.major == 2:
    print("RM-tools will no longer run with Python 2! Please use Python 3.")
    exit()


# -----------------------------------------------------------------------------#
def bwdepol_compute_predictions(
    freqArr_Hz, widths_Hz=None, phiMax_radm2=None, dPhi_radm2=None
):
    """Computes theoretical sensitivity and noise curves for given
    channelization.

    Parameters
    ----------
    freqArr_Hz:  array, float
                 array of the centers of the frequency channels

    Kwargs
    ------
    phiMax_radm2: float
                  Maximum absolute Faraday depth (rad/m^2)

    dPhi_radm2: float
                Faraday depth channel size (rad/m^2)

    nSamples: float
              Number of samples across the RMSF

    Returns
    -------
    adjoint_info: list
                  Faraday depth array, sensitivity array, noise array

    """
    # Calculate some wavelength parameters
    lambdaSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)
    # dFreq_Hz = np.nanmin(np.abs(np.diff(freqArr_Hz)))
    lambdaSqRange_m2 = np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2)
    dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
    dLambdaSqMax_m2 = np.nanmax(np.abs(np.diff(lambdaSqArr_m2)))

    # Set the Faraday depth range
    fwhmRMSF_radm2 = 2.0 * m.sqrt(3.0) / lambdaSqRange_m2
    if dPhi_radm2 is None:
        dPhi_radm2 = fwhmRMSF_radm2
    if phiMax_radm2 is None:
        phiMax_radm2 = 2 * m.sqrt(3.0) / dLambdaSqMin_m2

    # Faraday depth sampling.
    phiArr_radm2 = np.arange(0, phiMax_radm2 + 1e-6, dPhi_radm2)
    phiArr_radm2 = phiArr_radm2.astype("float64")

    print(
        "Computing out to a Faraday depth of {:g} rad/m^2 in steps of {:g} rad/m^2".format(
            phiMax_radm2, dPhi_radm2
        )
    )

    # Uniform weights only for prediction purposes
    weightArr = np.ones(freqArr_Hz.shape, dtype="float64")

    # Get channel widths if not given by user.
    K = 1.0 / np.sum(weightArr)
    if widths_Hz is None:
        widths_Hz = estimate_channel_bandwidth(freqArr_Hz)

    adjoint_varbs = [widths_Hz, freqArr_Hz, phiArr_radm2, K, weightArr]
    adjoint_info = adjoint_theory(adjoint_varbs, weightArr, show_progress=False)
    phiArr_radm2, adjoint_sens, adjoint_noise = adjoint_info

    adjoint_info[2] = adjoint_noise / np.max(adjoint_noise)  # Renormalize to unity.

    return adjoint_info


def main():
    """
    Start the function to generate the figures if called from the command line.
    """
    # Help string to be shown using the -h option
    descStr = """
    Calculate the theoretical sensitivity and noise curves for the bandwidth-
    depolarization-corrected RM synthesis method described in Fine et al. (2022).

    Takes in a ASCII file containing either 1 column (channel frequencies, in Hz)
    or two columns (channel frequencies and channel bandwidths in Hz, space separated).

    Generates interactive plots of the two curves. These are intended to guide
    users in deciding in what RM range traditional RM synthesis is sufficiently
    accurate, and over what RM range they may want to use the modified method.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "dataFile",
        metavar="dataFile.dat",
        nargs=1,
        help="ASCII file containing channel frequencies.",
    )
    parser.add_argument(
        "-l",
        dest="phiMax_radm2",
        type=float,
        default=None,
        help="absolute max Faraday depth sampled [Auto].",
    )
    parser.add_argument(
        "-d",
        dest="dPhi_radm2",
        type=float,
        default=None,
        help="width of Faraday depth channel [Auto].\n(overrides -s NSAMPLES flag)",
    )
    parser.add_argument(
        "-s",
        dest="nSamples",
        type=float,
        default=10,
        help="number of samples across the RMSF lobe [10].",
    )
    parser.add_argument(
        "-f",
        dest="plotfile",
        default=None,
        help="Filename to save plot to. [do not save]",
    )

    args = parser.parse_args()

    # Get data:
    try:
        data = np.loadtxt(args.dataFile[0], unpack=True, dtype="float64")
        if data.ndim == 1:  # If single column file, data is only channel freqs
            freqArr_Hz = data
            widths_Hz = None
        else:  # file has multiple columns
            freqArr_Hz = data[0]  # assume the first column is channel freqs
            widths_Hz = data[1]  # Assume widths are 2nd column if present.
    except:
        print(
            "Unable to read file. Please ensure file is readable and contains 1 or 2 columns."
        )
        exit

    adjoint_info = bwdepol_compute_predictions(
        freqArr_Hz=freqArr_Hz,
        widths_Hz=widths_Hz,
        phiMax_radm2=args.phiMax_radm2,
        dPhi_radm2=args.dPhi_radm2,
    )

    # plot adjoint info
    plot_adjoint_info(adjoint_info, units="arb. units")
    if args.plotfile is not None:
        plt.savefig(args.plotfile, bbox_inches="tight")
    else:
        plt.show()


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
