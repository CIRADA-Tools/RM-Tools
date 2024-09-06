#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#                                                                             #
# NAME:     rescale_I_model_3D.py                                             #
#                                                                             #
# PURPOSE:  Convert a 3D Stokes I model to a new reference frequency          #
#                                                                             #
# CREATED: 19-Mar-2024 by Cameron Van Eck
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2024 Cameron Van Eck                                          #
#                                                                             #
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


# Steps: 1) Read in covariance matrix, coefficient maps, I ref.freq., lambda^2_0
#       2a) Create fitDict, pixel-wise
#       2b) Invoke rescaling function pixel-wise
#       3) Drizzle output fitDict value into arrays
#       4) Write new coefficient maps

# Optional variations: user-specified reference frequency, unspecified
#                      (make uniform across all pixels)

import argparse
import multiprocessing as mp
import os
from functools import partial

import astropy.io.fits as pf
import numpy as np

from RMutils.util_misc import FitResult, renormalize_StokesI_model


def command_line():
    """Handle invocation from the command line, parsing inputs and running
    everything."""

    # Help string to be shown using the -h option
    descStr = """
    Convert a Stokes I model to a new reference frequency. This changes the
    model coefficients and their errors, but does not change (and does not
    recalculate) the actual model spectrum.
    The conversion of the coefficient uncertainties uses a first-order Taylor
    approximation, so the uncertainties are only valid for relatively small
    variations in reference frequency (depends a lot on the model, but 10%
    seems to be a good rule of thumb).
    The new reference frequency can either be from lambda^2_0 from an FDF cube,
    which will create Stokes I maps that match the coresponding frequency,
    a constant value given by the user, or unspecified (in which case the tool
    will make all pixels have a common reference frequency at the mean input reference frequency).
    The input files are assumed to be the products of do_fitIcube, with the usual filenames.
    Outputs are new coefficient (and error) maps and reference frequency map.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "covar_file",
        metavar="covariance.fits",
        help="FITS cube Stokes I fit covariance matrices.",
    )

    parser.add_argument(
        "-l",
        dest="lambda2_0",
        type=str,
        default=None,
        help="FDF cube from rmsynth3d. If given, will convert model to frequency matching the polarization products.",
    )

    parser.add_argument(
        "-f",
        dest="new_reffreq",
        type=float,
        default=None,
        help="New reference frequency (in Hz). If given, forces all pixels to this frequency. Incompatible with -l",
    )

    parser.add_argument(
        "-o",
        dest="outname",
        type=str,
        default=None,
        help="Output path+filename. If not given, defaults to input path+name (minus the covariance.fits).",
    )
    parser.add_argument(
        "-n",
        dest="num_cores",
        type=int,
        default=1,
        help="Number of cores to use for multiprocessing. Default is 1.",
    )

    parser.add_argument(
        "-w",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing files? [False].",
    )

    args = parser.parse_args()

    if (args.new_reffreq is not None) and (args.lambda2_0 is not None):
        raise Exception(
            "Please do not set both -f and -l flags -- chose one or neither."
        )

    if not os.path.exists(args.covar_file):
        raise Exception(
            f"Cannot find covarance file at {args.covar_file}, please check filename/path."
        )

    if args.covar_file[-15:] != "covariance.fits":
        raise Exception(
            "Input covariance file name doesn't end in covariance.fits; this is required."
        )

    basename = args.covar_file[:-15]

    if args.outname is None:
        args.outname = basename

    # Get data:
    covar_map, old_reffreq_map, coeffs, header = read_data(basename)

    # Create new frequency map:
    if args.lambda2_0 is not None:
        FDF_header = pf.getheader(args.lambda2_0)
        lam0Sq_m2 = FDF_header["LAMSQ0"]
        freq0 = 2.997924538e8 / np.sqrt(lam0Sq_m2)
        new_freq_map = np.ones_like(old_reffreq_map) * freq0
    elif args.new_reffreq is not None:
        new_freq_map = np.ones_like(old_reffreq_map) * args.new_reffreq
    else:
        freq0 = np.nanmean(old_reffreq_map)
        new_freq_map = np.ones_like(old_reffreq_map) * freq0

    # Get fit function from header:
    x = ["Fit model is" in card for card in header["HISTORY"]]
    line = header["HISTORY"][np.where(x)[0][-1]]
    fit_function = [x for x in line.split() if "polynomial" in x][0].split("-")[0]

    new_freq_map, new_coeffs, new_errors = rescale_I_model_3D(
        covar_map,
        old_reffreq_map,
        new_freq_map,
        coeffs,
        fit_function,
        num_cores=args.num_cores,
    )

    write_new_parameters(
        new_freq_map,
        new_coeffs,
        new_errors,
        args.outname,
        header,
        overwrite=args.overwrite,
    )


def read_data(basename):
    """Reads the covariance matrix map, (current) reference frequency map, and
    (current) coefficient+error maps.
    Input:
        basename (str): file path and name up to 'covariance.fits'/'reffreq.fits', etc.
    """

    if not (
        os.path.exists(basename + "coeff0.fits")
        and os.path.exists(basename + "coeff0err.fits")
    ):
        raise Exception("Cannot find coeff0 map. At least coeff 0 map must exist.")

    covar_file = basename + "covariance.fits"
    covar_map = pf.getdata(covar_file)

    freq_file = basename + "reffreq.fits"
    old_reffreq_map, header = pf.getdata(freq_file, header=True)
    # Grabs header from frequency map -- needed for fit function, and to use for
    # writing out products. Better to have 2D map header to avoid fussing with
    # extra axes.

    # Get coefficient maps (without knowing how many there are)
    # Reverse index order to match RM-Tools internal ordering (highest to lowest polynomial order)
    coeffs = np.zeros((6, *old_reffreq_map.shape), dtype=covar_map.dtype)
    for i in range(6):
        try:  # Keep trying higher orders
            data = pf.getdata(basename + f"coeff{i}.fits")
            coeffs[5 - i] = data
        except FileNotFoundError:
            break  # Once it runs out of valid coefficient maps, move on

    return covar_map, old_reffreq_map, coeffs, header


def rescale_I_pixel(data, fit_function):
    covar, coeff, old_freq, new_freq = data
    oldDict = {}  # Initialize a fitDict, which contains the relevant fit information
    oldDict["reference_frequency_Hz"] = old_freq
    oldDict["p"] = coeff
    oldDict["pcov"] = covar
    oldDict["fit_function"] = fit_function

    old_result = FitResult(
        params=coeff,
        fitStatus=np.nan,  # Placeholder
        chiSq=np.nan,  # Placeholder
        chiSqRed=np.nan,  # Placeholder
        AIC=np.nan,  # Placeholder
        polyOrd=len(coeff) - 1,
        nIter=0,  # Placeholder
        reference_frequency_Hz=old_freq,
        dof=np.nan,  # Placeholder
        pcov=covar,
        perror=np.zeros_like(coeff),  # Placeholder
        fit_function=fit_function,
    )

    new_fit_result = renormalize_StokesI_model(old_result, new_freq)
    return new_fit_result.params, new_fit_result.perror


def rescale_I_model_3D(
    covar_map, old_reffreq_map, new_freq_map, coeffs, fit_function="log", num_cores=1
):
    """Rescale the Stokes I model parameters to a new reference frequency, for
    an entire image (i.e., 3D pipeline products).

    Inputs:
        covar_map (4D array): covariance matrix map, such as produced by do_fitIcube.
        old_reffreq_map (2D array): map of current reference frequency, such as produced by do_fitIcube.
        new_freq_map (2D array): map of new reference frequencies.
        coeffs (3D array): model parameter map (going from highest to lowest order)
        coeff_errors (3D array): model parameter uncertainties map (highest to lowest order)

    Returns:
        new_freq_map (unchanged from input)
        new_coeffs (3D array): maps of new model parameters (highest to lowest order)
        new_errors (3D array): maps of new parameter uncertainties (highest to lowest)
    """

    # Initialize output arrays, keep dtype consistent
    new_coeffs = np.zeros_like(coeffs, dtype=coeffs.dtype)
    new_errors = np.zeros_like(coeffs, dtype=coeffs.dtype)
    rs = old_reffreq_map.shape[
        1
    ]  # Get the length of a row, for array indexing later on.

    # Set up inputs for parallelization:
    # Input order is: covariance matrix, coefficient vector, old frequency, new frequency
    inputs = list(
        zip(
            np.reshape(
                np.moveaxis(covar_map, (0, 1), (2, 3)), (old_reffreq_map.size, 6, 6)
            ),
            np.reshape(np.moveaxis(coeffs, 0, 2), (old_reffreq_map.size, 6)),
            old_reffreq_map.flat,
            new_freq_map.flat,
        )
    )
    with mp.Pool(num_cores) as pool_:
        results = pool_.map(
            partial(rescale_I_pixel, fit_function=fit_function), inputs, chunksize=100
        )

    for i, (p, perror) in enumerate(results):
        new_coeffs[:, i // rs, i % rs] = p
        new_errors[:, i // rs, i % rs] = perror

    return new_freq_map, new_coeffs, new_errors


def write_new_parameters(
    new_freq_map, new_coeffs, new_errors, out_basename, header, overwrite=False
):
    """Write out new parameter/uncertainty maps to FITS files.
    Inputs:
        new_freq_map (unchanged from input)
        new_coeffs (3D array): maps of new model parameters (highest to lowest order)
        new_errors (3D array): maps of new parameter uncertainties (highest to lowest)
        out_basename (str): base path+name of the files to be written out
                            (will be postpended with 'newcoeff0.fits', etc.)

    Returns: (nothing)
        Writes out coefficient maps (newcoeff0.fits, etc.) and
        coefficient errors (newcoeff0err.fits, etc.)
    """

    out_header = header.copy()
    out_header["HISTORY"] = "Stokes I model rescaled to new reference frequency."
    out_header["REFFREQ"] = (new_freq_map[0, 0], "Hz")
    if "BUNIT" in out_header:
        del out_header["BUNIT"]

    # Work out highest order of polynomial:
    # if any of the 6 possible coeff planes contain non-zero and non-nan values, it's a 'good' plane.
    max_order = (
        np.sum(np.any((new_coeffs != 0.0) & (~np.isnan(new_coeffs)), axis=(1, 2))) - 1
    )

    for i in range(max_order + 1):
        pf.writeto(
            out_basename + f"newcoeff{i}.fits",
            new_coeffs[5 - i],
            header=out_header,
            overwrite=overwrite,
        )
        pf.writeto(
            out_basename + f"newcoeff{i}err.fits",
            new_errors[5 - i],
            header=out_header,
            overwrite=overwrite,
        )


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    command_line()
