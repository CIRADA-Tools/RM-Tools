#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     do_RMsynth_1D_fromFITS.py                                         #
#                                                                             #
# PURPOSE:  Run RM-synthesis on an ASCII Stokes I, Q & U spectrum.            #
#                                                                             #
# MODIFIED: Summer 2019, by Boris Gbeasor                                     #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 - 2018 Cormac R. Purcell                                 #
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

# import time
import argparse
import os
import sys

# import pdb

if sys.version_info.major == 2:
    print("RM-tools will no longer run with Python 2! Please use Python 3.")
    exit()


import numpy as np
from astropy import wcs
from astropy.io import fits

from RMtools_1D.do_RMsynth_1D import run_rmsynth, saveOutput
from RMtools_3D.make_freq_file import get_freq_array


# -----------------------------------------------------------------------------#
def main():
    """
    Start the function to perform RM-synthesis if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    *** PROTOTYPE! FUNCTIONALITY NOT GUARANTEED! PLEASE TEST AND SUBMIT BUG REPORTS!***

    Run RM-synthesis on Stokes I, Q and U spectra (1D) stored in a FITS
    file. Does not currently account for errors.
    If these features are needed, please use the standard 1D function.
    """

    epilog_text = """
    Outputs with -S flag:
    _FDFdirty.dat: Dirty FDF/RM Spectrum [Phi, Q, U]
    _RMSF.dat: Computed RMSF [Phi, Q, U]
    _RMsynth.dat: list of derived parameters for RM spectrum
                (approximately equivalent to -v flag output)
    _RMsynth.json: dictionary of derived parameters for RM spectrum
    _weight.dat: Calculated channel weights [freq_Hz, weight]
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr,
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "dataFile",
        metavar="StokesQ.fits",
        nargs=1,
        help="FITS cube with Stokes Q data.",
    )
    parser.add_argument(
        "UFile", metavar="StokesU.fits", nargs=1, help="FITS cube with Stokes U data."
    )
    parser.add_argument(
        "xcoords",
        metavar="xcoords",
        nargs="?",
        type=float,
        default="1",
        help="X pixel location (FITS 1-indexed convention)",
    )
    parser.add_argument(
        "ycoords",
        metavar="ycoords",
        nargs="?",
        type=float,
        default="1",
        help="Y pixel location (FITS 1-indexed convention)",
    )
    parser.add_argument(
        "-c",
        dest="sky_coords",
        action="store_true",
        help="transform from sky coordinates (assumes X and Y are sky coordinates in degrees).",
    )
    parser.add_argument(
        "-I",
        dest="StokesI_fits",
        default=None,
        help="extract Stokes I from given file. [None]",
    )
    parser.add_argument(
        "-t",
        dest="fitRMSF",
        action="store_true",
        help="fit a Gaussian to the RMSF [False]",
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
        "-w",
        dest="weightType",
        default="variance",
        help="weighting [inverse variance] or 'uniform' (all 1s).",
    )
    parser.add_argument(
        "-o",
        dest="polyOrd",
        type=int,
        default=2,
        help="polynomial order to fit to I spectrum [2].",
    )
    parser.add_argument(
        "-i",
        dest="noStokesI",
        action="store_true",
        help="ignore the Stokes I spectrum [False].",
    )
    parser.add_argument(
        "-b",
        dest="bit64",
        action="store_true",
        help="use 64-bit floating point precision [False (uses 32-bit)]",
    )
    parser.add_argument(
        "-p", dest="showPlots", action="store_true", help="show the plots [False]."
    )
    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="verbose output [False]."
    )
    parser.add_argument(
        "-S", dest="saveOutput", action="store_true", help="save the arrays [False]."
    )
    parser.add_argument(
        "-D",
        dest="debug",
        action="store_true",
        help="turn on debugging messages & plots [False].",
    )
    parser.add_argument(
        "-U",
        dest="units",
        type=str,
        default=None,
        help="Intensity units of the data. [from FITS header]",
    )
    parser.add_argument(
        "-r",
        "--super-resolution",
        action="store_true",
        help="Optimise the resolution of the RMSF (as per Rudnick & Cotton). ",
    )
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.dataFile[0]):
        print("File does not exist: '%s'." % args.dataFile[0])
        sys.exit()
    prefixOut, ext = os.path.splitext(args.dataFile[0])
    dataDir, dummy = os.path.split(args.dataFile[0])
    # Set the floating point precision
    nBits = 32
    if args.bit64:
        nBits = 64
    verbose = args.verbose
    # data = clRM.readFile(args.dataFile[0],nBits, verbose)

    hduList = fits.open(args.dataFile[0])
    if args.sky_coords:
        imgwcs = wcs.WCS(hduList[0].header, naxis=(1, 2))
        xcoords, ycoords = np.round(
            imgwcs.all_world2pix(args.xcoords, args.ycoords, 0)
        ).astype(int)
        print("Extracting pixel {} x {} (1-indexed)".format(xcoords + 1, ycoords + 1))
    #        print(ycoords+1)
    else:
        xcoords = int(args.xcoords) - 1
        ycoords = int(args.ycoords) - 1
    freq_array = get_freq_array(args.dataFile[0])
    Q_array = get_data_Q_U(args.dataFile[0], ycoords, xcoords)
    U_array = get_data_Q_U(args.UFile[0], ycoords, xcoords)
    dQ_array = np.full(freq_array.shape, 1 * 10 ** (-3))
    dU_array = np.full(freq_array.shape, 1 * 10 ** (-3))

    Q_array[~np.isfinite(Q_array)] = np.nan
    U_array[~np.isfinite(U_array)] = np.nan
    data = [freq_array, Q_array, U_array, dQ_array, dU_array]

    if args.units is None:
        if "BUNIT" in hduList[0].header:
            args.units = hduList[0].header["BUNIT"]
        else:
            args.units = "Jy/beam"

    if (Q_array != 0).sum() == 0 and (Q_array != 0).sum() == 0:
        raise Exception("All QU values zero! Maybe invalid pixel?")

    if args.StokesI_fits is not None:
        I_array = get_data_Q_U(args.StokesI_fits, ycoords, xcoords)
        dI_array = np.full(freq_array.shape, 1 * 10 ** (-3))
        data = [freq_array, I_array, Q_array, U_array, dI_array, dQ_array, dU_array]
    # Run RM-synthesis on the spectra
    mDict, aDict = run_rmsynth(
        data=data,
        polyOrd=args.polyOrd,
        phiMax_radm2=args.phiMax_radm2,
        dPhi_radm2=args.dPhi_radm2,
        nSamples=args.nSamples,
        weightType=args.weightType,
        fitRMSF=args.fitRMSF,
        noStokesI=args.noStokesI,
        nBits=nBits,
        showPlots=args.showPlots,
        debug=args.debug,
        verbose=verbose,
        units=args.units,
        saveFigures=args.saveOutput,
        super_resolution=args.super_resolution,
    )
    if args.saveOutput:
        saveOutput(mDict, aDict, prefixOut, verbose)


def get_data_Q_U(filename, ycoords, xcoords):
    hduList = fits.open(filename)
    return hduList[0].data[:, ycoords, xcoords]


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
