#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RMsynth_3D.py                                                  #
#                                                                             #
# PURPOSE:  Run RM-synthesis on a Stokes Q & U cubes.                         #
#                                                                             #
# MODIFIED: 15-May-2016 by C. Purcell                                         #
#                                                                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2016 Cormac R. Purcell                                        #
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
#=============================================================================#

import sys
import os
import time
import argparse
import math as m
import numpy as np
import astropy.io.fits as pf
import pdb

from RMutils.util_RM import do_rmsynth_planes
from RMutils.util_RM import get_rmsf_planes
from RMutils.util_misc import interp_images
import cl_RMsynth_3D as cl

C = 2.997924538e8 # Speed of light [m/s]


#-----------------------------------------------------------------------------#
def main():
    
    """
    Start the function to perform RM-synthesis if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run RM-synthesis on a pair of Stokes Q and U cubes (3D). This script
    correctly deals with isolated clumps of flagged voxels in the cubes (NaNs).
    Saves cubes containing the complex Faraday dispersion function (FDF), a 
    cube of double-size Rotation Measure Spread Functions, a peak Faraday
    depth map, a first-moment map and a maximum polarised intensity map.
    """

    epilog_text="""
    Output files:
        Default:
        FDF_dirty.fits: FDF, in 3 extensions: Q,U, and PI.
        FDF_maxPI.fits: 2D map of peak polarized intensity per pixel.
        FDF_maxPI.fits: 2D map of Faraday depth of highest peak, per pixel.
        RMSF.fits: 4 extensions; first 3 are RMSF cubes [Q, U, PI]
                                 4th is 2D map of RMSF FWHM.
        With -f flag:
        FDF_dirty.fits is split into three constituent components:
            FDF_real_dirty.fits: Stokes Q
            FDF_im_dirty.fits: Stokes U
            FDF_tot_dirty.fits: Polarizd Intensity (sqrt(Q^2+U^2))
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,epilog=epilog_text,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fitsQ", metavar="StokesQ.fits", nargs=1,
                        help="FITS cube containing Stokes Q data.")
    parser.add_argument("fitsU", metavar="StokesU.fits", nargs=1,
                        help="FITS cube containing Stokes U data.")
    parser.add_argument("freqFile", metavar="freqs_Hz.dat", nargs=1,
                        help="ASCII file containing the frequency vector.")
    parser.add_argument("-i", dest="fitsI", default=None,
                        help="FITS cube containing Stokes I model [None].")
    parser.add_argument("-n", dest="noiseFile", default=None,
                        help="FITS file or cube containing noise values [None].")
    parser.add_argument("-w", dest="weightType", default="uniform",
                        help="weighting [uniform] (all 1s) or 'variance'.")
    parser.add_argument("-t", dest="fitRMSF", action="store_true",
                        help="Fit a Gaussian to the RMSF [False]")
    parser.add_argument("-l", dest="phiMax_radm2", type=float, default=None,
                        help="Absolute max Faraday depth sampled (overrides NSAMPLES) [Auto].")
    parser.add_argument("-d", dest="dPhi_radm2", type=float, default=None,
                        help="Width of Faraday depth channel [Auto].")
    parser.add_argument("-o", dest="prefixOut", default="",
                        help="Prefix to prepend to output files [None].")
    parser.add_argument("-s", dest="nSamples", type=float, default=5,
                        help="Number of samples across the FWHM RMSF.")
    parser.add_argument("-f", dest="write_seperate_FDF", action="store_true",
                        help="Separate dirty FDF components into individual files [False].")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="Verbose [False].")

    args = parser.parse_args()

    # Sanity checks
    for f in args.fitsQ + args.fitsU:
        if not os.path.exists(f):
            print("File does not exist: '%s'." % f)
            sys.exit()
    dataDir, dummy = os.path.split(args.fitsQ[0])
    verbose=args.verbose
    if args.fitsI is not None:
        dataI = cl.readFitsCube(args.fitsI, verbose)[1]
    else:
        dataI=None
    if args.noiseFile is not None:
        rmsArr_Jy = cl.readFitsCube(args.noiseFile[0], verbose)[1]
    else:
        rmsArr_Jy=None
    # Run RM-synthesis on the cubes
    cl.run_rmsynth(dataQ     = cl.readFitsCube(args.fitsQ[0], verbose)[1],
                dataU        = cl.readFitsCube(args.fitsU[0], verbose)[1],
                freqArr_Hz   = cl.readFreqFile(args.freqFile[0], verbose),
                headtemplate = cl.readFitsCube(args.fitsQ[0], verbose)[0],
                dataI        = dataI,
                rmsArr_Jy    = rmsArr_Jy,
                phiMax_radm2 = args.phiMax_radm2,
                dPhi_radm2   = args.dPhi_radm2,
                nSamples     = args.nSamples,
                weightType   = args.weightType,
                prefixOut    = args.prefixOut,
                outDir       = dataDir,
                fitRMSF      = args.fitRMSF,
                nBits        = 32,
                write_seperate_FDF = args.write_seperate_FDF,
                verbose      = verbose)



#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
