#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RMsynth_1D.py                                                  #
#                                                                             #
# PURPOSE:  Run RM-synthesis on an ASCII Stokes I, Q & U spectrum.            #
#                                                                             #
# MODIFIED: 15-Nov-2018 by J. West                                            #
#                                                                             #
#=============================================================================#
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
#=============================================================================#
 
import sys
import os
#import time
import argparse
#import pdb

import cl_RMsynth_1d as clRM

C = 2.997924538e8 # Speed of light [m/s]


#-----------------------------------------------------------------------------#
def main():
    """
    Start the function to perform RM-synthesis if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run RM-synthesis on Stokes I, Q and U spectra (1D) stored in an ASCII
    file. The Stokes I spectrum is first fit with a polynomial and the 
    resulting model used to create fractional q = Q/I and u = U/I spectra.

    The ASCII file should the following columns, in a space separated format:
    [freq_Hz, I, Q, U, I_err, Q_err, U_err]
    OR
    [freq_Hz, Q, U, Q_err, U_err]
    Stokes units are assumed to be Jy, but output will have same units as input.

    """

    epilog_text="""
    Outputs with -S flag:
    _FDFdirty.dat: Dirty FDF/RM Spectrum [Phi, Q, U]
    _RMSF.dat: Computed RMSF [Phi, Q, U]
    _RMsynth.dat: list of derived parameters for RM spectrum
                (approximately equivalent to -v flag output)
    _RMsynth.json: dictionary of derived parameters for RM spectrum
    _weight.dat: Calculated channel weights [freq_Hz, weight]
    """
    
    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,epilog=epilog_text,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataFile", metavar="dataFile.dat", nargs=1,
                        help="ASCII file containing Stokes spectra & errors.")
    parser.add_argument("-t", dest="fitRMSF", action="store_true",
                        help="fit a Gaussian to the RMSF [False]")
    parser.add_argument("-l", dest="phiMax_radm2", type=float, default=None,
                        help="absolute max Faraday depth sampled [Auto].")
    parser.add_argument("-d", dest="dPhi_radm2", type=float, default=None,
                        help="width of Faraday depth channel [Auto].\n(overrides -s NSAMPLES flag)")
    parser.add_argument("-s", dest="nSamples", type=float, default=10,
                        help="number of samples across the RMSF lobe [10].")
    parser.add_argument("-w", dest="weightType", default="variance",
                        help="weighting [inverse variance] or 'uniform' (all 1s).")
    parser.add_argument("-o", dest="polyOrd", type=int, default=2,
                        help="polynomial order to fit to I spectrum [2].")
    parser.add_argument("-i", dest="noStokesI", action="store_true",
                        help="ignore the Stokes I spectrum [False].")
    parser.add_argument("-b", dest="bit64", action="store_true",
                        help="use 64-bit floating point precision [False (uses 32-bit)]")
    parser.add_argument("-p", dest="showPlots", action="store_true",
                        help="show the plots [False].")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="verbose output [False].")
    parser.add_argument("-S", dest="saveOutput", action="store_true",
                        help="save the arrays [False].")
    parser.add_argument("-D", dest="debug", action="store_true",
                        help="turn on debugging messages & plots [False].")
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
    verbose=args.verbose
    data = clRM.readFile(args.dataFile[0],nBits, verbose)
    
    # Run RM-synthesis on the spectra
    dict, aDict = clRM.run_rmsynth(data           = data,
                polyOrd        = args.polyOrd,
                phiMax_radm2   = args.phiMax_radm2,
                dPhi_radm2     = args.dPhi_radm2,
                nSamples       = args.nSamples,
                weightType     = args.weightType,
                fitRMSF        = args.fitRMSF,
                noStokesI      = args.noStokesI,
                nBits          = nBits,
                showPlots      = args.showPlots,
                debug          = args.debug,
                verbose        = verbose)
    #pdb.set_trace()
    if args.saveOutput:
        clRM.saveOutput(dict, aDict, prefixOut, verbose)


#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
