#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RM-clean.py                                                    #
#                                                                             #
# PURPOSE:  Run RM-clean on a dirty Faraday dispersion function.              #
#                                                                             #
# MODIFIED: 16-Nov-2018 by J. West                                            #
#                                                                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 Cormac R. Purcell                                        #
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

import cl_RMclean_1D as clRM

C = 2.997924538e8 # Speed of light [m/s]


#-----------------------------------------------------------------------------#
def main():
    """
    Start the function to perform RM-clean if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run RM-CLEAN on an ASCII Faraday dispersion function (FDF), applying
    the rotation measure spread function created by the script
    'do_RMsynth_1D.py'. Saves ASCII files containing a deconvolved FDF &
    clean-component spectrum.
    """

    epilog_text="""
    By default, saves the following files:
    _FDFclean.dat: cleaned and restored FDF [Phi, Q, U]
    _FDFmodel.dat: clean component FDF [Phi, Q, U]
    _RMclean.dat: list of calculated paramaters describing FDF
    _RMclean.json: dictionary of calculated parameters
    """
    
    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,epilog=epilog_text,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataFile", metavar="dataFile.dat", nargs=1,
                        help="ASCII file containing original frequency spectra.")
    parser.add_argument("-c", dest="cutoff", type=float, default=-3,
                        help="CLEAN cutoff (+ve = absolute, -ve = sigma) [-3].")
    parser.add_argument("-n", dest="maxIter", type=int, default=1000,
                        help="maximum number of CLEAN iterations [1000].")
    parser.add_argument("-g", dest="gain", type=float, default=0.1,
                        help="CLEAN loop gain [0.1].")
    parser.add_argument("-p", dest="showPlots", action="store_true",
                        help="show the plots [False].")
    parser.add_argument("-a", dest="doAnimate", action="store_true",
                        help="animate the CLEAN plots [False]")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="Print verbose messages")

    args = parser.parse_args()

    # Form the input file names from prefix of the original data file
    fileRoot, dummy = os.path.splitext(args.dataFile[0])
    fdfFile = fileRoot + "_FDFdirty.dat"
    rmsfFile = fileRoot + "_RMSF.dat"
    weightFile = fileRoot + "_weight.dat"
    rmSynthFile = fileRoot + "_RMsynth.json"
    # Sanity checks
    for f in [weightFile, fdfFile, rmsfFile, rmSynthFile]:
        if not os.path.exists(f):
            print("File does not exist: '{:}'.".format(f), end=' ')
            sys.exit()
    nBits = 32
    dataDir, dummy = os.path.split(args.dataFile[0])
    mDictS, aDict = clRM.readFiles(fdfFile, rmsfFile, weightFile, rmSynthFile, nBits)  
    # Run RM-CLEAN on the spectrum
    clRM.run_rmclean(mDictS  = mDictS,
                aDict        = aDict,
                cutoff       = args.cutoff,
                maxIter      = args.maxIter,
                gain         = args.gain,
                prefixOut    = fileRoot,
                outDir       = dataDir,
                nBits        = nBits,
                showPlots    = args.showPlots,
                doAnimate    = args.doAnimate,
                verbose      = args.verbose)
    


    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
