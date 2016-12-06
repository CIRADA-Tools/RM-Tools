#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RM-clean.py                                                    #
#                                                                             #
# PURPOSE:  Run RM-clean on a dirty Faraday dispersion function.              #
#                                                                             #
# MODIFIED: 15-May-2016 by C. Purcell                                         #
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
import json
import math as m
import numpy as np

from RMutils.util_RM import do_rmclean
from RMutils.util_RM import do_rmclean_hogbom
from RMutils.util_RM import measure_FDF_parms
from RMutils.util_RM import measure_fdf_complexity

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
    'do_RMsynth_1D.py'. Saves an ASCII file containing a deconvolved FDF &
    clean-component spectrum.
    
    """
    
    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataFile", metavar="dataFile.dat", nargs=1,
                        help="ASCII file containing original spectra.")
    parser.add_argument("-c", dest="cutoff", type=float, default=-1,
                       help="CLEAN cutoff (+ve = absolute, -ve = sigma) [-1].")
    parser.add_argument("-n", dest="maxIter", type=int, default=1000,
                        help="maximum number of CLEAN iterations [1000].")
    parser.add_argument("-g", dest="gain", type=float, default=0.1,
                        help="CLEAN loop gain [0.1].")
    parser.add_argument("-p", dest="showPlots", action="store_true",
                        help="show the plots [False].")
    parser.add_argument("-a", dest="doAnimate", action="store_true",
                        help="animate the CLEAN plots [False]")
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
            print "File does not exist: '%s'." % f
            sys.exit()
    dataDir, dummy = os.path.split(args.dataFile[0])
    
    # Run RM-CLEAN on the spectrum
    run_rmclean(fdfFile      = fdfFile,
                rmsfFile     = rmsfFile,
                weightFile   = weightFile,
                rmSynthFile  = rmSynthFile,
                cutoff       = args.cutoff,
                maxIter      = args.maxIter,
                gain         = args.gain,
                prefixOut    = fileRoot,
                outDir       = dataDir,
                nBits        = 32,
                showPlots    = args.showPlots,
                doAnimate    = args.doAnimate)
    

#-----------------------------------------------------------------------------#
def run_rmclean(fdfFile, rmsfFile, weightFile, rmSynthFile, cutoff,
                maxIter=1000, gain=0.1, prefixOut="", outDir="", nBits=32,
                showPlots=False, doAnimate=False):
    """
    Run RM-CLEAN on a complex FDF spectrum given a RMSF.
    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)
    
    # Read the frequency vector for the lambda^2 array
    freqArr_Hz, weightArr = np.loadtxt(weightFile, unpack=True, dtype=dtFloat)
    lambdaSqArr_m2 = np.power(C/freqArr_Hz, 2.0)
 
    # Read the FDF from the ASCII file
    phiArr_radm2, FDFreal, FDFimag = np.loadtxt(fdfFile, unpack=True,
                                                dtype=dtFloat)
    dirtyFDF = FDFreal + 1j * FDFimag
    
    # Read the RMSF from the ASCII file
    phi2Arr_radm2, RMSFreal, RMSFimag = np.loadtxt(rmsfFile, unpack=True,
                                                   dtype=dtFloat)
    RMSFArr = RMSFreal + 1j * RMSFimag

    # Read the RM-synthesis parameters from the JSON file
    mDict = json.load(open(rmSynthFile, "r"))

    # If the cutoff is negative, assume it is a sigma level
    if cutoff<0:
        print "Using a sigma cutoff of %.1f." % (-1 * cutoff),
        cutoff = -1 * mDict["dQU_Jybm"] * cutoff
        print "Absolute value = %.3g" % cutoff

    startTime = time.time()
    
    # Perform RM-clean on the spectrum
    cleanFDF, ccArr, iterCountArr = \
              do_rmclean_hogbom(dirtyFDF         = dirtyFDF,
                                phiArr_radm2     = phiArr_radm2,
                                RMSFArr          = RMSFArr,
                                phi2Arr_radm2    = phi2Arr_radm2,
                                fwhmRMSFArr      = np.array(mDict["fwhmRMSF"]),
                                cutoff           = cutoff,
                                maxIter          = maxIter,
                                gain             = gain,
                                verbose          = False,
                                doPlots          = showPlots,
                                doAnimate        = doAnimate)
    cleanFDF #/= 1e3
    ccArr #/= 1e3

    # ALTERNATIVE RM_CLEAN CODE ----------------------------------------------#
    '''
    cleanFDF, ccArr, fwhmRMSF, iterCount = \
              do_rmclean(dirtyFDF     = dirtyFDF,
                         phiArr       = phiArr_radm2,
                         lamSqArr     = lamSqArr_m2,
                         cutoff       = cutoff,
                         maxIter      = maxIter,
                         gain         = gain,
                         weight       = weightArr,
                         RMSFArr      = RMSFArr,
                         RMSFphiArr   = phi2Arr_radm2,
                         fwhmRMSF     = mDict["fwhmRMSF"],
                         doPlots      = True)
    '''
    #-------------------------------------------------------------------------#

    endTime = time.time()
    cputime = (endTime - startTime)
    print "> RM-CLEAN completed in %.4f seconds." % cputime

    # Measure the parameters of the deconvolved FDF
    mDict = measure_FDF_parms(FDF         = cleanFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = mDict["fwhmRMSF"],
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = mDict["lam0Sq_m2"],
                              dQU         = mDict["dQU_Jybm"])
    mDict["cleanCutoff"] = cutoff
    mDict["nIter"] = int(iterCountArr)

    # Measure the complexity of the clean component spectrum
    M3 = measure_fdf_complexity(phiArr = phiArr_radm2,
                                ccFDF = ccArr)
    print "M3:", M3
    
    # Save the deconvolved FDF and CC model to ASCII files
    print "Saving the clean FDF and component model to ASCII files."
    outFile = prefixOut + "_FDFclean.dat"
    print "> %s" % outFile
    np.savetxt(outFile, zip(phiArr_radm2, cleanFDF.real, cleanFDF.imag))
    outFile = prefixOut + "_FDFmodel.dat"
    print "> %s" % outFile
    np.savetxt(outFile, zip(phiArr_radm2, ccArr))

    # Save the RM-clean measurements to a "key=value" text file
    print "Saving the measurements on the FDF in 'key=val' and JSON formats."
    outFile = prefixOut + "_RMclean.dat"
    print "> %s" % outFile
    FH = open(outFile, "w")
    for k, v in mDict.iteritems():
        FH.write("%s=%s\n" % (k, v))
    FH.close()
    outFile = prefixOut + "_RMclean.json"
    print "> %s" % outFile
    json.dump(mDict, open(outFile, "w"))
    
    # Pause to display the figure
    if showPlots or doAnimate:
        print "Press <RETURN> to exit ...",
        raw_input()

    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
