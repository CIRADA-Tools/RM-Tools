#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     cl_RMclean_1D.py                                                  #
#                                                                             #
# PURPOSE:  Command line functions for RM-clean                               #
#           on a dirty Faraday dispersion function.                           #
# CREATED:  16-Nov-2018 by J. West                                            #
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
import json
import math as m
import numpy as np
import pdb

from RMutils.util_RM import do_rmclean
from RMutils.util_RM import do_rmclean_hogbom
from RMutils.util_RM import measure_FDF_parms
from RMutils.util_RM import measure_fdf_complexity

C = 2.997924538e8 # Speed of light [m/s]

#-----------------------------------------------------------------------------#
def run_rmclean(mDictS, aDict, cutoff,
                maxIter=1000, gain=0.1, prefixOut="", outDir="", nBits=32,
                showPlots=False, doAnimate=False, verbose=False,log=print):
    """
    Run RM-CLEAN on a complex FDF spectrum given a RMSF.
    """
    phiArr_radm2 = aDict["phiArr_radm2"]
    freqArr_Hz = aDict["freqArr_Hz"]
    weightArr = aDict["weightArr"]
    dirtyFDF = aDict["dirtyFDF"]
    phi2Arr_radm2 = aDict["phi2Arr_radm2"]
    RMSFArr=aDict["RMSFArr"]


    lambdaSqArr_m2 = np.power(C/freqArr_Hz, 2.0)

    # If the cutoff is negative, assume it is a sigma level
    if verbose: log("Expected RMS noise = %.4g mJy/beam/rmsf" % (mDictS["dFDFth_Jybm"]*1e3))
    if cutoff<0:
        log("Using a sigma cutoff of %.1f." %  (-1 * cutoff))
        cutoff = -1 * mDictS["dFDFth_Jybm"] * cutoff
        log("Absolute value = %.3g" % cutoff)
    else:
        log("Using an absolute cutoff of %.3g (%.1f x expected RMS)." % (cutoff, cutoff/mDictS["dFDFth_Jybm"]))

    startTime = time.time()
    # Perform RM-clean on the spectrum
    cleanFDF, ccArr, iterCountArr = \
              do_rmclean_hogbom(dirtyFDF        = dirtyFDF,
                                phiArr_radm2    = phiArr_radm2,
                                RMSFArr         = RMSFArr,
                                phi2Arr_radm2   = phi2Arr_radm2,
                                fwhmRMSFArr     = np.array(mDictS["fwhmRMSF"]),
                                cutoff          = cutoff,
                                maxIter         = maxIter,
                                gain            = gain,
                                verbose         = verbose,
                                doPlots         = showPlots,
                                doAnimate       = doAnimate)

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
                         fwhmRMSF     = mDictS["fwhmRMSF"],
                         doPlots      = True)
    '''
    #-------------------------------------------------------------------------#

    endTime = time.time()
    cputime = (endTime - startTime)
    log("> RM-CLEAN completed in %.4f seconds." % cputime)

    # Measure the parameters of the deconvolved FDF
    mDict = measure_FDF_parms(FDF         = cleanFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = mDictS["fwhmRMSF"],
                              #dFDF        = mDictS["dFDFth_Jybm"],
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = mDictS["lam0Sq_m2"])
    mDict["cleanCutoff"] = cutoff
    mDict["nIter"] = int(iterCountArr)

    # Measure the complexity of the clean component spectrum
    mDict["mom2CCFDF"] = measure_fdf_complexity(phiArr = phiArr_radm2,
                                                FDF = ccArr)
    
    #Calculating observed errors (based on dFDFcorMAD)
    mDict["dPhiObserved_rm2"] = mDict["dPhiPeakPIfit_rm2"]*mDict["dFDFcorMAD_Jybm"]/mDictS["dFDFth_Jybm"]
    mDict["dAmpObserved_Jybm"] = mDict["dFDFcorMAD_Jybm"]
    mDict["dPolAngleFitObserved_deg"] = mDict["dPolAngleFit_deg"]*mDict["dFDFcorMAD_Jybm"]/mDictS["dFDFth_Jybm"]
    
    nChansGood = np.sum(np.where(lambdaSqArr_m2==lambdaSqArr_m2, 1.0, 0.0))
    varLamSqArr_m2 = (np.sum(lambdaSqArr_m2**2.0) -
                      np.sum(lambdaSqArr_m2)**2.0/nChansGood) / (nChansGood-1)
    mDict["dPolAngle0ChanObserved_deg"] = \
    np.degrees(np.sqrt( mDict["dFDFcorMAD_Jybm"]**2.0 / (4.0*(nChansGood-2.0)*mDict["ampPeakPIfit_Jybm"]**2.0) *
                 ((nChansGood-1)/nChansGood + mDictS["lam0Sq_m2"]**2.0/varLamSqArr_m2) ))

    
    # Save the deconvolved FDF and CC model to ASCII files
    log("Saving the clean FDF and component model to ASCII files.")
    outFile = prefixOut + "_FDFclean.dat"
    log("> %s" % outFile)
    np.savetxt(outFile, list(zip(phiArr_radm2, cleanFDF.real, cleanFDF.imag)))
    outFile = prefixOut + "_FDFmodel.dat"
    log("> %s" % outFile)
    np.savetxt(outFile, list(zip(phiArr_radm2, ccArr.real, ccArr.imag)))

    # Save the RM-clean measurements to a "key=value" text file
    log("Saving the measurements on the FDF in 'key=val' and JSON formats.")
    outFile = prefixOut + "_RMclean.dat"
    log("> %s" % outFile)
    FH = open(outFile, "w")
    for k, v in mDict.items():
        FH.write("%s=%s\n" % (k, v))
    FH.close()
    outFile = prefixOut + "_RMclean.json"
    log("> %s" % outFile)
    json.dump(mDict, open(outFile, "w"))

    
    # Print the results to the screen
    log()
    log('-'*80)
    log('RESULTS:\n')
    log('FWHM RMSF = %.4g rad/m^2' % (mDictS["fwhmRMSF"]))
    log('Pol Angle = %.4g (+/-%.4g observed, +- %.4g theoretical) deg' % (mDict["polAngleFit_deg"],mDict["dPolAngleFitObserved_deg"],mDict["dPolAngleFit_deg"]))
    log('Pol Angle 0 = %.4g (+/-%.4g observed, +- %.4g theoretical) deg' % (mDict["polAngle0Fit_deg"],mDict["dPolAngle0ChanObserved_deg"],mDict["dPolAngle0Fit_deg"]))
    log('Peak FD = %.4g (+/-%.4g observed, +- %.4g theoretical) rad/m^2' % (mDict["phiPeakPIfit_rm2"],mDict["dPhiObserved_rm2"],mDict["dPhiPeakPIfit_rm2"]))
    log('freq0_GHz = %.4g ' % (mDictS["freq0_Hz"]/1e9))
    log('I freq0 = %.4g mJy/beam' % (mDictS["Ifreq0_mJybm"]))
    log('Peak PI = %.4g (+/-%.4g observed, +- %.4g theoretical) mJy/beam' % (mDict["ampPeakPIfit_Jybm"]*1e3,mDict["dAmpObserved_Jybm"]*1e3,mDict["dAmpPeakPIfit_Jybm"]*1e3))
    log('QU Noise = %.4g mJy/beam' % (mDictS["dQU_Jybm"]*1e3))
    log('FDF Noise (theory)   = %.4g mJy/beam' % (mDictS["dFDFth_Jybm"]*1e3))
    log('FDF Noise (Corrected MAD) = %.4g mJy/beam' % (mDict["dFDFcorMAD_Jybm"]*1e3))
    log('FDF Noise (rms)   = %.4g mJy/beam' % (mDict["dFDFrms_Jybm"]*1e3))

    log('FDF SNR = %.4g ' % (mDict["snrPIfit"]))
    log()
    log('-'*80)

    # Pause to display the figure
    if showPlots or doAnimate:
        print("Press <RETURN> to exit ...", end=' ')
        input()
        
    return mDict
        
def readFiles(fdfFile, rmsfFile, weightFile, rmSynthFile, nBits):

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)
        
    # Read the RMSF from the ASCII file
    phi2Arr_radm2, RMSFreal, RMSFimag = np.loadtxt(rmsfFile, unpack=True, dtype=dtFloat)
    # Read the frequency vector for the lambda^2 array
    freqArr_Hz, weightArr = np.loadtxt(weightFile, unpack=True, dtype=dtFloat)
    # Read the FDF from the ASCII file
    phiArr_radm2, FDFreal, FDFimag = np.loadtxt(fdfFile, unpack=True, dtype=dtFloat)
    # Read the RM-synthesis parameters from the JSON file
    mDictS = json.load(open(rmSynthFile, "r"))
    dirtyFDF = FDFreal + 1j * FDFimag    
    RMSFArr = RMSFreal + 1j * RMSFimag
    
    #add array dictionary
    aDict = dict()
    aDict["phiArr_radm2"] = phiArr_radm2
    aDict["phi2Arr_radm2"] = phi2Arr_radm2
    aDict["RMSFArr"] = RMSFArr
    aDict["freqArr_Hz"] = freqArr_Hz
    aDict["weightArr"]=weightArr
    aDict["dirtyFDF"]=dirtyFDF
    
    return mDictS, aDict


