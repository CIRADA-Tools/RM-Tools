#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RMclean_1D.py                                                  #
#                                                                             #
# PURPOSE:  Command line functions for RM-clean                               #
#           on a dirty Faraday dispersion function.                           #
# CREATED:  16-Nov-2018 by J. West                                            #
# MODIFIED: 16-Nov-2018 by J. West                                            #
# MODIFIED: 23-October-2019 by A. Thomson                                     #
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

import time
import os
import sys
import json
import numpy as np

from RMutils.util_RM import do_rmclean_hogbom
from RMutils.util_RM import measure_FDF_parms
from RMutils.util_RM import measure_fdf_complexity

C = 2.997924538e8 # Speed of light [m/s]
#-----------------------------------------------------------------------------#
def run_rmclean(mDictS, aDict, cutoff,
                maxIter=1000, gain=0.1, nBits=32,
                showPlots=False, verbose=False, log=print):
    """Run RM-CLEAN on a complex FDF spectrum given a RMSF.

    Args:
        mDictS (dict): Summary of RM synthesis results.
        aDict (dict): Data output by RM synthesis.
        cutoff (float): CLEAN cutoff in flux units

    Kwargs:
        maxIter (int): Maximum number of CLEAN iterations per pixel.
        gain (float): CLEAN loop gain.
        nBits (int): Precision of floating point numbers.
        showPlots (bool): Show plots?
        verbose (bool): Verbosity.
        log (function): Which logging function to use.

    Returns:
        mDict (dict): Summary of RMCLEAN results.
        arrdict (dict): Data output by RMCLEAN.

    """
    phiArr_radm2 = aDict["phiArr_radm2"]
    freqArr_Hz = aDict["freqArr_Hz"]
    weightArr = aDict["weightArr"]
    dirtyFDF = aDict["dirtyFDF"]
    phi2Arr_radm2 = aDict["phi2Arr_radm2"]
    RMSFArr=aDict["RMSFArr"]


    lambdaSqArr_m2 = np.power(C/freqArr_Hz, 2.0)

    # If the cutoff is negative, assume it is a sigma level
    if verbose: log("Expected RMS noise = %.4g flux units" % (mDictS["dFDFth"]))
    if cutoff<0:
        if verbose: log("Using a sigma cutoff of %.1f." %  (-1 * cutoff))
        cutoff = -1 * mDictS["dFDFth"] * cutoff
        if verbose: log("Absolute value = %.3g" % cutoff)
    else:
        if verbose: log("Using an absolute cutoff of %.3g (%.1f x expected RMS)." % (cutoff, cutoff/mDictS["dFDFth"]))

    startTime = time.time()
    # Perform RM-clean on the spectrum
    cleanFDF, ccArr, iterCountArr, residFDF = \
              do_rmclean_hogbom(dirtyFDF        = dirtyFDF,
                                phiArr_radm2    = phiArr_radm2,
                                RMSFArr         = RMSFArr,
                                phi2Arr_radm2   = phi2Arr_radm2,
                                fwhmRMSFArr     = np.array(mDictS["fwhmRMSF"]),
                                cutoff          = cutoff,
                                maxIter         = maxIter,
                                gain            = gain,
                                verbose         = verbose,
                                doPlots         = showPlots)

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
    if verbose: log("> RM-CLEAN completed in %.4f seconds." % cputime)

    # Measure the parameters of the deconvolved FDF
    mDict = measure_FDF_parms(FDF         = cleanFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = mDictS["fwhmRMSF"],
                              #dFDF        = mDictS["dFDFth"],
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = mDictS["lam0Sq_m2"])
    mDict["cleanCutoff"] = cutoff
    mDict["nIter"] = int(iterCountArr)

    # Measure the complexity of the clean component spectrum
    mDict["mom2CCFDF"] = measure_fdf_complexity(phiArr = phiArr_radm2,
                                                FDF = ccArr)

    #Calculating observed errors (based on dFDFcorMAD)
    mDict["dPhiObserved_rm2"] = mDict["dPhiPeakPIfit_rm2"]*mDict["dFDFcorMAD"]/mDictS["dFDFth"]
    mDict["dAmpObserved"] = mDict["dFDFcorMAD"]
    mDict["dPolAngleFitObserved_deg"] = mDict["dPolAngleFit_deg"]*mDict["dFDFcorMAD"]/mDictS["dFDFth"]

    nChansGood = np.sum(np.where(lambdaSqArr_m2==lambdaSqArr_m2, 1.0, 0.0))
    varLamSqArr_m2 = (np.sum(lambdaSqArr_m2**2.0) -
                      np.sum(lambdaSqArr_m2)**2.0/nChansGood) / (nChansGood-1)
    mDict["dPolAngle0ChanObserved_deg"] = \
    np.degrees(np.sqrt( mDict["dFDFcorMAD"]**2.0 / (4.0*(nChansGood-2.0)*mDict["ampPeakPIfit"]**2.0) *
                 ((nChansGood-1)/nChansGood + mDictS["lam0Sq_m2"]**2.0/varLamSqArr_m2) ))





    if verbose:
        # Print the results to the screen
        log()
        log('-'*80)
        log('RESULTS:\n')
        log('FWHM RMSF = %.4g rad/m^2' % (mDictS["fwhmRMSF"]))
        log('Pol Angle = %.4g (+/-%.4g observed, +- %.4g theoretical) deg' % (mDict["polAngleFit_deg"],mDict["dPolAngleFitObserved_deg"],mDict["dPolAngleFit_deg"]))
        log('Pol Angle 0 = %.4g (+/-%.4g observed, +- %.4g theoretical) deg' % (mDict["polAngle0Fit_deg"],mDict["dPolAngle0ChanObserved_deg"],mDict["dPolAngle0Fit_deg"]))
        log('Peak FD = %.4g (+/-%.4g observed, +- %.4g theoretical) rad/m^2' % (mDict["phiPeakPIfit_rm2"],mDict["dPhiObserved_rm2"],mDict["dPhiPeakPIfit_rm2"]))
        log('freq0_GHz = %.4g ' % (mDictS["freq0_Hz"]/1e9))
        log('I freq0 = %.4g %s' % (mDictS["Ifreq0"],mDictS["units"]))
        log('Peak PI = %.4g (+/-%.4g observed, +- %.4g theoretical) %s' % (mDict["ampPeakPIfit"],mDict["dAmpObserved"],mDict["dAmpPeakPIfit"],mDictS["units"]))
        log('QU Noise = %.4g %s' % (mDictS["dQU"],mDictS["units"]))
        log('FDF Noise (theory)   = %.4g %s' % (mDictS["dFDFth"],mDictS["units"]))
        log('FDF Noise (Corrected MAD) = %.4g %s' % (mDict["dFDFcorMAD"],mDictS["units"]))
        log('FDF Noise (rms)   = %.4g %s' % (mDict["dFDFrms"],mDictS["units"]))

        log('FDF SNR = %.4g ' % (mDict["snrPIfit"]))
        log()
        log('-'*80)

    # Pause to display the figure
    if showPlots:
        plot_clean_spec(phiArr_radm2, dirtyFDF, cleanFDF, ccArr, residFDF,
                    cutoff,mDictS["units"])
        print("Press <RETURN> to exit ...", end=' ')
        input()

    #add array dictionary
    arrdict = dict()
    arrdict["phiArr_radm2"] = phiArr_radm2
    arrdict["freqArr_Hz"] = freqArr_Hz
    arrdict["cleanFDF"]=cleanFDF
    arrdict["ccArr"]=ccArr
    arrdict["iterCountArr"]=iterCountArr
    arrdict["residFDF"]=residFDF

    return mDict, arrdict


def saveOutput(outdict, arrdict, prefixOut="", verbose=False, log=print):
    """
    Saves RM-CLEAN results to text files. The clean (restored) FDF, model FDF 
    (clean components) are saved, as is two files (.dat and .json)
    reporting the fitting results as key-value pairs.
    Inputs:
        outdict: the results dictionary (mDict) from RM-CLEAN.
        arrdict: the array dictionary (aDict) from RM-CLEAN.
        prefixOut (str): name prefix to be given to output files 
            (including relative/absolute directory to save to)
        verbose (bool): print verbose messages?
        log (function): function to use when printing verbose messages.
    """
    # Get data
    phiArr_radm2 = arrdict["phiArr_radm2"]
    cleanFDF = arrdict["cleanFDF"]
    ccArr = arrdict["ccArr"]

    # Save the deconvolved FDF and CC model to ASCII files
    if verbose: log("Saving the clean FDF and component model to ASCII files.")
    outFile = prefixOut + "_FDFclean.dat"
    if verbose: log("> %s" % outFile)
    np.savetxt(outFile, list(zip(phiArr_radm2, cleanFDF.real, cleanFDF.imag)))
    outFile = prefixOut + "_FDFmodel.dat"
    if verbose: log("> %s" % outFile)
    np.savetxt(outFile, list(zip(phiArr_radm2, ccArr.real, ccArr.imag)))

    # Save the RM-clean measurements to a "key=value" text file
    if verbose: log("Saving the measurements on the FDF in 'key=val' and JSON formats.")
    outFile = prefixOut + "_RMclean.dat"
    if verbose: log("> %s" % outFile)
    FH = open(outFile, "w")
    for k, v in outdict.items():
        FH.write("%s=%s\n" % (k, v))
    FH.close()
    outFile = prefixOut + "_RMclean.json"
    if verbose: log("> %s" % outFile)
    json.dump(outdict, open(outFile, "w"))

def readFiles(fdfFile, rmsfFile, weightFile, rmSynthFile, nBits):
    """
    Read in the RM-synthesis output files and assemble back into dictionaries.
    Inputs:
        fdfFile (str): file path to the FDF.
        rmsfFile (str): file path to the RMSF.
        weightfile (str): file path to the channel weights.
        rmSynthFile (str): file path to the RMsynth json file.
        nBits (int): number of bits to use when storing the data.
    """

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


def plot_clean_spec(phiArr_radm2, dirtyFDF, cleanFDF, ccArr, residFDF,
                    cutoff,units):
    """
    Plotting code for CLEANed Faraday depth spectra.
    Inputs: 
        phiArr_radm2 (array): array of Faraday depth values.
        dirty FDF (array): dirty Faraday depth spectrum.
        cleanFDF (array): cleaned (restored) Faraday depth spectrum
        ccArr (array): clean component array
        residFDF (array): residual Faraday depth spectrum
        cutoff (float): clean threshold
        units (str): name of flux unit
    """
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(12.0, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    dirtyFDF=np.squeeze(dirtyFDF)
    cleanFDF=np.squeeze(cleanFDF)
    ccArr=np.squeeze(ccArr)
    residFDF=np.squeeze(residFDF)

    fig.show()


    ax1.cla()
    ax2.cla()
    ax1.step(phiArr_radm2, np.abs(dirtyFDF),
             color="grey", marker="None", mfc="w", mec="g", ms=10,
             where="mid", label="Dirty FDF")
    ax1.step(phiArr_radm2, np.abs(ccArr), color="g",
             marker="None", mfc="w", mec="g", ms=10, where="mid",
             label="Clean Components")
    ax1.step(phiArr_radm2, np.abs(residFDF), color="magenta",
             marker="None", mfc="w", mec="g", ms=10, where="mid",
             label="Residual FDF")
    ax1.step(phiArr_radm2, np.abs(cleanFDF), color="k",
             marker="None", mfc="w", mec="g", ms=10, where="mid", lw=1.5,
             label="Clean FDF")
    ax1.axhline(cutoff, color="r", ls="--", label="Clean cutoff")
    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax1.set_ylabel("Flux Density ("+units+')')
    leg = ax1.legend(numpoints=1, loc='upper right', shadow=False,
                     borderaxespad=0.3, bbox_to_anchor=(1.00, 1.00))
    for t in leg.get_texts():
        t.set_fontsize('small')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)
    [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax2.step(phiArr_radm2, np.abs(residFDF), color="magenta",
             marker="None", mfc="w", mec="g", ms=10, where="mid",
             label="Residual FDF")
    ax2.step(phiArr_radm2, np.abs(ccArr), color="g",
             marker="None", mfc="w", mec="g", ms=10, where="mid",
             label="Clean Components")
    ax2.axhline(cutoff, color="r", ls="--", label="Clean cutoff")
    ax2.set_ylim(0, cutoff*3.0)
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.set_ylabel("Flux Density ("+units+')')
    ax2.set_xlabel("$\phi$ rad m$^{-2}$")
    leg = ax2.legend(numpoints=1, loc='upper right', shadow=False,
                     borderaxespad=0.3, bbox_to_anchor=(1.00, 1.00))
    for t in leg.get_texts():
        t.set_fontsize('small')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)
    ax2.autoscale_view(True, True, True)
    plt.draw()

#-----------------------------------------------------------------------------#
def main():
    import argparse
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
    mDictS, aDict = readFiles(fdfFile, rmsfFile, weightFile, rmSynthFile, nBits)
    # Run RM-CLEAN on the spectrum
    outdict, arrdict = run_rmclean(mDictS  = mDictS,
                                aDict        = aDict,
                                cutoff       = args.cutoff,
                                maxIter      = args.maxIter,
                                gain         = args.gain,
                                nBits        = nBits,
                                showPlots    = args.showPlots,
                                verbose      = args.verbose)

    # Save output
    saveOutput(outdict,
            arrdict,
            prefixOut    = fileRoot,
            verbose      = args.verbose)



#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
