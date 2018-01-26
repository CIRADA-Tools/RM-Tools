#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_QUfit_1D_inest.py                                              #
#                                                                             #
# PURPOSE:  Code to simultaneously fit Stokes Q/I and U/I spectra with a      #
#           Faraday active models.                                            #
#                                                                             #
# MODIFIED: 26-Jan-2018 by C. Purcell                                         #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#   main           ... parse command line arguments and initiate procedure    #
#   run_qufit      ... main function of the QU-fitting procedure              #
#   prior_call     ... return the prior tranform fucntion given prior limits  #
#   lnlike_call    ... return a function to evaluate ln(like) given the data  #
#   wrap_arr       ... wrap the periodic values in an array                   #
#   wrap_chains    ... wrap and shift chains of periodic parameters           #
#                                                                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2018 Cormac R. Purcell                                        #
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
import shutil
import copy
import time
import imp
import argparse
import traceback
import math as m
import numpy as np
import numpy.ma as ma
import scipy.optimize as op
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

import pymultinest as pmn

from RMutils.util_misc import create_frac_spectra
from RMutils.util_misc import poly5
from RMutils.util_plotTk import plot_Ipqu_spectra_fig
from RMutils.util_plotTk import CustomNavbar
from RMutils.util_plotTk import tweakAxFormat
from RMutils import corner

C = 2.997924538e8 # Speed of light [m/s]


#-----------------------------------------------------------------------------#
def main():
    """
    Start the run_qufit procedure if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    """
    
    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataFile", metavar="dataFile.dat", nargs=1,
                        help="ASCII file containing Stokes spectra & errors.")
    parser.add_argument("-m", dest="modelNum", type=int, default=1,
                        help="model number to fit [1].")
    parser.add_argument("-o", dest="polyOrd", type=int, default=2,
                        help="polynomial order to fit to I spectrum [2].")
    parser.add_argument("-i", dest="noStokesI", action="store_true",
                        help="ignore the Stokes I spectrum [False].")
    parser.add_argument("-p", dest="showPlots", action="store_true",
                        help="show the plots [False].")
    parser.add_argument("-d", dest="debug", action="store_true",
                        help="turn on debugging messages/plots [False].")
    parser.add_argument("-r", dest="redo", action="store_true",
                        help="re-do the fitting [False].")
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.dataFile[0]):
        print "File does not exist: '%s'." % args.dataFile[0]
        sys.exit()
    dataDir, dummy = os.path.split(args.dataFile[0])
    
    # Run the QU-fitting procedure
    run_qufit(dataFile     = args.dataFile[0],
              modelNum     = args.modelNum,
              outDir       = dataDir,
              polyOrd      = args.polyOrd,
              nBits        = 32,
              noStokesI    = args.noStokesI,
              showPlots    = args.showPlots,
              debug        = args.debug,
              redo         = args.redo)
    

#-----------------------------------------------------------------------------#
def run_qufit(dataFile, modelNum, outDir="", polyOrd=3, nBits=32,
              noStokesI=False, showPlots=False, debug=False, redo=False):
    """Root function controlling the fitting procedure."""
    
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)
    
    # Output prefix is derived from the input file name
    prefixOut, ext = os.path.splitext(dataFile)
    prefixOut += "_nest/"
    if os.path.exists(prefixOut):
        if redo:
            shutil.rmtree(prefixOut, True)
            os.mkdir(prefixOut)
    else:
        os.mkdir(prefixOut)
    
    # Read the data-file. Format=space-delimited, comments='#'.
    print "Reading the data file '%s':" % dataFile
    # freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy
    try:
        print "> Trying [freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy]",
        (freqArr_Hz, IArr_Jy, QArr_Jy, UArr_Jy,
         dIArr_Jy, dQArr_Jy, dUArr_Jy) = \
         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
        print "... success."
    except Exception:
        print "...failed."
        # freq_Hz, Q_Jy, U_Jy, dQ_Jy, dU_Jy
        try:
            print "Reading [freq_Hz, Q_Jy, U_Jy,  dQ_Jy, dU_Jy]",
            (freqArr_Hz, QArr_Jy, UArr_Jy, dQArr_Jy, dUArr_Jy) = \
                         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
            print "... success."
            noStokesI = True
        except Exception:
            print "...failed."
            if debug:
                print traceback.format_exc()
            sys.exit()
    
    # If no Stokes I present, create a dummy spectrum = unity
    if noStokesI:
        print "Warn: no Stokes I data in use."
        IArr_Jy = np.ones_like(QArr_Jy)
        dIArr_Jy = np.zeros_like(QArr_Jy)

    # Convert to GHz and mJy for convenience
    print "Successfully read in the Stokes spectra."
    freqArr_GHz = freqArr_Hz / 1e9
    lamSqArr_m2 = np.power(C/freqArr_Hz, 2.0)
    IArr_mJy = IArr_Jy * 1e3
    QArr_mJy = QArr_Jy * 1e3
    UArr_mJy = UArr_Jy * 1e3
    dIArr_mJy = dIArr_Jy * 1e3
    dQArr_mJy = dQArr_Jy * 1e3
    dUArr_mJy = dUArr_Jy * 1e3

    # Fit the Stokes I spectrum and create the fractional spectra
    IModArr, qArr, uArr, dqArr, duArr, IfitDict = \
             create_frac_spectra(freqArr=freqArr_GHz,
                                 IArr=IArr_mJy,
                                 QArr=QArr_mJy,
                                 UArr=UArr_mJy,
                                 dIArr=dIArr_mJy,
                                 dQArr=dQArr_mJy,
                                 dUArr=dUArr_mJy,
                                 polyOrd=polyOrd,
                                 verbose=True)
    
    # Plot the data and the Stokes I model fit
    if showPlots:
        print "Plotting the input data and spectral index fit."
        freqHirArr_Hz =  np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)     
        IModHirArr_mJy = poly5(IfitDict["p"])(freqHirArr_Hz/1e9)    
        specFig = plt.figure(figsize=(12, 8))
        plot_Ipqu_spectra_fig(freqArr_Hz     = freqArr_Hz,
                              IArr_mJy       = IArr_mJy, 
                              qArr           = qArr, 
                              uArr           = uArr, 
                              dIArr_mJy      = dIArr_mJy,
                              dqArr          = dqArr,
                              duArr          = duArr,
                              freqHirArr_Hz  = freqHirArr_Hz,
                              IModArr_mJy    = IModHirArr_mJy,
                              fig            = specFig)
        
        # Use the custom navigation toolbar
        try:
            specFig.canvas.toolbar.pack_forget()
            CustomNavbar(specFig.canvas, specFig.canvas.toolbar.window)
        except Exception:
            pass

        # Display the figure
        specFig.canvas.draw()
        specFig.show()

    #-------------------------------------------------------------------------#

    # Load the model and parameters from the relevant file
    print "\nLoading the model from file 'models_ns/m%d.py' ..."  % modelNum
    mod = imp.load_source("m%d" % modelNum, "models_ns/m%d.py" % modelNum)
    global model
    model = mod.model
    inParms = mod.inParms
    runParmDict = mod.runParmDict

    # Unpack the inParms structure
    nDim = len(inParms)
    parNames = [x["parname"] for x in inParms]
    labels = [x["label"] for x in inParms]
    values = [x["value"] for x in inParms]    
    bounds = [x["bounds"] for x in inParms]
    priorTypes = [x["priortype"] for x in inParms]
    wraps = [x["wrap"] for x in inParms]
    
    # Set the prior function given the bounds of each parameter
    prior = prior_call(priorTypes, bounds, values)
    
    # Set the likelihood function
    lnlike = lnlike_call(parNames, lamSqArr_m2, qArr, dqArr, uArr, duArr)
    
    # Run nested sampling
    pmn.run(lnlike,
            prior,
            nDim,
            wrapped_params       = wraps,
            outputfiles_basename = prefixOut,
            n_live_points        = runParmDict["nPoints"],
            verbose              = runParmDict["verbose"])
    
    # Query the analyser object for results
    aObj = pmn.Analyzer(n_params=nDim, outputfiles_basename=prefixOut)
    statDict =  aObj.get_stats()
    fitDict =  aObj.get_best_fit()

    # Get the best fitting values and marginals
    p = fitDict["parameters"]
    lnLike = fitDict["log_likelihood"]
    lnEvidence = statDict["nested sampling global log-evidence"]
    dLnEvidence = statDict["nested sampling global log-evidence error"]
    med = [None] *nDim
    dp = [[None, None]]*nDim
    for i in range(nDim):   
        dp[i] = statDict["marginals"][i]['1sigma']
        dp[i] = statDict["marginals"][i]['1sigma']
        med[i] = statDict["marginals"][i]['median']
        
    # Re-centre and wrap periodic parameters
    iWrap = [i for i, e in enumerate(wraps) if e != 0]
    for i in iWrap:
        wrapLow = bounds[i][0]
        wrapHigh = bounds[i][1]
        rng = wrapHigh - wrapLow
        wrapCent = wrapLow + (wrapHigh - wrapLow)/2.0
        wrapLow += (p[i] - wrapCent)
        wrapHigh += (p[i] - wrapCent)
        dp[i] = ((dp[i]-wrapLow) % rng) + wrapLow
        
    # Calculate goodness-of-fit parameters
    nSamp = len(lamSqArr_m2)
    dof = nSamp - nDim -1
    chiSq = -2.0*lnLike
    chiSqRed = chiSq/dof
    AIC = 2.0*nDim - 2.0 * lnLike
    AICc = 2.0*nDim*(nDim+1)/(nSamp-nDim-1) - 2.0 * lnLike
    BIC = nDim * np.log(nSamp) - 2.0 * lnLike
        
    # Summary of run
    print("-"*80)
    print("RESULTS:")
    print "DOF:", dof
    print "CHISQ:", chiSq
    print "CHISQ RED:", chiSqRed
    print "AIC:", AIC
    print "AICc", AICc
    print "BIC", BIC
    print "ln(EVIDENCE)", lnEvidence
    print "dLn(EVIDENCE)", dLnEvidence
    print
    print '-'*80
    for i in range(len(p)):
        print("p%d = %.4f +/- %.4f/%.4f" % \
              (i, p[i], p[i]-dp[i][0], dp[i][1]-p[i]))

    # Plot the results
    if showPlots:
        print "Plotting the best-fitting model."
        lamSqHirArr_m2 =  np.linspace(lamSqArr_m2[0], lamSqArr_m2[-1], 10000)
        freqHirArr_Hz = C / np.sqrt(lamSqHirArr_m2)
        IModArr_mJy = poly5(IfitDict["p"])(freqHirArr_Hz/1e9)        
        pDict = {k:v for k, v in zip(parNames, p)}
        quModArr = model(pDict, lamSqHirArr_m2)
        specFig.clf()
        plot_Ipqu_spectra_fig(freqArr_Hz     = freqArr_Hz,
                              IArr_mJy       = IArr_mJy, 
                              qArr           = qArr, 
                              uArr           = uArr, 
                              dIArr_mJy      = dIArr_mJy,
                              dqArr          = dqArr,
                              duArr          = duArr,
                              freqHirArr_Hz  = freqHirArr_Hz,
                              IModArr_mJy    = IModArr_mJy,
                              qModArr        = quModArr.real, 
                              uModArr        = quModArr.imag,
                              fig            = specFig)
        specFig.canvas.draw()
        
        print "Plotting the corner-plot."
        chains =  aObj.get_equal_weighted_posterior()
        chains = wrap_chains(chains, wraps, bounds, p)
        cornerFig = corner.corner(xs      = chains[:, :nDim],
                                  labels  = labels,
                                  range   = [0.99999]*nDim,
                                  truths  = p,
                                  bins    = 30)
        cornerFig.show()
        
        print "> Press <RETURN> to exit ...",
        raw_input()

    
#-----------------------------------------------------------------------------#
def prior_call(priorTypes, bounds, values):
    """Returns a function to transform (0-1) range to the distribution of 
    values for each parameter. Note that a numpy vectorised version of this
    function fails because of type-errors."""

    def rfunc(p, nDim, nParms):
	for i in range(nDim):
            if priorTypes[i] == "log":
		bMin = np.log(np.abs(bounds[i][0]))
		bMax = np.log(np.abs(bounds[i][1]))	
		p[i] *= bMax - bMin
		p[i] += bMin
		p[i] = np.exp(p[i])
            elif priorTypes[i] == "normal":
                bMin, bMax = bounds[i]
                sigma = (bMax - bMin)/2.0
                mu = bMin + sigma
                p[i] = mu + sigma * ndtri(p[i])
            elif priorTypes[i] == "fixed":
		p[i] = values[i]
            else: # uniform (linear)
                bMin, bMax = bounds[i]
                p[i] = bMin + p[i] * (bMax - bMin)
        return p
    
    return rfunc

    
#-----------------------------------------------------------------------------#
def lnlike_call(parNames, lamSqArr_m2, qArr, dqArr, uArr, duArr):
    """ Returns a function to evaluate the log-likelihood """

    def lnlike(p, nDim, nParms):
        
        # Evaluate the model and calculate the ln(like)
        pDict = {k:v for k, v in zip(parNames, p)}
        quMod = model(pDict, lamSqArr_m2)
        dquArr = np.sqrt(np.power(dqArr, 2) + np.power(duArr, 2))
        chiSqNrm = np.nansum( np.power((qArr-quMod.real)/dqArr, 2) +
                              np.power((uArr-quMod.imag)/duArr, 2) +
                              np.log(2 * np.pi * np.power(dquArr, 2)) )
        return -chiSqNrm/2.0
    
    return lnlike


#-----------------------------------------------------------------------------#
def wrap_arr(arr, wrapLow=-90.0, wrapHigh=90.0):
    """Wrap the values in an array (e.g., angles)."""
    
    rng = wrapHigh - wrapLow
    arr = ((arr-wrapLow) % rng) + wrapLow
    return arr


#-----------------------------------------------------------------------------#
def wrap_chains(chains, wraps, bounds, p, verbose=False):

    # Get the indices of the periodic parameters
    iWrap = [i for i, e in enumerate(wraps) if e != 0]

    # Loop through the chains for periodic parameters
    for i in iWrap:
        wrapLow = bounds[i][0]
        wrapHigh = bounds[i][1]
        rng = wrapHigh - wrapLow
        
        # Shift the wrapping to centre on the best fit value
        wrapCent = wrapLow + (wrapHigh - wrapLow)/2.0
        wrapLow += (p[i] - wrapCent)
        wrapHigh += (p[i] - wrapCent)
        chains[:, i] = ((chains[:, i]-wrapLow) % rng) + wrapLow
        if verbose:
            print "> Wrapped parameter '%d' in range [%s, %s] ..." % \
            (i, wrapLow, wrapHigh),

    return chains
    

#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
