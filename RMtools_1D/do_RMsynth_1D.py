#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RMsynth_1D.py                                                  #
#                                                                             #
# PURPOSE:  Run RM-synthesis on an ASCII Stokes I, Q & U spectrum.            #
#                                                                             #
# MODIFIED: 27-Apr-2017 by C. Purcell                                         #
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
import traceback
import json
import math as m
import numpy as np
import matplotlib.pyplot as plt

from RMutils.util_RM import do_rmsynth
from RMutils.util_RM import do_rmsynth_planes
from RMutils.util_RM import get_rmsf_planes
from RMutils.util_RM import measure_FDF_parms
from RMutils.util_RM import measure_qu_complexity
from RMutils.util_misc import nanmedian
from RMutils.util_misc import toscalar
from RMutils.util_misc import create_frac_spectra
from RMutils.util_misc import poly5
from RMutils.util_plotTk import plot_Ipqu_spectra_fig
from RMutils.util_plotTk import plot_rmsf_fdf_fig
from RMutils.util_plotTk import CustomNavbar

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
    """
    
    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataFile", metavar="dataFile.dat", nargs=1,
                        help="ASCII file containing Stokes spectra & errors.")
    parser.add_argument("-t", dest="fitRMSF", action="store_true",
                        help="fit a Gaussian to the RMSF [False]")
    parser.add_argument("-l", dest="phiMax_radm2", type=float, default=None,
                        help="absolute max Faraday depth sampled [Auto].")
    parser.add_argument("-d", dest="dPhi_radm2", type=float, default=None,
                        help="width of Faraday depth channel [Auto].")
    parser.add_argument("-s", dest="nSamples", type=float, default=10,
                        help="number of samples across the RMSF lobe [10].")
    parser.add_argument("-w", dest="weightType", default="variance",
                        help="weighting [variance] or 'natural' (all 1s).")
    parser.add_argument("-o", dest="polyOrd", type=int, default=2,
                        help="polynomial order to fit to I spectrum [2].")
    parser.add_argument("-i", dest="noStokesI", action="store_true",
                        help="ignore the Stokes I spectrum [False].")
    parser.add_argument("-p", dest="showPlots", action="store_true",
                        help="show the plots [False].")
    parser.add_argument("-D", dest="debug", action="store_true",
                        help="turn on debugging messages [False].")
    args = parser.parse_args()
    
    # Sanity checks
    if not os.path.exists(args.dataFile[0]):
        print "File does not exist: '%s'." % args.dataFile[0]
        sys.exit()
    dataDir, dummy = os.path.split(args.dataFile[0])

    # Run RM-synthesis on the spectra
    run_rmsynth(dataFile     = args.dataFile[0],
                polyOrd      = args.polyOrd,
                phiMax_radm2 = args.phiMax_radm2,
                dPhi_radm2   = args.dPhi_radm2,
                nSamples     = args.nSamples,
                weightType   = args.weightType,
                fitRMSF      = args.fitRMSF,
                noStokesI    = args.noStokesI,
                nBits        = 32,
                showPlots    = args.showPlots,
                debug        = args.debug)

    
#-----------------------------------------------------------------------------#
def run_rmsynth(dataFile, polyOrd=3, phiMax_radm2=None, dPhi_radm2=None, 
                nSamples=10.0, weightType="variance", fitRMSF=False,
                noStokesI=False, nBits=32, showPlots=False, debug=False):
    """
    Read the I, Q & U data from the ASCII file and run RM-synthesis.
    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Output prefix is derived from the input file name
    prefixOut, ext = os.path.splitext(dataFile)

    # Read the data-file. Format=space-delimited, comments="#".
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
            print "> Trying [freq_Hz, Q_Jy, U_Jy,  dQ_Jy, dU_Jy]",
            (freqArr_Hz, QArr_Jy, UArr_Jy, dQArr_Jy, dUArr_Jy) = \
                         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
            print "... success."
            noStokesI = True
        except Exception:
            print "...failed."
            if debug:
                print traceback.format_exc()
            sys.exit()
    print "Successfully read in the Stokes spectra."

    # If no Stokes I present, create a dummy spectrum = unity
    if noStokesI:
        print "Warn: no Stokes I data in use."
        IArr_Jy = np.ones_like(QArr_Jy)
        dIArr_Jy = np.zeros_like(QArr_Jy)
        
    # Convert to GHz and mJy for convenience
    freqArr_GHz = freqArr_Hz / 1e9
    IArr_mJy = IArr_Jy * 1e3
    QArr_mJy = QArr_Jy * 1e3
    UArr_mJy = UArr_Jy * 1e3
    dIArr_mJy = dIArr_Jy * 1e3
    dQArr_mJy = dQArr_Jy * 1e3
    dUArr_mJy = dUArr_Jy * 1e3
    dQUArr_mJy = np.sqrt(np.power(dQArr_mJy, 2) + np.power(dUArr_mJy, 2))
    dQUArr_Jy = dQUArr_mJy / 1e3
 
    # Fit the Stokes I spectrum and create the fractional spectra
    IModArr, qArr, uArr, dqArr, duArr, fitDict = \
             create_frac_spectra(freqArr  = freqArr_GHz,
                                 IArr     = IArr_mJy,
                                 QArr     = QArr_mJy,
                                 UArr     = UArr_mJy,
                                 dIArr    = dIArr_mJy,
                                 dQArr    = dQArr_mJy,
                                 dUArr    = dUArr_mJy,
                                 polyOrd  = polyOrd,
                                 verbose  = True,
                                 debug    = debug)

    # Plot the data and the Stokes I model fit
    if showPlots:
        print "Plotting the input data and spectral index fit."
        freqHirArr_Hz =  np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)     
        IModHirArr_mJy = poly5(fitDict["p"])(freqHirArr_Hz/1e9)    
        specFig = plt.figure(figsize=(12.0, 8))
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

        # Use the custom navigation toolbar (does not work on Mac OS X)
        try:
            specFig.canvas.toolbar.pack_forget()
            CustomNavbar(specFig.canvas, specFig.canvas.toolbar.window)
        except Exception:
            pass

        # Display the figure
        specFig.show()

    #-------------------------------------------------------------------------#

    # Calculate some wavelength parameters
    lambdaSqArr_m2 = np.power(C/freqArr_Hz, 2.0)
    dFreq_Hz = np.nanmin(np.abs(np.diff(freqArr_Hz)))
    lambdaSqRange_m2 = ( np.nanmax(lambdaSqArr_m2) -
                         np.nanmin(lambdaSqArr_m2) )        
    dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
    dLambdaSqMax_m2 = np.nanmax(np.abs(np.diff(lambdaSqArr_m2)))

    # Set the Faraday depth range
    fwhmRMSF_radm2 = 2.0 * m.sqrt(3.0) / lambdaSqRange_m2
    if dPhi_radm2 is None:
        dPhi_radm2 = fwhmRMSF_radm2 / nSamples
    if phiMax_radm2 is None:
        phiMax_radm2 = m.sqrt(3.0) / dLambdaSqMax_m2
        phiMax_radm2 = max(phiMax_radm2, 600.0)    # Force the minimum phiMax

    # Faraday depth sampling. Zero always centred on middle channel
    nChanRM = round(abs((phiMax_radm2 - 0.0) / dPhi_radm2)) * 2.0 + 1.0
    startPhi_radm2 = - (nChanRM-1.0) * dPhi_radm2 / 2.0
    stopPhi_radm2 = + (nChanRM-1.0) * dPhi_radm2 / 2.0
    phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, nChanRM)
    phiArr_radm2 = phiArr_radm2.astype(dtFloat)
    print "PhiArr = %.2f to %.2f by %.2f (%d chans)." % (phiArr_radm2[0],
                                                        phiArr_radm2[-1],
                                                        float(dPhi_radm2),
                                                        nChanRM)
    
    # Calculate the weighting as 1/sigma^2 or all 1s (natural)
    if weightType=="variance":
        weightArr = 1.0 / np.power(dQUArr_mJy, 2.0)
    else:
        weightType = "natural"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    print "Weight type is '%s'." % weightType

    startTime = time.time()
    
    # Perform RM-synthesis on the spectrum
    dirtyFDF, lam0Sq_m2 = do_rmsynth_planes(dataQ           = qArr,
                                            dataU           = uArr,
                                            lambdaSqArr_m2  = lambdaSqArr_m2, 
                                            phiArr_radm2    = phiArr_radm2, 
                                            weightArr       = weightArr,
                                            nBits           = 32,
                                            verbose         = True)
    
    # Calculate the Rotation Measure Spread Function
    RMSFArr, phi2Arr_radm2, fwhmRMSFArr, fitStatArr = \
        get_rmsf_planes(lambdaSqArr_m2  = lambdaSqArr_m2,
                        phiArr_radm2    = phiArr_radm2, 
                        weightArr       = weightArr, 
                        mskArr          = np.isnan(qArr),
                        lam0Sq_m2       = lam0Sq_m2, 
                        double          = True, 
                        fitRMSF         = fitRMSF, 
                        fitRMSFreal     = False, 
                        nBits           = 32,
                        verbose         = True)
    fwhmRMSF = float(fwhmRMSFArr)
    
    # ALTERNATE RM-SYNTHESIS CODE --------------------------------------------#

    #dirtyFDF, [phi2Arr_radm2, RMSFArr], lam0Sq_m2, fwhmRMSF = \
    #          do_rmsynth(qArr, uArr, lambdaSqArr_m2, phiArr_radm2, weightArr)
    
    #-------------------------------------------------------------------------#
    
    endTime = time.time()
    cputime = (endTime - startTime)
    print "> RM-synthesis completed in %.2f seconds." % cputime
    
    # Determine the Stokes I value at lam0Sq_m2 from the Stokes I model
    # Multiply the dirty FDF by Ifreq0 to recover the PI in Jy
    freq0_Hz = C / m.sqrt(lam0Sq_m2)
    Ifreq0_mJybm = poly5(fitDict["p"])(freq0_Hz/1e9)
    dirtyFDF *= (Ifreq0_mJybm / 1e3)    # FDF is in Jy 
    
    # Measure the parameters of the dirty FDF
    mDict = measure_FDF_parms(FDF         = dirtyFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = fwhmRMSF,
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = lam0Sq_m2,
                              dQU         = nanmedian(dQUArr_Jy))
    mDict["Ifreq0_mJybm"] = toscalar(Ifreq0_mJybm)
    mDict["polyCoeffs"] =  ",".join([str(x) for x in fitDict["p"]])
    mDict["IfitStat"] = fitDict["fitStatus"]
    mDict["IfitChiSqRed"] = fitDict["chiSqRed"]
    mDict["lam0Sq_m2"] = toscalar(lam0Sq_m2)
    mDict["freq0_Hz"] = toscalar(freq0_Hz)
    mDict["fwhmRMSF"] = toscalar(fwhmRMSF)
    mDict["dQU_Jybm"] = toscalar(nanmedian(dQUArr_Jy))

    # Measure the complexity of the q and u spectra
    mDict["fracPol"] = mDict["ampPeakPIfit_Jybm"]/(Ifreq0_mJybm/1e3)
    mDict.update( measure_qu_complexity(freqArr_Hz = freqArr_Hz,
                                        qArr       = qArr,
                                        uArr       = uArr,
                                        dqArr      = dqArr,
                                        duArr      = duArr,
                                        fracPol    = mDict["fracPol"],
                                        psi0_deg   = mDict["polAngle0Fit_deg"],
                                        RM_radm2   = mDict["phiPeakPIfit_rm2"],
                                        doPlots    = showPlots,
                                        debug      = debug) )
    
    # Save the  dirty FDF, RMSF and weight array to ASCII files
    print "Saving the dirty FDF, RMSF weight arrays to ASCII files."
    outFile = prefixOut + "_FDFdirty.dat"
    print "> %s" % outFile
    np.savetxt(outFile, zip(phiArr_radm2, dirtyFDF.real, dirtyFDF.imag))
    outFile = prefixOut + "_RMSF.dat"
    print "> %s" % outFile
    np.savetxt(outFile, zip(phi2Arr_radm2, RMSFArr.real, RMSFArr.imag))
    outFile = prefixOut + "_weight.dat"
    print "> %s" % outFile
    np.savetxt(outFile, zip(freqArr_Hz, weightArr))

    # Save the measurements to a "key=value" text file
    print "Saving the measurements on the FDF in 'key=val' and JSON formats."
    outFile = prefixOut + "_RMsynth.dat"
    print "> %s" % outFile
    FH = open(outFile, "w")
    for k, v in mDict.iteritems():
        FH.write("%s=%s\n" % (k, v))
    FH.close()
    outFile = prefixOut + "_RMsynth.json"
    print "> %s" % outFile
    json.dump(dict(mDict), open(outFile, "w"))

    # Print the results to the screen
    print
    print '-'*80
    print 'RESULTS:\n'
    print 'Pol Angle = %.4g (+/-%3g) deg' % (mDict["polAngleFit_deg"],
                                                 mDict["dPolAngleFit_deg"])
    print 'Pol Angle 0 = %.4g (+/-%3g) deg' % (mDict["polAngle0Fit_deg"],
                                                 mDict["dPolAngle0Fit_deg"])
    print 'Peak FD = %.4g (+/-%3g) rad/m^2' % (mDict["phiPeakPIfit_rm2"],
                                                   mDict["dPhiPeakPIfit_rm2"])
    print 'freq0_GHz = %.4g ' % (mDict["freq0_Hz"]/1e9)
    print 'I freq0 = %.4g mJy/beam' % (mDict["Ifreq0_mJybm"])
    print 'Peak PI = %.4g (+/-%3g) mJy/beam' % (mDict["ampPeakPIfit_Jybm"]*1e3,
                                               mDict["dAmpPeakPIfit_Jybm"]*1e3)
    print 'RMS Noise = %.4g mJy/beam' % (mDict["dQU_Jybm"]*1e3)
    
    print 'SNR = %.4g ' % (mDict["snrPIfit"])
    
    print
    print '-'*80

    # Plot the RM Spread Function and dirty FDF
    if showPlots:
        fdfFig = plt.figure(figsize=(12.0, 8))
        plot_rmsf_fdf_fig(phiArr     = phiArr_radm2,
                          FDF        = dirtyFDF,
                          phi2Arr    = phi2Arr_radm2,
                          RMSFArr    = RMSFArr,
                          fwhmRMSF   = fwhmRMSF,
                          vLine      = mDict["phiPeakPIfit_rm2"],
                          fig        = fdfFig)

        # Use the custom navigation toolbar
        try:
            fdfFig.canvas.toolbar.pack_forget()
            CustomNavbar(fdfFig.canvas, fdfFig.canvas.toolbar.window)
        except Exception:
            pass

        # Display the figure
        fdfFig.show()
        print "Press <RETURN> to exit ...",
        raw_input()


#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
