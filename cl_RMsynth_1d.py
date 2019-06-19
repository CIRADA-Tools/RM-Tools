#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     cl_RMsynth_1D.py                                                  #
#                                                                             #
# PURPOSE: API for runnning RM-synthesis on an ASCII Stokes I, Q & U spectrum.#
#                                                                             #
# MODIFIED: 16-Nov-2018 by J. West                                            #
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
#import os
import time
#import argparse
import traceback
import json
import math as m
import numpy as np
import matplotlib.pyplot as plt
#import pdb

from RMutils.util_RM import do_rmsynth
from RMutils.util_RM import do_rmsynth_planes
from RMutils.util_RM import get_rmsf_planes
from RMutils.util_RM import measure_FDF_parms
from RMutils.util_RM import measure_qu_complexity
from RMutils.util_RM import measure_fdf_complexity
from RMutils.util_misc import nanmedian
from RMutils.util_misc import toscalar
from RMutils.util_misc import create_frac_spectra
from RMutils.util_misc import poly5
from RMutils.util_misc import MAD
from RMutils.util_plotTk import plot_Ipqu_spectra_fig
from RMutils.util_plotTk import plot_rmsf_fdf_fig
from RMutils.util_plotTk import plot_complexity_fig
from RMutils.util_plotTk import CustomNavbar
from RMutils.util_plotTk import plot_rmsIQU_vs_nu_ax

C = 2.997924538e8 # Speed of light [m/s]

#-----------------------------------------------------------------------------#
def run_rmsynth(data, polyOrd=3, phiMax_radm2=None, dPhi_radm2=None, 
                nSamples=10.0, weightType="variance", fitRMSF=False,
                noStokesI=False, phiNoise_radm2=1e6, nBits=32, showPlots=False,
                debug=False, verbose=False, log=print):
    """
    Read the I, Q & U data and run RM-synthesis.
    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy
    try:
        if verbose: log("> Trying [freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy]", end=' ')
        (freqArr_Hz, IArr_Jy, QArr_Jy, UArr_Jy, dIArr_Jy, dQArr_Jy, dUArr_Jy) = data 
        if verbose: log("... success.")
    except Exception:
        if verbose: log("...failed.")
        # freq_Hz, q_Jy, u_Jy, dq_Jy, du_Jy
        try:
            if verbose: log("> Trying [freq_Hz, q_Jy, u_Jy,  dq_Jy, du_Jy]", end=' ')
            (freqArr_Hz, QArr_Jy, UArr_Jy, dQArr_Jy, dUArr_Jy) = data 
            if verbose: log("... success.")
            noStokesI = True
        except Exception:
            if verbose: log("...failed.")
            if debug:
                log(traceback.format_exc())
            sys.exit()
    if verbose: log("Successfully read in the Stokes spectra.")

    # If no Stokes I present, create a dummy spectrum = unity
    if noStokesI:
        log("Warn: no Stokes I data in use.")
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
    dQUArr_mJy = (dQArr_mJy + dUArr_mJy)/2.0
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
        if verbose: log("Plotting the input data and spectral index fit.")
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
#        try:
#            specFig.canvas.toolbar.pack_forget()
#            CustomNavbar(specFig.canvas, specFig.canvas.toolbar.window)
#        except Exception:
#            pass

        # Display the figure
#        if not plt.isinteractive():
#            specFig.show()

        # DEBUG (plot the Q, U and average RMS spectrum)
        if debug:
            rmsFig = plt.figure(figsize=(12.0, 8))
            ax = rmsFig.add_subplot(111)
            ax.plot(freqArr_Hz/1e9, dQUArr_mJy, marker='o', color='k', lw=0.5,
                    label='rms <QU>')
            ax.plot(freqArr_Hz/1e9, dQArr_mJy, marker='o', color='b', lw=0.5,
                    label='rms Q')
            ax.plot(freqArr_Hz/1e9, dUArr_mJy, marker='o', color='r', lw=0.5,
                    label='rms U')
            xRange = (np.nanmax(freqArr_Hz)-np.nanmin(freqArr_Hz))/1e9 
            ax.set_xlim( np.min(freqArr_Hz)/1e9 - xRange*0.05,
                         np.max(freqArr_Hz)/1e9 + xRange*0.05)
            ax.set_xlabel('$\\nu$ (GHz)')
            ax.set_ylabel('RMS (mJy bm$^{-1}$)')
            ax.set_title("RMS noise in Stokes Q, U and <Q,U> spectra")
#            rmsFig.show()

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
    nChanRM = int(round(abs((phiMax_radm2 - 0.0) / dPhi_radm2)) * 2.0 + 1.0)
    startPhi_radm2 = - (nChanRM-1.0) * dPhi_radm2 / 2.0
    stopPhi_radm2 = + (nChanRM-1.0) * dPhi_radm2 / 2.0
    phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, nChanRM)
    phiArr_radm2 = phiArr_radm2.astype(dtFloat)
    if verbose: log("PhiArr = %.2f to %.2f by %.2f (%d chans)." % (phiArr_radm2[0],
                                                         phiArr_radm2[-1],
                                                         float(dPhi_radm2),
                                                         nChanRM))
                                                             
    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weightType=="variance":
        weightArr = 1.0 / np.power(dQUArr_mJy, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    if verbose: log("Weight type is '%s'." % weightType)

    startTime = time.time()
    
    # Perform RM-synthesis on the spectrum
    dirtyFDF, lam0Sq_m2 = do_rmsynth_planes(dataQ           = qArr,
                                            dataU           = uArr,
                                            lambdaSqArr_m2  = lambdaSqArr_m2, 
                                            phiArr_radm2    = phiArr_radm2, 
                                            weightArr       = weightArr,
                                            nBits           = nBits,
                                            verbose         = True,
                                            log             = log)

    # Calculate the Rotation Measure Spread Function
    RMSFArr, phi2Arr_radm2, fwhmRMSFArr, fitStatArr = \
        get_rmsf_planes(lambdaSqArr_m2  = lambdaSqArr_m2,
                        phiArr_radm2    = phiArr_radm2, 
                        weightArr       = weightArr, 
                        mskArr          = ~np.isfinite(qArr),
                        lam0Sq_m2       = lam0Sq_m2, 
                        double          = True, 
                        fitRMSF         = fitRMSF, 
                        fitRMSFreal     = False, 
                        nBits           = nBits,
                        verbose         = True,
                        log             = log)
    fwhmRMSF = float(fwhmRMSFArr)
    
    # ALTERNATE RM-SYNTHESIS CODE --------------------------------------------#

    #dirtyFDF, [phi2Arr_radm2, RMSFArr], lam0Sq_m2, fwhmRMSF = \
    #          do_rmsynth(qArr, uArr, lambdaSqArr_m2, phiArr_radm2, weightArr)
    
    #-------------------------------------------------------------------------#
    
    endTime = time.time()
    cputime = (endTime - startTime)
    if verbose: log("> RM-synthesis completed in %.2f seconds." % cputime)
    
    # Determine the Stokes I value at lam0Sq_m2 from the Stokes I model
    # Multiply the dirty FDF by Ifreq0 to recover the PI in Jy
    freq0_Hz = C / m.sqrt(lam0Sq_m2)
    Ifreq0_mJybm = poly5(fitDict["p"])(freq0_Hz/1e9)
    dirtyFDF *= (Ifreq0_mJybm / 1e3)    # FDF is in Jy 

    # Calculate the theoretical noise in the FDF !!Old formula only works for wariance weights!
    #dFDFth_Jybm = np.sqrt(1./np.sum(1./dQUArr_Jy**2.)) 
    dFDFth_Jybm = np.sqrt( np.sum(weightArr**2 * dQUArr_Jy**2) / (np.sum(weightArr))**2 )
    
    
    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties
    mDict = measure_FDF_parms(FDF         = dirtyFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = fwhmRMSF,
                              dFDF        = dFDFth_Jybm,
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = lam0Sq_m2)
    mDict["Ifreq0_mJybm"] = toscalar(Ifreq0_mJybm)
    mDict["polyCoeffs"] =  ",".join([str(x) for x in fitDict["p"]])
    mDict["IfitStat"] = fitDict["fitStatus"]
    mDict["IfitChiSqRed"] = fitDict["chiSqRed"]
    mDict["lam0Sq_m2"] = toscalar(lam0Sq_m2)
    mDict["freq0_Hz"] = toscalar(freq0_Hz)
    mDict["fwhmRMSF"] = toscalar(fwhmRMSF)
    mDict["dQU_Jybm"] = toscalar(nanmedian(dQUArr_Jy))
    mDict["dFDFth_Jybm"] = toscalar(dFDFth_Jybm)
    if mDict['phiPeakPIfit_rm2'] == None:
        log('Peak is at edge of RM spectrum! Peak fitting failed!\n')
        log('Rerunning with Phi_max twice as large.')
        #The following code re-runs everything with higher phiMax, 
        #Then overwrite the appropriate variables so as to continue on without
        #interuption.
        mDict, aDict = run_rmsynth(data           = data,
                polyOrd        = polyOrd,
                phiMax_radm2   = phiMax_radm2*2,
                dPhi_radm2     = dPhi_radm2,
                nSamples       = nSamples,
                weightType     = weightType,
                fitRMSF        = fitRMSF,
                noStokesI      = noStokesI,
                nBits          = nBits,
                showPlots      = False,
                debug          = debug,
                verbose        = verbose)
        phiArr_radm2=aDict["phiArr_radm2"]
        phi2Arr_radm2=aDict["phi2Arr_radm2"]
        RMSFArr=aDict["RMSFArr"]
        freqArr_Hz=aDict["freqArr_Hz"]
        weightArr=aDict["weightArr"]
        dirtyFDF=aDict["dirtyFDF"]


        
    # Measure the complexity of the q and u spectra
    mDict["fracPol"] = mDict["ampPeakPIfit_Jybm"]/(Ifreq0_mJybm/1e3)
    mD, pD = measure_qu_complexity(freqArr_Hz = freqArr_Hz,
                                   qArr       = qArr,
                                   uArr       = uArr,
                                   dqArr      = dqArr,
                                   duArr      = duArr,
                                   fracPol    = mDict["fracPol"],
                                   psi0_deg   = mDict["polAngle0Fit_deg"],
                                   RM_radm2   = mDict["phiPeakPIfit_rm2"])
    mDict.update(mD)
    
    # Debugging plots for spectral complexity measure
    if debug:
        tmpFig = plot_complexity_fig(xArr=pD["xArrQ"],
                                     qArr=pD["yArrQ"],
                                     dqArr=pD["dyArrQ"],
                                     sigmaAddqArr=pD["sigmaAddArrQ"],
                                     chiSqRedqArr=pD["chiSqRedArrQ"],
                                     probqArr=pD["probArrQ"],
                                     uArr=pD["yArrU"],
                                     duArr=pD["dyArrU"],
                                     sigmaAdduArr=pD["sigmaAddArrU"],
                                     chiSqReduArr=pD["chiSqRedArrU"],
                                     probuArr=pD["probArrU"],
                                     mDict=mDict)
        tmpFig.show()
    
    #add array dictionary
    aDict = dict()
    aDict["phiArr_radm2"] = phiArr_radm2
    aDict["phi2Arr_radm2"] = phi2Arr_radm2
    aDict["RMSFArr"] = RMSFArr
    aDict["freqArr_Hz"] = freqArr_Hz
    aDict["weightArr"]=weightArr
    aDict["dirtyFDF"]=dirtyFDF
    
    if verbose: 
       # Print the results to the screen
       log()
       log('-'*80)
       log('RESULTS:\n')
       log('FWHM RMSF = %.4g rad/m^2' % (mDict["fwhmRMSF"]))
    
       log('Pol Angle = %.4g (+/-%.4g) deg' % (mDict["polAngleFit_deg"],
                                              mDict["dPolAngleFit_deg"]))
       log('Pol Angle 0 = %.4g (+/-%.4g) deg' % (mDict["polAngle0Fit_deg"],
                                                mDict["dPolAngle0Fit_deg"]))
       log('Peak FD = %.4g (+/-%.4g) rad/m^2' % (mDict["phiPeakPIfit_rm2"],
                                                mDict["dPhiPeakPIfit_rm2"]))
       log('freq0_GHz = %.4g ' % (mDict["freq0_Hz"]/1e9))
       log('I freq0 = %.4g mJy/beam' % (mDict["Ifreq0_mJybm"]))
       log('Peak PI = %.4g (+/-%.4g) mJy/beam' % (mDict["ampPeakPIfit_Jybm"]*1e3,
                                                mDict["dAmpPeakPIfit_Jybm"]*1e3))
       log('QU Noise = %.4g mJy/beam' % (mDict["dQU_Jybm"]*1e3))
       log('FDF Noise (theory)   = %.4g mJy/beam' % (mDict["dFDFth_Jybm"]*1e3))
       log('FDF Noise (Corrected MAD) = %.4g mJy/beam' % (mDict["dFDFcorMAD_Jybm"]*1e3))
       log('FDF Noise (rms)   = %.4g mJy/beam' % (mDict["dFDFrms_Jybm"]*1e3))
       log('FDF SNR = %.4g ' % (mDict["snrPIfit"]))
       log('sigma_add(q) = %.4g (+%.4g, -%.4g)' % (mDict["sigmaAddQ"],
                                            mDict["dSigmaAddPlusQ"],
                                            mDict["dSigmaAddMinusQ"]))
       log('sigma_add(u) = %.4g (+%.4g, -%.4g)' % (mDict["sigmaAddU"],
                                            mDict["dSigmaAddPlusU"],
                                            mDict["dSigmaAddMinusU"]))
       log()
       log('-'*80)



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
#        try:
#            fdfFig.canvas.toolbar.pack_forget()
#            CustomNavbar(fdfFig.canvas, fdfFig.canvas.toolbar.window)
#        except Exception:
#            pass
        
        # Display the figure
#        fdfFig.show()

    # Pause if plotting enabled
    if showPlots or debug:        
        plt.show()
        #        #if verbose: print "Press <RETURN> to exit ...",
#        input()

    return mDict, aDict
    
def readFile(dataFile, nBits, verbose):
    """
    Read the I, Q & U data from the ASCII file.
    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Output prefix is derived from the input file name
    

    # Read the data-file. Format=space-delimited, comments="#".
    if verbose: print("Reading the data file '%s':" % dataFile)
    # freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy
    try:
        if verbose: print("> Trying [freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy]", end=' ')
        (freqArr_Hz, IArr_Jy, QArr_Jy, UArr_Jy,
         dIArr_Jy, dQArr_Jy, dUArr_Jy) = \
         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
        if verbose: print("... success.")
        data=[freqArr_Hz, IArr_Jy, QArr_Jy, UArr_Jy, dIArr_Jy, dQArr_Jy, dUArr_Jy]
    except Exception:
        if verbose: print("...failed.")
        # freq_Hz, q_Jy, u_Jy, dq_Jy, du_Jy
        try:
            if verbose: print("> Trying [freq_Hz, q_Jy, u_Jy,  dq_Jy, du_Jy]", end=' ')
            (freqArr_Hz, QArr_Jy, UArr_Jy, dQArr_Jy, dUArr_Jy) = \
                         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
            if verbose: print("... success.")
            data=[freqArr_Hz, QArr_Jy, UArr_Jy, dQArr_Jy, dUArr_Jy]

            noStokesI = True
        except Exception:
            if verbose: print("...failed.")
            if debug:
                print(traceback.format_exc())
            sys.exit()
    if verbose: print("Successfully read in the Stokes spectra.")
    return data

def saveOutput(outdict, arrdict, prefixOut, verbose):
    # Save the  dirty FDF, RMSF and weight array to ASCII files
    if verbose: print("Saving the dirty FDF, RMSF weight arrays to ASCII files.")
    outFile = prefixOut + "_FDFdirty.dat"
    if verbose: 
        print("> %s" % outFile)
    np.savetxt(outFile, list(zip(arrdict["phiArr_radm2"], arrdict["dirtyFDF"].real, arrdict["dirtyFDF"].imag)))
    
    outFile = prefixOut + "_RMSF.dat"
    if verbose: 
        print("> %s" % outFile)       
    np.savetxt(outFile, list(zip(arrdict["phi2Arr_radm2"], arrdict["RMSFArr"].real, arrdict["RMSFArr"].imag)))
    
    outFile = prefixOut + "_weight.dat"
    if verbose: 
        print("> %s" % outFile)
    np.savetxt(outFile, list(zip(arrdict["freqArr_Hz"], arrdict["weightArr"])))

    # Save the measurements to a "key=value" text file
    outFile = prefixOut + "_RMsynth.dat"

    if verbose: 
        print("Saving the measurements on the FDF in 'key=val' and JSON formats.")
        print("> %s" % outFile)

    FH = open(outFile, "w")
    for k, v in outdict.items():
        FH.write("%s=%s\n" % (k, v))
    FH.close()
       

    outFile = prefixOut + "_RMsynth.json"
    
    if verbose: 
        print("> %s" % outFile)
    json.dump(dict(outdict), open(outFile, "w"))       



