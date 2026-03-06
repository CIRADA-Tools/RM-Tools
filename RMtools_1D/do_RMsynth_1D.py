#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     do_RMsynth_1D.py                                                  #
#                                                                             #
# PURPOSE: API for runnning RM-synthesis on an ASCII Stokes I, Q & U spectrum.#
#                                                                             #
# MODIFIED: 16-Nov-2018 by J. West                                            #
# MODIFIED: 23-October-2019 by A. Thomson                                     #
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

import json
import math as m
import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c as speed_of_light
from scipy.interpolate import interp1d

from RMutils.util_misc import (
    calculate_StokesI_model,
    create_frac_spectra,
    nanmedian,
    renormalize_StokesI_model,
    toscalar,
)
from RMutils.util_plotTk import (
    plot_complexity_fig,
    plot_Ipqu_spectra_fig,
    plot_rmsf_fdf_fig,
)
from RMutils.util_RM import (
    do_rmsynth_planes,
    get_rmsf_planes,
    measure_FDF_parms,
    measure_qu_complexity,
)

if sys.version_info.major == 2:
    print("RM-tools will no longer run with Python 2! Please use Python 3.")
    exit()


# -----------------------------------------------------------------------------#
def run_rmsynth(
    data,
    polyOrd=2,
    phiMax_radm2=None,
    dPhi_radm2=None,
    nSamples=10.0,
    weightType="variance",
    fitRMSF=False,
    noStokesI=False,
    modStokesI=None,
    phiNoise_radm2=1e6,
    nBits=32,
    showPlots=False,
    debug=False,
    verbose=False,
    log=print,
    units="Jy/beam",
    prefixOut="prefixOut",
    saveFigures=None,
    fit_function="log",
    super_resolution=False,
):
    """Run RM synthesis on 1D data.

    Args:
        data (list): Contains frequency and polarization data as either:
            [freq_Hz, I, Q, U, dI, dQ, dU]
                freq_Hz (array_like): Frequency of each channel in Hz.
                I (array_like): Stokes I intensity in each channel.
                Q (array_like): Stokes Q intensity in each channel.
                U (array_like): Stokes U intensity in each channel.
                dI (array_like): Error in Stokes I intensity in each channel.
                dQ (array_like): Error in Stokes Q intensity in each channel.
                dU (array_like): Error in Stokes U intensity in each channel.
            or
            [freq_Hz, q, u,  dq, du]
                freq_Hz (array_like): Frequency of each channel in Hz.
                q (array_like): Fractional Stokes Q intensity (Q/I) in each channel.
                u (array_like): Fractional Stokes U intensity (U/I) in each channel.
                dq (array_like): Error in fractional Stokes Q intensity in each channel.
                du (array_like): Error in fractional Stokes U intensity in each channel.

    Kwargs:
        polyOrd (int): Order of polynomial to fit to Stokes I spectrum.
        phiMax_radm2 (float): Maximum absolute Faraday depth (rad/m^2).
        dPhi_radm2 (float): Faraday depth channel size (rad/m^2).
        nSamples (float): Number of samples across the RMSF.
        weightType (str): Can be "variance" or "uniform"
            "variance" -- Weight by uncertainty in Q and U.
            "uniform" -- Weight uniformly (i.e. with 1s)
        fitRMSF (bool): Fit a Gaussian to the RMSF?
        noStokesI (bool: Is Stokes I data provided?
        modStokesI (array_like): Stokes I model across for each channel (optional)
        phiNoise_radm2 (float): ????
        nBits (int): Precision of floating point numbers.
        showPlots (bool): Show plots?
        debug (bool): Turn on debugging messages & plots?
        verbose (bool): Verbosity.
        log (function): Which logging function to use.
        units (str): Units of data.

    Returns:
        mDict (dict): Summary of RM synthesis results.
        aDict (dict): Data output by RM synthesis.

    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    # freq_Hz, I, Q, U, dI, dQ, dU
    try:
        if verbose:
            log("> Trying [freq_Hz, I, Q, U, dI, dQ, dU]", end=" ")
        (freqArr_Hz, IArr, QArr, UArr, dIArr, dQArr, dUArr) = data
        if verbose:
            log("... success.")
    except Exception:
        if verbose:
            log("...failed.")
        # freq_Hz, q, u, dq, du
        try:
            if verbose:
                log("> Trying [freq_Hz, q, u,  dq, du]", end=" ")
            (freqArr_Hz, QArr, UArr, dQArr, dUArr) = data
            if verbose:
                log("... success.")
            noStokesI = True
        except Exception:
            if verbose:
                log("...failed.")
            if debug:
                log(traceback.format_exc())
            sys.exit()
    if verbose:
        log("Successfully read in the Stokes spectra.")

    # If no Stokes I present, create a dummy spectrum = unity
    if noStokesI:
        if verbose:
            log("Warn: no Stokes I data in use.")
        IArr = np.ones_like(QArr)
        dIArr = np.zeros_like(QArr)

    # Convert to GHz for convenience
    freqArr_GHz = freqArr_Hz / 1e9
    dQUArr = (dQArr + dUArr) / 2.0

    # Fit the Stokes I spectrum and create the fractional spectra
    IModArr, qArr, uArr, dqArr, duArr, fit_result = create_frac_spectra(
        freqArr=freqArr_Hz,
        IArr=IArr,
        QArr=QArr,
        UArr=UArr,
        dIArr=dIArr,
        dQArr=dQArr,
        dUArr=dUArr,
        polyOrd=polyOrd,
        verbose=True,
        debug=debug,
        fit_function=fit_function,
        modStokesI=modStokesI,
    )

    dquArr = np.abs(dqArr + duArr) / 2.0
    dquArr = np.where(np.isfinite(dquArr), dquArr, np.nan)

    # Plot the data and the Stokes I model fit
    if verbose:
        log("Plotting the input data and spectral index fit.")
    freqHirArr_Hz = np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)
    if modStokesI is None:
        IModHirArr = calculate_StokesI_model(fit_result, freqHirArr_Hz)
    elif modStokesI is not None:
        modStokesI_interp = interp1d(freqArr_Hz, modStokesI)
        IModHirArr = modStokesI_interp(freqHirArr_Hz)
    if showPlots or saveFigures:
        specFig = plt.figure(facecolor="w", figsize=(12.0, 8))
        plot_Ipqu_spectra_fig(
            freqArr_Hz=freqArr_Hz,
            IArr=IArr,
            qArr=qArr,
            uArr=uArr,
            dIArr=np.abs(dIArr),
            dqArr=np.abs(dqArr),
            duArr=np.abs(duArr),
            freqHirArr_Hz=freqHirArr_Hz,
            IModArr=IModHirArr,
            fig=specFig,
            units=units,
        )
    if saveFigures:
        outFilePlot = prefixOut + "_spectra-plots.pdf"
        specFig.savefig(outFilePlot, bbox_inches="tight")

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
        rmsFig = plt.figure(facecolor="w", figsize=(12.0, 8))
        ax = rmsFig.add_subplot(111)
        ax.plot(freqArr_GHz, dQUArr, marker="o", color="k", lw=0.5, label="noise <QU>")
        ax.plot(freqArr_GHz, dQArr, marker="o", color="b", lw=0.5, label="noise Q")
        ax.plot(freqArr_GHz, dUArr, marker="o", color="r", lw=0.5, label="noise U")
        xRange = (np.nanmax(freqArr_Hz) - np.nanmin(freqArr_Hz)) / 1e9
        ax.set_xlim(
            np.min(freqArr_Hz) / 1e9 - xRange * 0.05,
            np.max(freqArr_Hz) / 1e9 + xRange * 0.05,
        )
        ax.set_xlabel(r"$\nu$ (GHz)")
        ax.set_ylabel("RMS " + units)
        ax.set_title("RMS noise in Stokes Q, U and <Q,U> spectra")
    #            rmsFig.show()

    # -------------------------------------------------------------------------#

    # Calculate some wavelength parameters
    lambdaSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)
    # dFreq_Hz = np.nanmin(np.abs(np.diff(freqArr_Hz)))
    lambdaSqRange_m2 = np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2)
    # dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
    dLambdaSqMax_m2 = np.nanmax(np.abs(np.diff(lambdaSqArr_m2)))

    # Set the Faraday depth range
    if not super_resolution:
        fwhmRMSF_radm2 = 3.8 / lambdaSqRange_m2  # Dickey+2019 theoretical RMSF width
    else:  # If super resolution, use R&C23 theoretical width
        fwhmRMSF_radm2 = 2.0 / (np.nanmax(lambdaSqArr_m2) + np.nanmin(lambdaSqArr_m2))
    if dPhi_radm2 is None:
        dPhi_radm2 = fwhmRMSF_radm2 / nSamples
    if phiMax_radm2 is None:
        phiMax_radm2 = m.sqrt(3.0) / dLambdaSqMax_m2
        phiMax_radm2 = max(
            phiMax_radm2, fwhmRMSF_radm2 * 10.0
        )  # Force the minimum phiMax to 10 FWHM

    # Faraday depth sampling. Zero always centred on middle channel
    nChanRM = int(round(abs((phiMax_radm2 - 0.0) / dPhi_radm2)) * 2.0 + 1.0)
    startPhi_radm2 = -(nChanRM - 1.0) * dPhi_radm2 / 2.0
    stopPhi_radm2 = +(nChanRM - 1.0) * dPhi_radm2 / 2.0
    phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, nChanRM)
    phiArr_radm2 = phiArr_radm2.astype(dtFloat)
    if verbose:
        log(
            "PhiArr = %.2f to %.2f by %.2f (%d chans)."
            % (phiArr_radm2[0], phiArr_radm2[-1], float(dPhi_radm2), nChanRM)
        )

    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weightType == "variance":
        weightArr = 1.0 / np.power(dquArr, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    if verbose:
        log("Weight type is '%s'." % weightType)

    startTime = time.time()

    # Perform RM-synthesis on the spectrum
    dirtyFDF, lam0Sq_m2 = do_rmsynth_planes(
        dataQ=qArr,
        dataU=uArr,
        lambdaSqArr_m2=lambdaSqArr_m2,
        phiArr_radm2=phiArr_radm2,
        weightArr=weightArr,
        nBits=nBits,
        log=log,
        lam0Sq_m2=0 if super_resolution else None,
    )

    # Calculate the Rotation Measure Spread Function
    RMSFArr, phi2Arr_radm2, fwhmRMSFArr, fitStatArr, _ = get_rmsf_planes(
        lambdaSqArr_m2=lambdaSqArr_m2,
        phiArr_radm2=phiArr_radm2,
        weightArr=weightArr,
        mskArr=~np.isfinite(qArr),
        lam0Sq_m2=lam0Sq_m2,
        double=True,
        fitRMSF=fitRMSF or super_resolution,
        fitRMSFreal=super_resolution,
        nBits=nBits,
        verbose=verbose,
        log=log,
    )
    fwhmRMSF = float(fwhmRMSFArr)

    # ALTERNATE RM-SYNTHESIS CODE --------------------------------------------#

    # dirtyFDF, [phi2Arr_radm2, RMSFArr], lam0Sq_m2, fwhmRMSF = \
    #          do_rmsynth(qArr, uArr, lambdaSqArr_m2, phiArr_radm2, weightArr)

    # -------------------------------------------------------------------------#

    endTime = time.time()
    cputime = endTime - startTime
    if verbose:
        log("> RM-synthesis completed in %.2f seconds." % cputime)

    # Convert Stokes I model to polarization reference frequency. If lambda^2_0 is
    # non-zero, use that as polarization reference frequency and adapt Stokes I model.
    # If lambda^2_0 is zero, make polarization reference frequency equal to
    # Stokes I reference frequency.

    if lam0Sq_m2 == 0:  # Rudnick-Cotton adapatation
        freq0_Hz = fit_result.reference_frequency_Hz
    else:  # standard RM-synthesis
        freq0_Hz = speed_of_light.value / m.sqrt(lam0Sq_m2)
        if modStokesI is None:
            fit_result = renormalize_StokesI_model(fit_result, freq0_Hz)
        else:
            fit_result = fit_result.with_options(reference_frequency_Hz=freq0_Hz)

    # Set Ifreq0 (Stokes I at reference frequency) from either supplied model
    # (interpolated as required) or fit model, as appropriate.
    # Multiply the dirty FDF by Ifreq0 to recover the PI
    if modStokesI is None:
        Ifreq0 = calculate_StokesI_model(fit_result, freq0_Hz)
    elif modStokesI is not None:
        modStokesI_interp = interp1d(freqArr_Hz, modStokesI)
        Ifreq0 = modStokesI_interp(freq0_Hz)
    dirtyFDF *= Ifreq0  # FDF is in fracpol units initially, convert back to flux

    # Calculate the theoretical noise in the FDF !!Old formula only works for wariance weights!
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)
    dFDFth = np.abs(Ifreq0) * np.sqrt(
        np.nansum(weightArr**2 * np.nan_to_num(dquArr) ** 2) / (np.sum(weightArr)) ** 2
    )

    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties
    mDict = measure_FDF_parms(
        FDF=dirtyFDF,
        phiArr=phiArr_radm2,
        fwhmRMSF=fwhmRMSF,
        dFDF=dFDFth,
        lamSqArr_m2=lambdaSqArr_m2,
        lam0Sq=lam0Sq_m2,
    )
    mDict["Ifreq0"] = toscalar(Ifreq0)
    mDict["polyCoeffs"] = ",".join(
        [str(x.astype(np.float32)) for x in fit_result.params]
    )
    mDict["polyCoefferr"] = ",".join(
        [str(x.astype(np.float32)) for x in fit_result.perror]
    )
    mDict["polyOrd"] = fit_result.polyOrd
    mDict["IfitStat"] = fit_result.fitStatus
    mDict["IfitChiSqRed"] = fit_result.chiSqRed
    mDict["fit_function"] = fit_function
    mDict["lam0Sq_m2"] = toscalar(lam0Sq_m2)
    mDict["freq0_Hz"] = toscalar(freq0_Hz)
    mDict["fwhmRMSF"] = toscalar(fwhmRMSF)
    mDict["dQU"] = toscalar(nanmedian(dQUArr))
    mDict["dFDFth"] = toscalar(dFDFth)
    mDict["units"] = units

    if (fit_result.fitStatus >= 128) and verbose:
        log("WARNING: Stokes I model contains negative values!")
    elif (fit_result.fitStatus >= 64) and verbose:
        log("Caution: Stokes I model has low signal-to-noise.")

    # Add information on nature of channels:
    good_channels = np.where(np.logical_and(weightArr != 0, np.isfinite(qArr)))[0]
    mDict["min_freq"] = float(np.min(freqArr_Hz[good_channels]))
    mDict["max_freq"] = float(np.max(freqArr_Hz[good_channels]))
    mDict["N_channels"] = good_channels.size
    mDict["median_channel_width"] = float(np.median(np.diff(freqArr_Hz)))

    # Measure the complexity of the q and u spectra
    # Use 'ampPeakPIfitEff' for bias correct PI
    mDict["fracPol"] = toscalar(mDict["ampPeakPIfitEff"] / (Ifreq0))
    mD, pD = measure_qu_complexity(
        freqArr_Hz=freqArr_Hz,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        fracPol=mDict["fracPol"],
        psi0_deg=mDict["polAngle0Fit_deg"],
        RM_radm2=mDict["phiPeakPIfit_rm2"],
    )
    mDict.update(mD)

    # Debugging plots for spectral complexity measure
    if debug:
        tmpFig = plot_complexity_fig(
            xArr=pD["xArrQ"],
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
            mDict=mDict,
        )
        if saveFigures:
            if verbose:
                print("Saving debug plots:")
            outFilePlot = prefixOut + ".debug-plots.pdf"
            if verbose:
                print("> " + outFilePlot)
            tmpFig.savefig(outFilePlot, bbox_inches="tight")
        else:
            tmpFig.show()

    # add array dictionary
    # Force dtypes as given by input
    aDict = dict()
    aDict["phiArr_radm2"] = phiArr_radm2.astype(dtFloat)
    aDict["phi2Arr_radm2"] = phi2Arr_radm2.astype(dtFloat)
    aDict["RMSFArr"] = RMSFArr.astype(dtComplex)
    aDict["freqArr_Hz"] = freqArr_Hz.astype(dtFloat)
    aDict["weightArr"] = weightArr.astype(dtFloat)
    aDict["dirtyFDF"] = dirtyFDF.astype(dtComplex)

    if verbose:
        # Print the results to the screen
        log()
        log("-" * 80)
        log("RESULTS:\n")
        log("FWHM RMSF = %.4g rad/m^2" % (mDict["fwhmRMSF"]))

        log(
            "Pol Angle = %.4g (+/-%.4g) deg"
            % (mDict["polAngleFit_deg"], mDict["dPolAngleFit_deg"])
        )
        log(
            "Pol Angle 0 = %.4g (+/-%.4g) deg"
            % (mDict["polAngle0Fit_deg"], mDict["dPolAngle0Fit_deg"])
        )
        log(
            "Peak FD = %.4g (+/-%.4g) rad/m^2"
            % (mDict["phiPeakPIfit_rm2"], mDict["dPhiPeakPIfit_rm2"])
        )
        log("freq0_GHz = %.4g " % (mDict["freq0_Hz"] / 1e9))
        log("I freq0 = %.4g %s" % (mDict["Ifreq0"], units))
        log(
            "Peak PI = %.4g (+/-%.4g) %s"
            % (mDict["ampPeakPIfit"], mDict["dAmpPeakPIfit"], units)
        )
        log("QU Noise = %.4g %s" % (mDict["dQU"], units))
        log("FDF Noise (theory)   = %.4g %s" % (mDict["dFDFth"], units))
        log("FDF Noise (Corrected MAD) = %.4g %s" % (mDict["dFDFcorMAD"], units))
        log("FDF SNR = %.4g " % (mDict["snrPIfit"]))
        log(
            "sigma_add (combined) = %.4g (+%.4g, -%.4g)"
            % (mDict["sigmaAddC"], mDict["dSigmaAddPlusC"], mDict["dSigmaAddMinusC"])
        )
        log("Fitted polynomial order = {} ".format(mDict["polyOrd"]))
        log()
        log("-" * 80)

    # Plot the RM Spread Function and dirty FDF
    if showPlots or saveFigures:
        fdfFig = plt.figure(facecolor="w", figsize=(12.0, 8))
        plot_rmsf_fdf_fig(
            phiArr=phiArr_radm2,
            FDF=dirtyFDF,
            phi2Arr=phi2Arr_radm2,
            RMSFArr=RMSFArr,
            fwhmRMSF=fwhmRMSF,
            vLine=mDict["phiPeakPIfit_rm2"],
            fig=fdfFig,
            units=units,
        )

    # Pause if plotting enabled
    if showPlots:
        plt.show()
    if saveFigures:
        if verbose:
            print("Saving RMSF and dirty FDF plot:")
        outFilePlot = prefixOut + "_RMSF-dirtyFDF-plots.pdf"
        if verbose:
            print("> " + outFilePlot)
        fdfFig.savefig(outFilePlot, bbox_inches="tight")
        #        #if verbose: print "Press <RETURN> to exit ...",
    #        input()

    return mDict, aDict


def readFile(dataFile, nBits, verbose=True, debug=False):
    """
    Read the I, Q & U data from the ASCII file.

    Inputs:
        datafile (str): relative or absolute path to file.
        nBits (int): number of bits to store the data as.
        verbose (bool): Print verbose messages to terminal?
        debug (bool): Print full traceback in case of failure?

    Returns:
        data (list of arrays): List containing the columns found in the file.
        If Stokes I is present, this will be [freq_Hz, I, Q, U, dI, dQ, dU],
        else [freq_Hz, q, u,  dq, du].
    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    # Output prefix is derived from the input file name

    # Read the data-file. Format=space-delimited, comments="#".
    if verbose:
        print("Reading the data file '%s':" % dataFile)
    # freq_Hz, I, Q, U, dI, dQ, dU
    try:
        if verbose:
            print("> Trying [freq_Hz, I, Q, U, dI, dQ, dU]", end=" ")
        (freqArr_Hz, IArr, QArr, UArr, dIArr, dQArr, dUArr) = np.loadtxt(
            dataFile, unpack=True, dtype=dtFloat
        )
        if verbose:
            print("... success.")
        data = [freqArr_Hz, IArr, QArr, UArr, dIArr, dQArr, dUArr]
    except Exception:
        if verbose:
            print("...failed.")
        # freq_Hz, q, u, dq, du
        try:
            if verbose:
                print("> Trying [freq_Hz, q, u,  dq, du]", end=" ")
            (freqArr_Hz, QArr, UArr, dQArr, dUArr) = np.loadtxt(
                dataFile, unpack=True, dtype=dtFloat
            )
            if verbose:
                print("... success.")
            data = [freqArr_Hz, QArr, UArr, dQArr, dUArr]

        except Exception:
            if verbose:
                print("...failed.")
            if debug:
                print(traceback.format_exc())
            sys.exit()
    if verbose:
        print("Successfully read in the Stokes spectra.")
    return data


def saveOutput(outdict, arrdict, prefixOut, verbose):
    # Save the  dirty FDF, RMSF and weight array to ASCII files
    if verbose:
        print("Saving the dirty FDF, RMSF weight arrays to ASCII files.")
    outFile = prefixOut + "_FDFdirty.dat"
    if verbose:
        print("> %s" % outFile)
    np.savetxt(
        outFile,
        list(
            zip(
                arrdict["phiArr_radm2"],
                arrdict["dirtyFDF"].real,
                arrdict["dirtyFDF"].imag,
            )
        ),
    )

    outFile = prefixOut + "_RMSF.dat"
    if verbose:
        print("> %s" % outFile)
    np.savetxt(
        outFile,
        list(
            zip(
                arrdict["phi2Arr_radm2"],
                arrdict["RMSFArr"].real,
                arrdict["RMSFArr"].imag,
            )
        ),
    )

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

    for k, v in outdict.items():
        if isinstance(v, (np.float64, float)):
            outdict[k] = float(v)
        elif isinstance(v, (np.int64, int)):
            outdict[k] = int(v)
        elif isinstance(v, np.ndarray):
            outdict[k] = v.tolist()
        elif isinstance(v, (np.bool_, bool)):
            outdict[k] = bool(v)

    json.dump(dict(outdict), open(outFile, "w"))


# -----------------------------------------------------------------------------#
def main():
    import argparse

    """
    Start the function to perform RM-synthesis if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run RM-synthesis on Stokes I, Q and U spectra (1D) stored in an ASCII
    file. The Stokes I spectrum is first fit with a polynomial or power law
    and the resulting model used to create fractional q = Q/I and u = U/I spectra.

    The ASCII file should the following columns, in a space separated format:
    [freq_Hz, I, Q, U, I_err, Q_err, U_err]
    OR
    [freq_Hz, Q, U, Q_err, U_err]


    To get outputs, one or more of the following flags must be set: -S, -p, -v.
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
        metavar="dataFile.dat",
        nargs=1,
        help="ASCII file containing Stokes spectra & errors.",
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
        help="weighting [inverse 'variance'] or 'uniform' (all 1s).",
    )
    parser.add_argument(
        "-f",
        dest="fit_function",
        type=str,
        default="log",
        help="Stokes I fitting function: 'linear' or ['log'] polynomials.",
    )
    parser.add_argument(
        "-o",
        dest="polyOrd",
        type=int,
        default=2,
        help="polynomial order to fit to I spectrum: 0-5 supported, 2 is default.\nSet to negative number to enable dynamic order selection.",
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
        "-S",
        dest="saveOutput",
        action="store_true",
        help="save the arrays and plots [False].",
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
        default="Jy/beam",
        help="Intensity units of the data. [Jy/beam]",
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
    data = readFile(args.dataFile[0], nBits, verbose=verbose, debug=args.debug)

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
        prefixOut=prefixOut,
        saveFigures=args.saveOutput,
        fit_function=args.fit_function,
        super_resolution=args.super_resolution,
    )

    if args.saveOutput:
        saveOutput(mDict, aDict, prefixOut, verbose)


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
