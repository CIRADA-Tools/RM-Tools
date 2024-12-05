#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     rmtools_bwdepol.py                                                #
#                                                                             #
# PURPOSE: Algorithm for finding polarized sources while accounting for       #
#          bandwidth depolarization.                                          #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2020 Canadian Initiative for Radio Astronomy Data Analysis     #                                                                             #
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

import argparse
import math as m
import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.constants import c as speed_of_light
from matplotlib.ticker import MaxNLocator

from RMtools_1D.do_RMsynth_1D import saveOutput
from RMutils.util_misc import (
    MAD,
    create_frac_spectra,
    nanmedian,
    poly5,
    progress,
    toscalar,
)
from RMutils.util_plotTk import (
    plot_complexity_fig,
    plot_dirtyFDF_ax,
    plot_Ipqu_spectra_fig,
)
from RMutils.util_RM import (
    calc_parabola_vertex,
    extrap,
    fit_rmsf,
    measure_qu_complexity,
)

if sys.version_info.major == 2:
    print("RM-tools will no longer run with Python 2! Please use Python 3.")
    exit()


# -----------------------------------------------------------------------------#
def rotation_integral_limit(freq, phi):
    """Calculates the analytic solution to the channel polarization integral
    at one limit frequency.

    Parameters
    ----------
    freq: float
          frequency in Hz

    phi: float
          Faraday depth (rad/m^2)

    Returns
    -------
    intergral_lim: complex
                   intergral limit

    """
    funct1 = freq * np.exp(2.0j * phi * ((speed_of_light.value / freq) ** 2))
    funct2 = speed_of_light.value * np.sqrt((np.abs(phi) * np.pi))
    funct3 = -1.0j + np.sign(phi)
    funct4 = sp.special.erf(
        np.sqrt(np.abs(phi)) * (speed_of_light.value / freq) * (-1.0j + np.sign(phi))
    )

    intergral_lim = funct1 + (funct2 * funct3 * funct4)

    return intergral_lim


def rotation_operator(channel_width, channel_center, phi):
    """Rotation operator for channel with top-hat-in-frequency sensitivity.
    Computes the net effect on a polarization vector for a channel of given
    center frequnecy and bandwidth, for a given RM.

    Parameters
    ----------
    channel_width: float
                   channel bandwidth in Hz

    channel_center: float
                    channel frequency in Hz

    phi: float
         channel frequency in hz

    Returns
    -------
    (complex) rotation operator for that channel

    """
    b = channel_center + 0.5 * channel_width
    a = channel_center - 0.5 * channel_width

    int_a = rotation_integral_limit(a, phi)
    int_b = rotation_integral_limit(b, phi)

    return (1 / channel_width) * (int_b - int_a)


def estimate_channel_bandwidth(freq_array):
    """Estimates the bandwidth per channel given the spacing between channel
    centers. Only looks at the first 2 channels.

    Parameters
    ----------
    freq_array: array-like
                array of channel centers

    Returns
    -------
    ban: float
         seperation between first two channel centers

    """
    ban = freq_array[1] - freq_array[0]
    return ban


def l2_to_freq_array(lambda_square_array):
    """returns the freqency array, corresponding to a lambda square array"""

    f = speed_of_light.value**2 / lambda_square_array
    return np.sqrt(f)


def adjoint_theory(adjoint_vars, dQUArr, show_progress=False, log=print):
    """Calculates the theoretical sensitivity and noise for the adjoint method

    Parameters
    ----------
    adjoint_vars: list
                  list like object containing organized as
                  [widths_Hz, freqArr_Hz, phiArr_radm2, K, weightArr]

    dQUArr: array like
            array containing the error in Stokes Q, and U

    show_progress: Boolean
                   If set to True, shows progress, Default is False

    log: function
         logging function, default is print

    Returns
    -------
    adjoint_info: list
                  list containing phiArr, and the theoretical noise and
                  sensitivity organized as
                  [phiArr_radm2, adjoint_sens, adjoint_noise]

    """

    widths_Hz, freqArr_Hz, phiArr_radm2, K, weightArr = adjoint_vars
    adjoint_noise = np.ones(len(phiArr_radm2))
    adjoint_sens = np.ones(len(phiArr_radm2))

    nPhi = len(phiArr_radm2)
    if show_progress:
        log("Calculating Theoretical Sensitivity & Noise")
        progress(40, 0)
    for i in range(nPhi):
        if show_progress:
            progress(40, ((i + 1) * 100.0 / nPhi))

        r_i = rotation_operator(widths_Hz, freqArr_Hz, phiArr_radm2[i])
        adjoint_noise2 = (
            np.sum((weightArr * dQUArr) ** 2 * np.abs(r_i) ** 2)
            / np.sum(weightArr) ** 2
        )  # equation 34
        adjoint_noise[i] = np.sqrt(adjoint_noise2)

        adjoint_sens[i] = K * np.sum(weightArr * (np.abs(r_i) ** 2))

    adjoint_info = [phiArr_radm2, adjoint_sens, adjoint_noise]
    return adjoint_info


def plot_adjoint_info(mylist, units="Jy/beam"):
    """plots theoretical noise, sensitivity"""
    fig, ax = plt.subplots(2, dpi=100, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    [phiArr_radm2, adjoint_sens, adjoint_noise] = mylist

    ax[1].plot(
        phiArr_radm2,
        adjoint_sens / adjoint_noise * np.max(adjoint_noise),
    )
    ax[1].set_xlabel(r"$\phi$ (rad m$^{-2}$)")
    ax[1].set_ylabel("S:N multiplier")
    ax[1].set_title("Theoretical S:N after bandwidth depolarization")
    # plot 2
    ax[0].plot(
        phiArr_radm2,
        adjoint_sens,
    )
    ax[0].set_xlabel(r"$\phi$ (rad m$^{-2}$)")
    ax[0].set_ylabel("Sensitivity")
    ax[0].set_title("Theoretical Sensitivity after bandwidth depolarization")
    return


# -----------------------------------------------------------------------------#


def analytical_chan_pol(f, ban, phi, xi_knot=0, p=1):
    """Calculates the average analytic solution to the channel polarization
    integral per channel

    Based on equation 13 of Schnitzeler & Lee (2015)

    Parameters
    ----------
    f: float
       channel center frequency in Hz

    ban: float
         channel bandwidth in Hz

    phi: float
         Faraday depth value in rad/m^2

    xi_knot: float
             inital polarization angle in radians

    p: float
       polarzied intensity

    Returns
    -------
    avg_p_tilda: complex
                 the average complex polarization, for the bandwidth,
                 real is Q, imaginary is U

    """
    a = f - (ban / 2)
    b = f + (ban / 2)  # integral start and stop values

    ya = rotation_integral_limit(
        a,
        phi,
    )
    yb = rotation_integral_limit(
        b,
        phi,
    )  # check orig for xi_knot

    i = p * (yb - ya)
    avg_p_tilda = i / ban

    return avg_p_tilda


def bwdepol_simulation(peak_rm, freqArr_Hz, widths_Hz):
    """Farday thin simulated source of the same RM as the measured source,
    with unit intensity

    Parameters
    ----------
    peak_rm: float
             peak in Faraday depth value (in rad/m^2) for sim

    freqArr_hz: array like
                frequency array (in Hz)

    width_Hz: float
              channel width in Hz

    Returns
    -------
    data:
         Dirty FDF for the simulated data formated as a list of arrays
         [freq_Hz, q, u,  dq, du]

    """
    if widths_Hz == None:
        widths_Hz = estimate_channel_bandwidth(freqArr_Hz)

    p_tilda = analytical_chan_pol(freqArr_Hz, widths_Hz, peak_rm)
    size_f = len(freqArr_Hz)
    dq = np.ones(size_f)
    du = np.ones(size_f)

    # format  = [freq_Hz, q, u,  dq, du]
    data = [freqArr_Hz, np.real(p_tilda), np.imag(p_tilda), dq, du]
    return data


# -----------------------------------------------------------------------------#
def bwdepol_tweakAxFormat(
    ax,
    pad=10,
    loc="upper right",
    linewidth=1,
    ncol=1,
    bbox_to_anchor=(1.00, 1.00),
    showLeg=True,
):
    """Tweaks some default plotting parameters for the RMSF, returns ax"""
    # Axis/tic formatting
    ax.tick_params(pad=pad)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markeredgewidth(linewidth)

    # Legend formatting
    if showLeg:
        leg = ax.legend(
            numpoints=1,
            loc=loc,
            shadow=False,
            borderaxespad=0.3,
            ncol=ncol,
            bbox_to_anchor=bbox_to_anchor,
        )
        for t in leg.get_texts():
            t.set_fontsize("small")
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_alpha(0.5)

    return ax


def gauss(p, peak_rm):
    """Return a fucntion to evaluate a Gaussian with parameters
    off set by peak_rm

    Parameters
    ----------
    p: list
       parameters for Gaussian, p = [ampplitude, mean, FWHM]

    peak_rm: float
             peak in Faraday depth (in rad/m^2), used to center Gaussian

    Returns
    -------
    rfun: fuction
          Gaussian with specified parameters, off set my peak_rm

    """

    a, b, w = p
    gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
    s = w / gfactor

    def rfunc(x):
        y = a * np.exp(-((x - b - peak_rm) ** 2.0) / (2.0 * s**2.0))
        return y

    return rfunc


def bwdepol_plot_RMSF_ax(
    ax,
    phiArr,
    RMSFArr,
    peak_rm,
    fwhmRMSF=None,
    axisYright=False,
    axisXtop=False,
    doTitle=False,
):
    """Modified for bwdepol, Plots each ax for the RMSF plotting"""

    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot the RMSF
    ax.step(phiArr, RMSFArr.real, where="mid", color="tab:blue", lw=0.5, label="Real")
    ax.step(
        phiArr, RMSFArr.imag, where="mid", color="tab:red", lw=0.5, label="Imaginary"
    )
    ax.step(phiArr, np.abs(RMSFArr), where="mid", color="k", lw=1.0, label="PI")
    ax.axhline(0, color="grey")
    if doTitle:
        ax.text(0.05, 0.84, "RMSF", transform=ax.transAxes)

    # Plot the Gaussian fit
    if fwhmRMSF is not None:
        yGauss = np.max(np.abs(RMSFArr)) * gauss([1.0, 0.0, fwhmRMSF], peak_rm)(phiArr)
        ax.plot(
            phiArr,
            yGauss,
            color="g",
            marker="None",
            mfc="w",
            mec="g",
            ms=10,
            label="Gaussian",
            lw=2.0,
            ls="--",
        )

    # Scaling
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(phiArr) - np.nanmin(phiArr)
    ax.set_xlim(np.nanmin(phiArr) - xRange * 0.01, np.nanmax(phiArr) + xRange * 0.01)
    ax.set_ylabel("Normalised Units")
    ax.set_xlabel(r"$\phi$ rad m$^{-2}$")
    ax.axhline(0, color="grey")

    # Format tweaks
    ax = bwdepol_tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


def bwdepol_plot_rmsf_fdf_fig(
    phiArr,
    FDF,
    phi2Arr,
    RMSFArr,
    peak_rm,
    fwhmRMSF=None,
    gaussParm=[],
    vLine=None,
    fig=None,
    units="flux units",
):
    """Modified for bwdepol, Plot the RMSF and FDF on a single figure"""

    # Default to a pyplot figure
    if fig == None:
        fig = plt.figure(figsize=(12.0, 8))
    # Plot the RMSF
    ax1 = fig.add_subplot(211)
    bwdepol_plot_RMSF_ax(
        ax=ax1,
        phiArr=phi2Arr,
        RMSFArr=RMSFArr,
        peak_rm=peak_rm,
        fwhmRMSF=fwhmRMSF,
        doTitle=True,
    )
    [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax1.set_xlabel("")

    ax2 = fig.add_subplot(212, sharex=ax1)
    plot_dirtyFDF_ax(
        ax=ax2,
        phiArr=phiArr,
        FDFArr=FDF,
        gaussParm=gaussParm,
        vLine=vLine,
        doTitle=True,
        units=units,
    )
    return fig


# -----------------------------------------------------------------------------#
# modified for adjoint
def bwdepol_get_rmsf_planes(
    freqArr_Hz,
    widths_Hz,
    phiArr_radm2,
    peak_rm,
    weightArr=None,
    mskArr=None,
    lam0Sq_m2=None,
    double=True,
    fitRMSF=True,
    fitRMSFreal=False,
    nBits=64,
    verbose=False,
    log=print,
):
    """Calculate the Rotation Measure Spread Function from inputs. This version
    returns a cube (1, 2 or 3D) of RMSF spectra based on the shape of a
    boolean mask array, where flagged data are True and unflagged data False.
    If only whole planes (wavelength channels) are flagged then the RMSF is the
    same for all pixels and the calculation is done once and replicated to the
    dimensions of the mask. If some isolated voxels are flagged then the RMSF
    is calculated by looping through each wavelength plane, which can take some
    time. By default the routine returns the analytical width of the RMSF main
    lobe but can also use MPFIT to fit a Gaussian.

    This has been modified from the convientual RMtools_1D version for bwdepol


    Parameters
    ----------
    freqArr_Hz      ... vector of frequency values
    phiArr_radm2    ... vector of trial Faraday depth values
    weightArr       ... vector of weights, default [None] is no weighting
    maskArr         ... cube of mask values used to shape return cube [None]
    lam0Sq_m2       ... force a reference lambda^2 value (def=calculate) [None]
    double          ... pad the Faraday depth to double-size [True]
    fitRMSF         ... fit the main lobe of the RMSF with a Gaussian [False]
    fitRMSFreal     ... fit RMSF.real, rather than abs(RMSF) [False]
    nBits           ... precision of data arrays [64]
    verbose         ... print feedback during calculation [False]
    log             ... function to be used to output messages [print]


    Returns
    -------
    RMSFcube: array
              a cube (1, 2 or 3D) of RMSF spectra based on the shape of a
              boolean mask array

    phi2Arr: array
             array of the Faraday Depth (in rad/m^2)

    fwhmRMSFArr: array
                 fwhm of the RMSF

    statArr: array

    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    # For cleaning the RMSF should extend by 1/2 on each side in phi-space
    if double:
        nPhi = phiArr_radm2.shape[0]
        nExt = np.ceil(nPhi / 2.0)
        resampIndxArr = np.arange(2.0 * nExt + nPhi) - nExt
        phi2Arr = extrap(resampIndxArr, np.arange(nPhi, dtype="int"), phiArr_radm2)
    else:
        phi2Arr = phiArr_radm2

    # Set the weight array
    if weightArr is None:
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)

    # Set the mask array (default to 1D, no masked channels)
    if mskArr is None:
        mskArr = np.zeros_like(freqArr_Hz, dtype="bool")
        nDims = 1
    else:
        mskArr = mskArr.astype("bool")
        nDims = len(mskArr.shape)

    # Sanity checks on array sizes
    if not weightArr.shape == freqArr_Hz.shape:
        log("Err: wavelength^2 and weight arrays must be the same shape.")
        return None, None, None, None
    if not nDims <= 3:
        log("Err: mask dimensions must be <= 3.")
        return None, None, None, None
    if not mskArr.shape[0] == freqArr_Hz.shape[0]:
        log("Err: mask depth does not match lambda^2 vector (%d vs %d).", end=" ")
        (mskArr.shape[0], freqArr_Hz.shape[-1])
        log("     Check that the mask is in [z, y, x] order.")
        return None, None, None, None

    # Reshape the mask array to 3 dimensions
    if nDims == 1:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], 1, 1))
    elif nDims == 2:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], mskArr.shape[1], 1))

    # Initialise the complex RM Spread Function cube
    nX = mskArr.shape[-1]
    nY = mskArr.shape[-2]
    nPix = nX * nY
    nPhi = phi2Arr.shape[0]
    RMSFcube = np.ones((nPhi, nY, nX), dtype=dtComplex)

    # If full planes are flagged then set corresponding weights to zero
    xySum = np.sum(np.sum(mskArr, axis=1), axis=1)
    mskPlanes = np.where(xySum == nPix, 0, 1)
    weightArr *= mskPlanes

    # Check for isolated clumps of flags (# flags in a plane not 0 or nPix)
    flagTotals = np.unique(xySum).tolist()
    try:
        flagTotals.remove(0)
    except Exception:
        pass
    try:
        flagTotals.remove(nPix)
    except Exception:
        pass

    lambdaSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)
    # Calculate the analytical FWHM width of the main lobe
    fwhmRMSF = (
        2.0 * m.sqrt(3.0) / (np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2))
    )

    # Create simulated data set with simRM = peakRM
    RMSF_data = bwdepol_simulation(peak_rm, freqArr_Hz, widths_Hz)
    # RMSFArr = fdf from bwdepol_simulation
    RMSFArr, _, _ = do_adjoint_rmsynth_planes(
        freqArr_Hz,
        RMSF_data[1],
        RMSF_data[2],
        phiArr_radm2,
        widths_Hz=widths_Hz,
        weightArr=weightArr,
        lam0Sq_m2=lam0Sq_m2,
        verbose=verbose,
        log=print,
    )

    # Fit the RMSF main lobe
    fitStatus = -1
    if fitRMSF:
        if verbose:
            log("Fitting Gaussian to the main lobe.")
        mp = fit_rmsf(phi2Arr, np.abs(RMSFArr) / np.max(np.abs(RMSFArr)))
        if mp is None or mp.status < 1:
            pass
            log("Err: failed to fit the RMSF.")
            log("     Defaulting to analytical value.")
        else:
            fwhmRMSF = mp.params[2]
            fitStatus = mp.status

    # Replicate along X and Y axes
    RMSFcube = np.tile(RMSFArr[:, np.newaxis, np.newaxis], (1, nY, nX))
    fwhmRMSFArr = np.ones((nY, nX), dtype=dtFloat) * fwhmRMSF
    statArr = np.ones((nY, nX), dtype="int") * fitStatus

    # Remove redundant dimensions
    fwhmRMSFArr = np.squeeze(fwhmRMSFArr)
    statArr = np.squeeze(statArr)
    RMSFcube = RMSFcube.reshape(-1)

    return RMSFcube, phi2Arr, fwhmRMSFArr, statArr


# -----------------------------------------------------------------------------#
def bwdepol_measure_FDF_parms(
    FDF,
    phiArr,
    fwhmRMSF,
    adjoint_sens,
    adjoint_noise,
    dFDF=None,
    lamSqArr_m2=None,
    lam0Sq=None,
    snrDoBiasCorrect=5.0,
):
    """
    Measure standard parameters from a complex Faraday Dispersion Function.
    Currently this function assumes that the noise levels in the Stokes Q
    and U spectra are the same.
    Returns a dictionary containing measured parameters.

    This has been modified from the convientual RMtools_1D version for bwdepol
    """

    # Determine the peak channel in the FDF, its amplitude and index
    absFDF = np.abs(FDF)
    rm_fdf = (
        absFDF / adjoint_noise
    )  # RM spectrum in S:N units (normalized by RM-dependent noise)
    amp_fdf = (
        absFDF / adjoint_sens
    )  # RM spectrum normalized by (RM-dependent) sensitivity
    indxPeakPIchan = np.nanargmax(rm_fdf[1:-1]) + 1  # Masks out the edge channels

    # new theoretical dFDF correction for adjoint method
    # This is noise in the adjoint-spectrum.
    dFDF = adjoint_noise[indxPeakPIchan]

    # This is the error in the amplitude (accounting for re-normalization)
    dampPeakPI = dFDF / adjoint_sens[indxPeakPIchan]

    # Measure the RMS noise in the spectrum after masking the peak
    # changed all absFDF to rm_fdf
    # Since this is normalized by theoretical noise, it's effectively testing
    # the noise relative to the theoretical noise.
    dPhi = np.nanmin(np.diff(phiArr))
    fwhmRMSF_chan = np.ceil(fwhmRMSF / dPhi)
    iL = int(max(0, indxPeakPIchan - fwhmRMSF_chan * 2))
    iR = int(min(len(absFDF), indxPeakPIchan + fwhmRMSF_chan * 2))
    absFDFmsked = rm_fdf.copy()
    absFDFmsked[iL:iR] = np.nan
    absFDFmsked = absFDFmsked[np.where(absFDFmsked == absFDFmsked)]
    if float(len(absFDFmsked)) / len(absFDF) < 0.3:
        dFDFcorMAD = MAD(rm_fdf)
    else:
        dFDFcorMAD = MAD(absFDFmsked)

    # The noise is re-normalized by the predicted noise at the peak RM.
    dFDFcorMAD = dFDFcorMAD * adjoint_noise[indxPeakPIchan]

    nChansGood = np.sum(np.where(lamSqArr_m2 == lamSqArr_m2, 1.0, 0.0))
    varLamSqArr_m2 = (
        np.sum(lamSqArr_m2**2.0) - np.sum(lamSqArr_m2) ** 2.0 / nChansGood
    ) / (nChansGood - 1)

    # Determine the peak in the FDF, its amplitude and Phi using a
    # 3-point parabolic interpolation
    phiPeakPIfit = None
    dPhiPeakPIfit = None
    ampPeakPIfit = None
    snrPIfit = None
    ampPeakPIfitEff = None
    indxPeakPIfit = None
    peakFDFimagFit = None
    peakFDFrealFit = None
    polAngleFit_deg = None
    dPolAngleFit_deg = None
    polAngle0Fit_deg = None
    dPolAngle0Fit_deg = None

    # Only do the 3-point fit if peak is 1-channel from either edge
    if indxPeakPIchan > 0 and indxPeakPIchan < len(FDF) - 1:
        phiPeakPIfit, ampPeakPIfit = calc_parabola_vertex(
            phiArr[indxPeakPIchan - 1],
            amp_fdf[indxPeakPIchan - 1],
            phiArr[indxPeakPIchan],
            amp_fdf[indxPeakPIchan],
            phiArr[indxPeakPIchan + 1],
            amp_fdf[indxPeakPIchan + 1],
        )

        snrPIfit = ampPeakPIfit * adjoint_sens[indxPeakPIchan] / dFDF

        # Error on fitted Faraday depth (RM) is same as channel
        # but using fitted PI
        dPhiPeakPIfit = fwhmRMSF / (2.0 * snrPIfit)

        # Correct the peak for polarisation bias (POSSUM report 11)
        ampPeakPIfitEff = ampPeakPIfit
        if snrPIfit >= snrDoBiasCorrect:
            ampPeakPIfitEff = np.sqrt(ampPeakPIfit**2.0 - 2.3 * dampPeakPI**2.0)

        # Calculate the polarisation angle from the fitted peak
        # Uncertainty from Eqn A.12 in Brentjens & De Bruyn 2005
        indxPeakPIfit = np.interp(
            phiPeakPIfit, phiArr, np.arange(phiArr.shape[-1], dtype="f4")
        )
        peakFDFimagFit = np.interp(phiPeakPIfit, phiArr, FDF.imag)
        peakFDFrealFit = np.interp(phiPeakPIfit, phiArr, FDF.real)
        polAngleFit_deg = (
            0.5 * np.degrees(np.arctan2(peakFDFimagFit, peakFDFrealFit)) % 180
        )
        dPolAngleFit_deg = np.degrees(1 / (2.0 * snrPIfit))

        # Calculate the derotated polarisation angle and uncertainty
        # Uncertainty from Eqn A.20 in Brentjens & De Bruyn 2005
        polAngle0Fit_deg = (
            np.degrees(np.radians(polAngleFit_deg) - phiPeakPIfit * lam0Sq)
        ) % 180
        dPolAngle0Fit_rad = np.sqrt(
            nChansGood
            / (4.0 * (nChansGood - 2.0) * snrPIfit**2.0)
            * ((nChansGood - 1) / nChansGood + lam0Sq**2.0 / varLamSqArr_m2)
        )
        dPolAngle0Fit_deg = np.degrees(dPolAngle0Fit_rad)

    # Store the measurements in a dictionary and return
    mDict = {
        "dFDFcorMAD": toscalar(dFDFcorMAD),
        "phiPeakPIfit_rm2": toscalar(phiPeakPIfit),
        "dPhiPeakPIfit_rm2": toscalar(dPhiPeakPIfit),
        "ampPeakPIfit": toscalar(ampPeakPIfit),
        "ampPeakPIfitEff": toscalar(ampPeakPIfitEff),
        "dAmpPeakPIfit": toscalar(dampPeakPI),
        "snrPIfit": toscalar(snrPIfit),
        "indxPeakPIfit": toscalar(indxPeakPIfit),
        "peakFDFimagFit": toscalar(peakFDFimagFit),
        "peakFDFrealFit": toscalar(peakFDFrealFit),
        "polAngleFit_deg": toscalar(polAngleFit_deg),
        "dPolAngleFit_deg": toscalar(dPolAngleFit_deg),
        "polAngle0Fit_deg": toscalar(polAngle0Fit_deg),
        "dPolAngle0Fit_deg": toscalar(dPolAngle0Fit_deg),
    }

    return mDict


# -----------------------------------------------------------------------------#
def do_adjoint_rmsynth_planes(
    freqArr_Hz,
    dataQ,
    dataU,
    phiArr_radm2,
    widths_Hz=None,
    weightArr=None,
    lam0Sq_m2=None,
    nBits=64,
    verbose=False,
    log=print,
):
    """Perform RM-synthesis on Stokes Q and U cubes (1,2 or 3D). This version
    of the routine loops through spectral planes and is faster than the pixel-
    by-pixel code. This version also correctly deals with isolated clumps of
    NaN-flagged voxels within the data-cube (unlikely in interferometric cubes,
    but possible in single-dish cubes). Input data must be in standard python
    [z,y,x] order, where z is the frequency axis in ascending order.

    This has been modified from the convientual RMtools_1D version for bwdepol

    Parameters
    ----------
    dataQ           ... 1, 2 or 3D Stokes Q data array
    dataU           ... 1, 2 or 3D Stokes U data array
    lambdaSqArr_m2  ... vector of wavelength^2 values (assending freq order)
    phiArr_radm2    ... vector of trial Faraday depth values
    weightArr       ... vector of weights, default [None] is Uniform (all 1s)
    nBits           ... precision of data arrays [32]
    verbose         ... print feedback during calculation [False]
    log             ... function to be used to output messages [print]

    Returns
    -------
    FDFcube: array
             Faraday Dispersion Function (FDF)

    lam0Sq_m2: array
               lam0Sq_m2 is the weighted mean of lambda^2 distribution
               (B&dB Eqn. 32)

    adjoint_vars: list
                  information to generate theoretical noise, sensitivity
                  adjoint_vars = [widths_Hz, freqArr_Hz, phiArr_radm2, K,
                                  weightArr]

    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    lambdaSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)
    # Set the weight array
    if weightArr is None:
        weightArr = np.ones(lambdaSqArr_m2.shape, dtype=dtFloat)
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)

    # Sanity check on array sizes
    if not weightArr.shape == lambdaSqArr_m2.shape:
        log("Err: Lambda^2 and weight arrays must be the same shape.")
        return None, None
    if not dataQ.shape == dataU.shape:
        log("Err: Stokes Q and U data arrays must be the same shape.")
        return None, None
    nDims = len(dataQ.shape)
    if not nDims <= 3:
        log("Err: data dimensions must be <= 3.")
        return None, None
    if not dataQ.shape[0] == lambdaSqArr_m2.shape[0]:
        log(
            "Err: Data depth does not match lambda^2 vector ({} vs {}).".format(
                dataQ.shape[0], lambdaSqArr_m2.shape[0]
            ),
            end=" ",
        )
        log("     Check that data is in [z, y, x] order.")
        return None, None

    # Reshape the data arrays to 3 dimensions
    if nDims == 1:
        dataQ = np.reshape(dataQ, (dataQ.shape[0], 1, 1))
        dataU = np.reshape(dataU, (dataU.shape[0], 1, 1))
    elif nDims == 2:
        dataQ = np.reshape(dataQ, (dataQ.shape[0], dataQ.shape[1], 1))
        dataU = np.reshape(dataU, (dataU.shape[0], dataU.shape[1], 1))

    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY, nX]
    pCube = (dataQ + 1j * dataU) * weightArr[:, np.newaxis, np.newaxis]

    # Check for NaNs (flagged data) in the cube & set to zero
    mskCube = np.isnan(pCube)
    pCube = np.nan_to_num(pCube)

    # If full planes are flagged then set corresponding weights to zero
    mskPlanes = np.sum(np.sum(~mskCube, axis=1), axis=1)
    mskPlanes = np.where(mskPlanes == 0, 0, 1)
    weightArr *= mskPlanes

    # Initialise the complex Faraday Dispersion Function cube
    nX = dataQ.shape[-1]
    nY = dataQ.shape[-2]
    nPhi = phiArr_radm2.shape[0]
    FDFcube = np.zeros((nPhi, nY, nX), dtype=dtComplex)

    # lam0Sq_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a global lam0Sq_m2 value, ignoring isolated flagged voxels
    K = 1.0 / np.sum(weightArr)
    if lam0Sq_m2 is None:
        lam0Sq_m2 = K * np.sum(weightArr * lambdaSqArr_m2)

    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weightCube = np.invert(mskCube) * weightArr[:, np.newaxis, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
        KArr[KArr == np.inf] = 0
        KArr = np.nan_to_num(KArr)

    # Do the RM-synthesis on each plane
    if verbose:
        log("Running RM-synthesis by channel.")
        progress(40, 0)

    # calculate channel widths if necessary
    if widths_Hz == None:
        widths_Hz = estimate_channel_bandwidth(freqArr_Hz)
    for i in range(nPhi):
        if verbose:
            progress(40, ((i + 1) * 100.0 / nPhi))
        cor = np.exp(2j * phiArr_radm2[i] * lam0Sq_m2)
        r_i = rotation_operator(widths_Hz, freqArr_Hz, phiArr_radm2[i])[
            :, np.newaxis, np.newaxis
        ]
        arg0 = pCube * cor * np.conj(r_i)
        arg = arg0
        FDFcube[i, :, :] = KArr * np.sum(arg, axis=0)

    # information to generate theoretical noise, sensitivity
    adjoint_vars = [widths_Hz, freqArr_Hz, phiArr_radm2, K, weightArr]

    # Remove redundant dimensions in the FDF array
    FDFcube = np.squeeze(FDFcube)
    return FDFcube, lam0Sq_m2, adjoint_vars


# -----------------------------------------------------------------------------#
def run_adjoint_rmsynth(
    data,
    polyOrd=3,
    phiMax_radm2=None,
    dPhi_radm2=None,
    nSamples=10.0,
    weightType="variance",
    fitRMSF=True,
    noStokesI=False,
    phiNoise_radm2=1e6,
    nBits=64,
    showPlots=False,
    debug=False,
    verbose=False,
    log=print,
    units="Jy/beam",
):
    """Run bwdepol RM synthesis on 1D data.

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
    #    dtComplex = "complex" + str(2*nBits)

    # freq_Hz, I, Q, U, dI, dQ, dU
    if data.shape[0] == 7:
        if verbose:
            log("> Seven columns found, trying [freq_Hz, I, Q, U, dI, dQ, dU]", end=" ")
        (freqArr_Hz, IArr, QArr, UArr, dIArr, dQArr, dUArr) = data
        widths_Hz = None
    elif data.shape[0] == 8:
        if verbose:
            log(
                "> Eight columns found, trying [freq_Hz, widths_Hz, I, Q, U, dI, dQ, dU]",
                end=" ",
            )
        (freqArr_Hz, widths_Hz, IArr, QArr, UArr, dIArr, dQArr, dUArr) = data
    elif data.shape[0] == 6:
        if verbose:
            log(
                "> Six columns found, trying [freq_Hz, widths_Hz, Q, U, dQ, dU]",
                end=" ",
            )
        (freqArr_Hz, width_Hz, QArr, UArr, dQArr, dUArr) = data
    elif data.shape[0] == 5:
        if verbose:
            log("> Five columns found, trying [freq_Hz, Q, U, dQ, dU]", end=" ")
        (freqArr_Hz, QArr, UArr, dQArr, dUArr) = data
        widths_Hz = None
        noStokesI = True
    else:
        log("Failed to read in data, aborting.")
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
        freqArr=freqArr_GHz,
        IArr=IArr,
        QArr=QArr,
        UArr=UArr,
        dIArr=dIArr,
        dQArr=dQArr,
        dUArr=dUArr,
        polyOrd=polyOrd,
        verbose=True,
        debug=debug,
    )

    # Plot the data and the Stokes I model fit
    if showPlots:
        if verbose:
            log("Plotting the input data and spectral index fit.")
        freqHirArr_Hz = np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)
        IModHirArr = poly5(fit_result.params)(freqHirArr_Hz / 1e9)
        specFig = plt.figure(figsize=(12.0, 8))
        plot_Ipqu_spectra_fig(
            freqArr_Hz=freqArr_Hz,
            IArr=IArr,
            qArr=qArr,
            uArr=uArr,
            dIArr=dIArr,
            dqArr=dqArr,
            duArr=duArr,
            freqHirArr_Hz=freqHirArr_Hz,
            IModArr=IModHirArr,
            fig=specFig,
            units=units,
        )

        # DEBUG (plot the Q, U and average RMS spectrum)
        if debug:
            rmsFig = plt.figure(figsize=(12.0, 8))
            ax = rmsFig.add_subplot(111)
            ax.plot(
                freqArr_Hz / 1e9,
                dQUArr,
                marker="o",
                color="k",
                lw=0.5,
                label="rms <QU>",
            )
            ax.plot(
                freqArr_Hz / 1e9, dQArr, marker="o", color="b", lw=0.5, label="rms Q"
            )
            ax.plot(
                freqArr_Hz / 1e9, dUArr, marker="o", color="r", lw=0.5, label="rms U"
            )
            xRange = (np.nanmax(freqArr_Hz) - np.nanmin(freqArr_Hz)) / 1e9
            ax.set_xlim(
                np.min(freqArr_Hz) / 1e9 - xRange * 0.05,
                np.max(freqArr_Hz) / 1e9 + xRange * 0.05,
            )
            ax.set_xlabel(r"$\nu$ (GHz)")
            ax.set_ylabel("RMS " + units)
            ax.set_title("RMS noise in Stokes Q, U and <Q,U> spectra")

    # Calculate some wavelength parameters
    lambdaSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)
    # dFreq_Hz = np.nanmin(np.abs(np.diff(freqArr_Hz)))
    lambdaSqRange_m2 = np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2)
    # dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
    dLambdaSqMax_m2 = np.nanmax(np.abs(np.diff(lambdaSqArr_m2)))

    # Set the Faraday depth range
    fwhmRMSF_radm2 = 2.0 * m.sqrt(3.0) / lambdaSqRange_m2
    if dPhi_radm2 is None:
        dPhi_radm2 = fwhmRMSF_radm2 / nSamples
    if phiMax_radm2 is None:
        phiMax_radm2 = m.sqrt(3.0) / dLambdaSqMax_m2
        phiMax_radm2 = max(phiMax_radm2, 600.0)  # Force the minimum phiMax

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
        weightArr = 1.0 / np.power(dQUArr, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    if verbose:
        log("Weight type is '%s'." % weightType)

    startTime = time.time()

    # Perform adjoint RM-synthesis on the spectrum
    dirtyFDF, lam0Sq_m2, adjoint_vars = do_adjoint_rmsynth_planes(
        freqArr_Hz=freqArr_Hz,
        widths_Hz=widths_Hz,
        dataQ=qArr,
        dataU=uArr,
        phiArr_radm2=phiArr_radm2,
        weightArr=weightArr,
        nBits=nBits,
        verbose=verbose,
        log=log,
    )

    # generate adjoint_noise and adjoint__sens
    adjoint_info = adjoint_theory(adjoint_vars, dQUArr, show_progress=False)
    phiArr_radm2, adjoint_sens, adjoint_noise = adjoint_info

    # calculate peak RM
    absFDF = np.abs(dirtyFDF)
    rm_fdf = absFDF / adjoint_noise  # used for finding peak in RM
    indxPeakPIchan = np.nanargmax(rm_fdf[1:-1]) + 1
    peak_rm = phiArr_radm2[indxPeakPIchan]

    # Calculate the Rotation Measure Spread Function
    RMSFArr, phi2Arr_radm2, fwhmRMSFArr, fitStatArr = bwdepol_get_rmsf_planes(
        freqArr_Hz=freqArr_Hz,
        widths_Hz=widths_Hz,
        phiArr_radm2=phiArr_radm2,
        weightArr=weightArr,
        mskArr=~np.isfinite(qArr),
        lam0Sq_m2=lam0Sq_m2,
        double=True,
        fitRMSF=fitRMSF,
        fitRMSFreal=False,
        nBits=nBits,
        verbose=verbose,
        log=log,
        peak_rm=peak_rm,
    )
    fwhmRMSF = float(fwhmRMSFArr)
    endTime = time.time()
    cputime = endTime - startTime
    if verbose:
        log("> RM-synthesis completed in %.2f seconds." % cputime)

    # Determine the Stokes I value at lam0Sq_m2 from the Stokes I model
    # Multiply the dirty FDF by Ifreq0 to recover the PI
    freq0_Hz = speed_of_light.value / m.sqrt(lam0Sq_m2)
    Ifreq0 = poly5(fit_result.params)(freq0_Hz / 1e9)
    dirtyFDF *= Ifreq0  # FDF is in fracpol units initially, convert back to flux

    # Calculate the theoretical noise in the FDF !!
    # Old formula only works for wariance weights!
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)
    dFDFth = np.sqrt(
        np.sum(weightArr**2 * np.nan_to_num(dQUArr) ** 2) / (np.sum(weightArr)) ** 2
    )

    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties

    mDict = bwdepol_measure_FDF_parms(
        FDF=dirtyFDF,
        phiArr=phiArr_radm2,
        fwhmRMSF=fwhmRMSF,
        adjoint_sens=adjoint_sens,
        adjoint_noise=adjoint_noise,
        dFDF=dFDFth,
        lamSqArr_m2=lambdaSqArr_m2,
        lam0Sq=lam0Sq_m2,
    )

    mDict["Ifreq0"] = toscalar(Ifreq0)
    mDict["polyCoeffs"] = ",".join([str(x) for x in fit_result.params])
    mDict["IfitStat"] = fit_result.fitStatus
    mDict["IfitChiSqRed"] = fit_result.chiSqRed
    mDict["lam0Sq_m2"] = toscalar(lam0Sq_m2)
    mDict["freq0_Hz"] = toscalar(freq0_Hz)
    mDict["fwhmRMSF"] = toscalar(fwhmRMSF)
    mDict["dQU"] = toscalar(nanmedian(dQUArr))
    # mDict["dFDFth"] = toscalar(dFDFth)
    mDict["units"] = units

    if fit_result.fitStatus >= 128:
        log("WARNING: Stokes I model contains negative values!")
    elif fit_result.fitStatus >= 64:
        log("Caution: Stokes I model has low signal-to-noise.")

    # Add information on nature of channels:
    good_channels = np.where(np.logical_and(weightArr != 0, np.isfinite(qArr)))[0]
    mDict["min_freq"] = float(np.min(freqArr_Hz[good_channels]))
    mDict["max_freq"] = float(np.max(freqArr_Hz[good_channels]))
    mDict["N_channels"] = good_channels.size
    if widths_Hz != None:
        mDict["median_channel_width"] = float(np.median(widths_Hz))
    else:
        mDict["median_channel_width"] = float(np.median(np.diff(freqArr_Hz)))

    # Measure the complexity of the q and u spectra
    mDict["fracPol"] = mDict["ampPeakPIfit"] / (Ifreq0)
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
        tmpFig.show()

    # add array dictionary
    aDict = dict()
    aDict["phiArr_radm2"] = phiArr_radm2
    aDict["phi2Arr_radm2"] = phi2Arr_radm2
    aDict["RMSFArr"] = RMSFArr
    aDict["freqArr_Hz"] = freqArr_Hz
    aDict["weightArr"] = weightArr
    aDict["dirtyFDF"] = dirtyFDF / adjoint_sens

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
        log()
        log("-" * 80)

    # Plot the RM Spread Function and dirty FDF
    if showPlots:
        plot_adjoint_info(adjoint_info, units=units)
        fdfFig = plt.figure(figsize=(12.0, 8))
        bwdepol_plot_rmsf_fdf_fig(
            phiArr=phiArr_radm2,
            FDF=(dirtyFDF / adjoint_sens),
            phi2Arr=phiArr_radm2,
            RMSFArr=RMSFArr,
            peak_rm=peak_rm,
            fwhmRMSF=fwhmRMSF,
            vLine=mDict["phiPeakPIfit_rm2"],
            fig=fdfFig,
            units=units,
        )

    if showPlots or debug:
        plt.show()

    return mDict, aDict


# -----------------------------------------------------------------------------#
def main():
    """
    Start the function to perform bwdepol RM-synthesis if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run bandwidth-depolarization-corrected RM-synthesis (based on Fine et al 2022)
    on Stokes I, Q and U spectra (1D) stored in an ASCII file.

    Behaves similarly to rmsynth1d except that the input file can optionally
    contain a column with the channel widths in Hz. If this column is not
    given, the channel widths will be assumed to be uniform and calculated
    based on the difference between the frequencies of the first two channels.

    The ASCII file requires one of the following column configurations,
    depending on whether Stokes I and channel width information are available,
    in a space separated format:
    [freq_Hz, I, Q, U, I_err, Q_err, U_err]
    [freq_Hz, widths_Hz, I, Q, U, I_err, Q_err, U_err]
    [freq_Hz, Q, U, Q_err, U_err]
    [freq_Hz, widths_Hz, Q, U, Q_err, U_err]


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
        action="store_false",
        help="fit a Gaussian to the RMSF [True; set flag to disable]",
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
        help="weighting [inverse variance] or 'uniform' (all 1s).",
    )
    parser.add_argument(
        "-o",
        dest="polyOrd",
        type=int,
        default=2,
        help="polynomial order to fit to I spectrum [2].",
    )
    parser.add_argument(
        "-i",
        dest="noStokesI",
        action="store_true",
        help="ignore the Stokes I spectrum [False].",
    )
    parser.add_argument(
        "-p", dest="showPlots", action="store_true", help="show the plots [False]."
    )
    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="verbose output [False]."
    )
    parser.add_argument(
        "-S", dest="saveOutput", action="store_true", help="save the arrays [False]."
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
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.dataFile[0]):
        print("File does not exist: '%s'." % args.dataFile[0])
        sys.exit()
    prefixOut, ext = os.path.splitext(args.dataFile[0])
    dataDir, dummy = os.path.split(args.dataFile[0])
    # Set the floating point precision
    nBits = 64

    # Read in the data. Don't parse until inside the first function.
    data = np.loadtxt(args.dataFile[0], unpack=True, dtype="float" + str(nBits))

    # Run (modified) RM-synthesis on the spectra
    mDict, aDict = run_adjoint_rmsynth(
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
        verbose=args.verbose,
        units=args.units,
    )

    if args.saveOutput:
        saveOutput(mDict, aDict, prefixOut, args.verbose)


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
