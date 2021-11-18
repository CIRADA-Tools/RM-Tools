#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RMsynth_1D.py                                                  #
#                                                                             #
# PURPOSE: API for runnning RM-synthesis on an ASCII Stokes I, Q & U spectrum.#
#                                                                             #
# MODIFIED: 16-Nov-2018 by J. West                                            #
# MODIFIED: 23-October-2019 by A. Thomson                                     #
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
import time
import traceback
import json
import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from RMutils.util_RM import do_rmsynth
#from RMutils.util_RM import get_rmsf_planes
#from RMutils.util_RM import fit_rmsf
from RMutils.util_RM import detect_peak
from RMutils.util_RM import extrap
#from RMutils.util_RM import measure_FDF_parms
from RMutils.util_RM import calc_parabola_vertex
from RMutils.util_RM import measure_qu_complexity
from RMutils.util_RM import measure_fdf_complexity
from RMutils.util_misc import nanmedian
from RMutils.util_misc import toscalar
from RMutils.util_misc import create_frac_spectra
from RMutils.util_misc import poly5
from RMutils.util_misc import MAD
from RMutils.util_plotTk import plot_Ipqu_spectra_fig
#from RMutils.util_plotTk import plot_rmsf_fdf_fig
from RMutils.util_plotTk import plot_complexity_fig
from RMutils.util_plotTk import CustomNavbar
from RMutils.util_plotTk import plot_rmsIQU_vs_nu_ax
from RMutils.mpfit import mpfit
from RMutils.util_plotTk import plot_dirtyFDF_ax   


#import time
import argparse
import scipy as sp
#import pdb

if sys.version_info.major == 2:
    print('RM-tools will no longer run with Python 2! Please use Python 3.')
    exit()

from astropy.io import fits
from RMtools_1D.make_freq_file import get_freq_array
#from RMtools_1D.do_RMsynth_1D import run_rmsynth, saveOutput
import numpy as np
from astropy import wcs
import sys
from RMutils.util_misc import progress  


C = 2.997924538e8 # Speed of light [m/s]



#-----------------------------------------------------------------------------#

# helper functions

def analyticv3(f, phi):
   '''alculates the average analytic solution to the channel polarization integral for 1 channel end'''
   funct1 = (f * np.exp(2.0j * phi * ((C/f)**2)))
   funct2 = (C * np.sqrt((np.abs(phi)*np.pi)))
   funct3 = (-1.0j + np.sign(phi))
   funct4 = (sp.special.erf(np.sqrt(np.abs(phi)) * (C / f)*(-1.0j + np.sign(phi))))
    

   ya = (funct1 + (funct2 * funct3 * funct4))
    
   return ya
    
def rotation_op(delta_v, v_knot, phi):  
    '''Rotation operator based on equation #25'''
    b = v_knot + 0.5 * delta_v
    a = v_knot - 0.5 * delta_v
    
    int_a = analyticv3(a, phi)
    int_b = analyticv3(b, phi)
    
    return (1/delta_v) * (int_b - int_a)


def bandwidth(f_array):
    '''Returns bandwidth per channel of a frequency array'''

    ban = f_array[1] - f_array[0]
    return ban


def freq_array(lambda_square_array):
    '''returns the freqency array, corresponding to a lambda square array'''
    
    f = C**2/ lambda_square_array
    return np.sqrt(f)
    

def v_array_func(r_array):
    '''equation 28'''
    
    v_array1 = np.abs(r_array) / r_array # doesnt have to be an array
    
    return v_array1

def adjoint_theory(adjoint_varbs, dQUArr, show_progress=False, log=print):
    '''Calculates the theoretical sensitivity and noise'''
    
    delta_v, v_array1, phiArr_radm2, K, weightArr = adjoint_varbs 
    adjoint_noise = np.ones(len(phiArr_radm2))
    adjoint_sens = np.ones(len(phiArr_radm2))
    
    nPhi = len(phiArr_radm2)
    if show_progress:
        log('Calculating Theoretical Sensitivity & Noise')
        progress(40, 0)
    for i in range(nPhi):
        if show_progress:
            progress(40, ((i+1)*100.0/nPhi))
            
        r_i = rotation_op(delta_v, v_array1, phiArr_radm2[i])
        adjoint_noise2 = np.sum((weightArr * dQUArr)**2 * np.abs(r_i)**2) / np.sum(weightArr)**2 # equation 34
        adjoint_noise[i] = (np.sqrt(adjoint_noise2) )
        
        adjoint_sens[i] = K * np.sum(weightArr * (np.abs(r_i)**2))
        
    
    
    adjoint_info = [phiArr_radm2, adjoint_sens, adjoint_noise]
    return adjoint_info
    


def plot_adjoint_info(mylist, units='Jy/beam'):
    '''plots theoretical noise, sensitivity'''
    fig, ax = plt.subplots(2, dpi=100, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    [phiArr_radm2, adjoint_sens, adjoint_noise] = mylist
    
           
    ax[1].plot(phiArr_radm2, adjoint_noise,)
    ax[1].set_xlabel('$\phi$ (rad m$^{-2}$)')
    ax[1].set_ylabel('Noise' + ' (' + units + ')')
    ax[1].set_title('Theoretical Noise')
    # plot 2   
    ax[0].plot(phiArr_radm2, adjoint_sens,)
    ax[0].set_xlabel('$\phi$ (rad m$^{-2}$)')
    #ax[0].set_ylabel('Theoretical Sensitivity')
    ax[0].set_title('Theoretical Sensitivity')
    return

#-----------------------------------------------------------------------------#
#mode 2
# also a part of the RSMF
def depolarization(data, phiMax_radm2=None, polyOrd=3, debug=False,dPhi_radm2=None,
                nSamples=10.0, weightType="variance",
                noStokesI=False, phiNoise_radm2=1e6,
                verbose=False, log=print,units='Jy/beam',):
    """Plots theoretical sensitivity and noise only AKA mode2.

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
    nBits = 64
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # freq_Hz, I, Q, U, dI, dQ, dU
    try:
        if verbose: log("> Trying [freq_Hz, I, Q, U, dI, dQ, dU]", end=' ')
        (freqArr_Hz, IArr, QArr, UArr, dIArr, dQArr, dUArr) = data
        if verbose: log("... success.")
    except Exception:
        if verbose: log("...failed.")
        # freq_Hz, q, u, dq, du
        try:
            if verbose: log("> Trying [freq_Hz, q, u,  dq, du]", end=' ')
            (freqArr_Hz, QArr, UArr, dQArr, dUArr) = data
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
        if verbose: log("Warn: no Stokes I data in use.")
        IArr = np.ones_like(QArr)
        dIArr = np.zeros_like(QArr)

    # Convert to GHz for convenience
    freqArr_GHz = freqArr_Hz / 1e9
    dQUArr = (dQArr + dUArr)/2.0

    # Fit the Stokes I spectrum and create the fractional spectra
    IModArr, qArr, uArr, dqArr, duArr, fitDict = \
             create_frac_spectra(freqArr  = freqArr_GHz,
                                 IArr     = IArr,
                                 QArr     = QArr,
                                 UArr     = UArr,
                                 dIArr    = dIArr,
                                 dQArr    = dQArr,
                                 dUArr    = dUArr,
                                 polyOrd  = polyOrd,
                                 verbose  = True,
                                 debug    = debug)

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
    

    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weightType=="variance":
        weightArr = 1.0 / np.power(dQUArr, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    if verbose: log("Weight type is '%s'." % weightType)

    
    # generate adjoint_noise and adjoint__sens
    K = 1.0 / np.sum(weightArr)
    v_array = freq_array(lambdaSqArr_m2)
    delta_v = bandwidth(v_array)

    adjoint_varbs = [delta_v, v_array, phiArr_radm2, K, weightArr]
    adjoint_info = adjoint_theory(adjoint_varbs, dQUArr, show_progress=True)
    phiArr_radm2, adjoint_sens, adjoint_noise = adjoint_info 

    # plot adjoint info
    
    plot_adjoint_info(adjoint_info, units=units)
    plt.show()

    return 
#-----------------------------------------------------------------------------#
# functions for simulation, part of calculating new RSMF 
    

def sim_dirtyFDF(data, phiMax_radm2=None, polyOrd=3, debug=False,dPhi_radm2=None,
                nSamples=10.0, weightType="variance",
                noStokesI=True, phiNoise_radm2=1e6,
                verbose=False, log=print,units='Jy/beam', ):
    """
    """

    # Default data types
    nBits = 64
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # freq_Hz, I, Q, U, dI, dQ, dU
   
    (freqArr_Hz, QArr, UArr, dQArr, dUArr) = data
   

    # If no Stokes I present, create a dummy spectrum = unity
    if noStokesI:
        IArr = np.ones_like(QArr)
        dIArr = np.zeros_like(QArr)

    # Convert to GHz for convenience
    freqArr_GHz = freqArr_Hz / 1e9
    dQUArr = (dQArr + dUArr)/2.0

    # Fit the Stokes I spectrum and create the fractional spectra
    IModArr, qArr, uArr, dqArr, duArr, fitDict = \
             create_frac_spectra(freqArr  = freqArr_GHz,
                                 IArr     = IArr,
                                 QArr     = QArr,
                                 UArr     = UArr,
                                 dIArr    = dIArr,
                                 dQArr    = dQArr,
                                 dUArr    = dUArr,
                                 polyOrd  = polyOrd,
                                 verbose  = True,
                                 debug    = debug)

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
    

    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weightType=="variance":
        weightArr = 1.0 / np.power(dQUArr, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    if verbose: log("Weight type is '%s'." % weightType)

    
    # generate adjoint_noise and adjoint__sens
    K = 1.0 / np.sum(weightArr)
    v_array = freq_array(lambdaSqArr_m2)
    delta_v = bandwidth(v_array)

    adjoint_varbs = [delta_v, v_array, phiArr_radm2, K, weightArr]
    adjoint_info = adjoint_theory(adjoint_varbs, dQUArr, show_progress=False)
    phiArr_radm2, adjoint_sens, adjoint_noise = adjoint_info 

  
    dirtyFDF, lam0Sq_m2, adjoint_varbs= do_rmsynth_planes(dataQ           = qArr,
                                            dataU           = uArr,
                                            lambdaSqArr_m2  = lambdaSqArr_m2,
                                            phiArr_radm2    = phiArr_radm2,
                                            weightArr       = weightArr,
                                            nBits           = nBits,
                                            verbose         = False,
                                            log             = False)


    return dirtyFDF
    
# noise is 0, with unit intensity
def bandwidth_avg_array(f, ban, phi, xi_knot=0, p=1):
    '''Calculates the average analytic solution to the channel polarization integral for 1 channel
    
    Based on equation 13 of Schnitzeler & Lee (2015)
    
    Args:
    f = channel center frequency (in Hz)
    ban = bandwidth (in Hz)
    phi =  faraday depth value (in rad/m2)
    xi_knot = initial polarization angle (in rad)
    p = polarized intensity
    
    Returns:
    avg_p_tilda = the average complex polarization, for the bandwidth, real is Q, imaginary is U
    '''
    a = f - (ban / 2)
    b = f + (ban / 2) # integral start and stop values
                   
    ya =  analyticv3(a, phi, )
    yb =  analyticv3(b, phi, ) # check orig for xi_knot
                     
    i = p* (yb - ya)
    avg_p_tilda = i / ban
    
    return avg_p_tilda


def simulation(peak_rm, v_array, phiMax_radm2=None):
    '''simulated source of the same RM as the measured source, 
    with unit intensity
    
    Returns:
        Dirty FDF for the simulated data'''
    
    s = 1.0e-5 # error for simulation
    
    ban = bandwidth(v_array)
    
    p_tilda = bandwidth_avg_array(v_array, ban, peak_rm)
    size_f = len(v_array)
    noise_1 =  np.random.normal(0, s, size_f)
    noise_2 =  np.random.normal(0, s, size_f)
    dq = s * np.ones(size_f)
    du = s * np.ones(size_f)
    q = np.real(p_tilda) + noise_1
    u = np.imag(p_tilda) + noise_2
    p_tilda = q + 1.0j * u
    
    #imput_data = list = [freq_Hz, q, u,  dq, du]
    data=[v_array, np.real(p_tilda), np.imag(p_tilda), dq, du]
    
    #phiMax_radm2 = 1.2* peak_rm

    
    dirtyFDF = sim_dirtyFDF(data, phiMax_radm2=phiMax_radm2, polyOrd=3, debug=False,dPhi_radm2=None,
                nSamples=10.0, weightType="variance", phiNoise_radm2=1e6,units='Jy/beam', )
    
    return dirtyFDF
    
# p_tilda = bandwidth_avg_array(f, ban, phi, xi_knot, p)
    

#-----------------------------------------------------------------------------#
# modified plotting for the RMSF
 
    
def tweakAxFormat(ax, pad=10, loc='upper right', linewidth=1, ncol=1,
                   bbox_to_anchor=(1.00, 1.00), showLeg=True):
    
    # Axis/tic formatting
    ax.tick_params(pad=pad)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markeredgewidth(linewidth)
        
    # Legend formatting
    if showLeg:
        leg = ax.legend(numpoints=1, loc=loc, shadow=False,
                        borderaxespad=0.3, ncol=ncol,
                        bbox_to_anchor=bbox_to_anchor)
        for t in leg.get_texts():
            t.set_fontsize('small') 
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_alpha(0.5)

    return ax

def gauss(p, peak_rm):
    """Return a fucntion to evaluate a Gaussian with parameters
    p = [amp, mean, FWHM]
    off set my peak_rm"""
    
    a, b, w = p
    gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
    s = w / gfactor
    
    def rfunc(x):
        y = a * np.exp(-(x-b-peak_rm)**2.0 /(2.0 * s**2.0))
        return y
    
    return rfunc

def plot_RMSF_ax(ax, phiArr, RMSFArr, peak_rm,fwhmRMSF=None, axisYright=False,
                 axisXtop=False, doTitle=False):

    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")    
        
    # Plot the RMSF
    ax.step(phiArr, RMSFArr.real, where='mid', color='tab:blue', lw=0.5,
            label='Real')
    ax.step(phiArr, RMSFArr.imag, where='mid', color='tab:red', lw=0.5,
            label='Imaginary')
    ax.step(phiArr, np.abs(RMSFArr) , where='mid', color='k', lw=1.0,
            label='PI')
    ax.axhline(0, color='grey')
    if doTitle:
        ax.text(0.05, 0.84, 'RMSF', transform=ax.transAxes)

    # Plot the Gaussian fit
    if fwhmRMSF is not None:
        yGauss = np.max(np.abs(RMSFArr))*gauss([1.0, 0.0, fwhmRMSF],peak_rm)(phiArr)
        ax.plot(phiArr, yGauss, color='g',marker='None',mfc='w',
                mec='g', ms=10, label='Gaussian', lw=2.0, ls='--')
    
    # Scaling
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(phiArr)-np.nanmin(phiArr)
    ax.set_xlim( np.nanmin(phiArr) - xRange*0.01,
                 np.nanmax(phiArr) + xRange*0.01)
    ax.set_ylabel('Normalised Units')
    ax.set_xlabel('$\phi$ rad m$^{-2}$')
    ax.axhline(0, color='grey')

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True,True,True)
    
def plot_rmsf_fdf_fig(phiArr, FDF, phi2Arr, RMSFArr, peak_rm,fwhmRMSF=None,
                      gaussParm=[], vLine=None, fig=None,units='flux units'):
    """Plot the RMSF and FDF on a single figure."""

    # Default to a pyplot figure
    if fig==None:
        fig = plt.figure(figsize=(12.0, 8))
    # Plot the RMSF
    ax1 = fig.add_subplot(211)    
    plot_RMSF_ax(ax=ax1,
                 phiArr   = phi2Arr,
                 RMSFArr  = RMSFArr,
                 peak_rm  = peak_rm,
                 fwhmRMSF = fwhmRMSF,
                 doTitle  = True)
    [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax1.set_xlabel("")
    
    # Plot the FDF
    #Why are these next two lines here? Removing as part of units fix.
#    if len(gaussParm)==3:
#        gaussParm[0] *= 1e3
    ax2 = fig.add_subplot(212, sharex=ax1)
    plot_dirtyFDF_ax(ax=ax2,
                     phiArr     = phiArr,
                     FDFArr = FDF,
                     gaussParm  = gaussParm,
                     vLine      = vLine,
                     doTitle    = True,
                     units      = units)

    return fig

    


#-----------------------------------------------------------------------------#
# modified for adjoint 
def get_rmsf_planes(lambdaSqArr_m2, phiArr_radm2, peak_rm, weightArr=None, mskArr=None, 
                    lam0Sq_m2= None, double=True, fitRMSF=False,
                    fitRMSFreal=False, nBits=64, verbose=False,
                    log=print):
    """Calculate the Rotation Measure Spread Function from inputs. This version
    returns a cube (1, 2 or 3D) of RMSF spectra based on the shape of a
    boolean mask array, where flagged data are True and unflagged data False.
    If only whole planes (wavelength channels) are flagged then the RMSF is the
    same for all pixels and the calculation is done once and replicated to the
    dimensions of the mask. If some isolated voxels are flagged then the RMSF
    is calculated by looping through each wavelength plane, which can take some
    time. By default the routine returns the analytical width of the RMSF main
    lobe but can also use MPFIT to fit a Gaussian.
    
    lambdaSqArr_m2  ... vector of wavelength^2 values (assending freq order)
    phiArr_radm2    ... vector of trial Faraday depth values
    weightArr       ... vector of weights, default [None] is no weighting    
    maskArr         ... cube of mask values used to shape return cube [None]
    lam0Sq_m2       ... force a reference lambda^2 value (def=calculate) [None]
    double          ... pad the Faraday depth to double-size [True]
    fitRMSF         ... fit the main lobe of the RMSF with a Gaussian [False]
    fitRMSFreal     ... fit RMSF.real, rather than abs(RMSF) [False]
    nBits           ... precision of data arrays [32]
    verbose         ... print feedback during calculation [False]
    log             ... function to be used to output messages [print]

    """
    
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)
    
    # For cleaning the RMSF should extend by 1/2 on each side in phi-space
    if double:
        nPhi = phiArr_radm2.shape[0]
        nExt = np.ceil(nPhi/2.0)
        resampIndxArr = np.arange(2.0 * nExt + nPhi) - nExt
        phi2Arr = extrap(resampIndxArr, np.arange(nPhi, dtype='int'),
                         phiArr_radm2)
    else:
        phi2Arr = phiArr_radm2

    # Set the weight array
    if weightArr is None:
        weightArr = np.ones(lambdaSqArr_m2.shape, dtype=dtFloat)
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)

    # Set the mask array (default to 1D, no masked channels)
    if mskArr is None:
        mskArr = np.zeros_like(lambdaSqArr_m2, dtype="bool")
        nDims = 1
    else:
        mskArr = mskArr.astype("bool")
        nDims = len(mskArr.shape)
    
    # Sanity checks on array sizes
    if not weightArr.shape  == lambdaSqArr_m2.shape:
        log("Err: wavelength^2 and weight arrays must be the same shape.")
        return None, None, None, None
    if not nDims <= 3:
        log("Err: mask dimensions must be <= 3.")
        return None, None, None, None
    if not mskArr.shape[0] == lambdaSqArr_m2.shape[0]:
        log("Err: mask depth does not match lambda^2 vector (%d vs %d).", end=' ')
        (mskArr.shape[0], lambdaSqArr_m2.shape[-1])
        log("     Check that the mask is in [z, y, x] order.")
        return None, None, None, None
    
    # Reshape the mask array to 3 dimensions
    if nDims==1:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], 1, 1))
    elif nDims==2:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], mskArr.shape[1], 1))
    
    # Create a unit cube for use in RMSF calculation (negative of mask)
    #CVE: unit cube removed: it wasn't accurate for non-uniform weights, and was no longer used
    
    # Initialise the complex RM Spread Function cube
    nX = mskArr.shape[-1]
    nY = mskArr.shape[-2]
    nPix = nX * nY
    nPhi = phi2Arr.shape[0]
    RMSFcube = np.ones((nPhi, nY, nX), dtype=dtComplex)

    # If full planes are flagged then set corresponding weights to zero
    xySum =  np.sum(np.sum(mskArr, axis=1), axis=1)
    mskPlanes = np.where(xySum==nPix, 0, 1)
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
    do1Dcalc = True
    if len(flagTotals)>0:
        do1Dcalc = False
    
    # lam0Sq is the weighted mean of LambdaSq distribution (B&dB Eqn. 32)
    # Calculate a single lam0Sq_m2 value, ignoring isolated flagged voxels
    K = 1.0 / np.nansum(weightArr)
    lam0Sq_m2 = K * np.nansum(weightArr * lambdaSqArr_m2)

    # Calculate the analytical FWHM width of the main lobe    
    fwhmRMSF = 2.0 * m.sqrt(3.0)/(np.nanmax(lambdaSqArr_m2) -
                                  np.nanmin(lambdaSqArr_m2))

    # Do a simple 1D calculation and replicate along X & Y axes
    if do1Dcalc:
        if verbose:
            log("Calculating 1D RMSF and replicating along X & Y axes.")

        # Calculate the RMSF # set up the simulation
        phiMax_radm2= phiArr_radm2[-1]
            
        v_array = freq_array(lambdaSqArr_m2)
        RMSFArr =  simulation(peak_rm, v_array, phiMax_radm2=phiMax_radm2)
        #RMSFArr = fdf from simulation
        
        # Fit the RMSF main lobe
        fitStatus = -1
        if fitRMSF:
            if verbose:
                log("Fitting Gaussian to the main lobe.")
            if fitRMSFreal:
                mp = fit_rmsf(phi2Arr, RMSFcube.real,)
            else:
                mp = fit_rmsf(phi2Arr, np.abs(RMSFArr))
            if mp is None or mp.status<1:
                 pass
                 log("Err: failed to fit the RMSF.")
                 log("     Defaulting to analytical value.")
            else:
                fwhmRMSF = mp.params[2]
                fitStatus = mp.status

        # Replicate along X and Y axes
        RMSFcube = np.tile(RMSFArr[:, np.newaxis, np.newaxis], (1, nY, nX))
        fwhmRMSFArr = np.ones((nY,nX), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nY, nX), dtype="int") * fitStatus

  
    

    # Remove redundant dimensions
    
    fwhmRMSFArr = np.squeeze(fwhmRMSFArr)
    statArr = np.squeeze(statArr)
    
    RMSFcube = RMSFcube.reshape(-1)
      
    return RMSFcube, phi2Arr, fwhmRMSFArr, statArr
#-----------------------------------------------------------------------------#
# modifed measure_FDF_parms for adjoint
def measure_FDF_parms(FDF, phiArr, fwhmRMSF, adjoint_sens, adjoint_noise,
                      dFDF=None, lamSqArr_m2=None,
                      lam0Sq=None, snrDoBiasCorrect=5.0):
    """
    Measure standard parameters from a complex Faraday Dispersion Function.
    Currently this function assumes that the noise levels in the Stokes Q
    and U spectra are the same.
    Returns a dictionary containing measured parameters.
    """
    
    # Determine the peak channel in the FDF, its amplitude and index
    absFDF = np.abs(FDF)
    rm_fdf = absFDF / adjoint_noise # used for finding peak in RM
    amp_fdf = absFDF/ adjoint_sens # used for finding amp peak
    indxPeakPIchan = np.nanargmax(rm_fdf[1:-1])+1  #Masks out the edge channels, since they can't be fit to.
    ampPeakPIchan = amp_fdf[indxPeakPIchan]
    
    # new dFDF correction for adjoint method
    dFDF = adjoint_noise[indxPeakPIchan]

    # Measure the RMS noise in the spectrum after masking the peak
    # changed all absFDF to amp_fdf
    dPhi = np.nanmin(np.diff(phiArr))
    fwhmRMSF_chan = np.ceil(fwhmRMSF/dPhi)
    iL = int(max(0, indxPeakPIchan-fwhmRMSF_chan*2))
    iR = int(min(len(absFDF), indxPeakPIchan+fwhmRMSF_chan*2))
    absFDFmsked = amp_fdf.copy()
    absFDFmsked[iL:iR] = np.nan
    absFDFmsked = absFDFmsked[np.where(absFDFmsked==absFDFmsked)]
    if float(len(absFDFmsked))/len(absFDF)<0.3:
        dFDFcorMAD = MAD(amp_fdf)
        dFDFrms = np.sqrt( np.mean(amp_fdf**2) )
    else:
        dFDFcorMAD = MAD(absFDFmsked)
        dFDFrms = np.sqrt( np.mean(absFDFmsked**2) )
    
    # Measure the RM of the peak channel
    phiPeakPIchan = phiArr[indxPeakPIchan]
    dPhiPeakPIchan = fwhmRMSF * dFDF / (2.0 * ampPeakPIchan)
    snrPIchan = ampPeakPIchan / dFDF
    
    # Correct the peak for polarisation bias (POSSUM report 11)
    ampPeakPIchanEff = ampPeakPIchan
    if snrPIchan >= snrDoBiasCorrect:
        ampPeakPIchanEff = np.sqrt(ampPeakPIchan**2.0 - 2.3 * dFDF**2.0)

    # Calculate the polarisation angle from the channel
    peakFDFimagChan = FDF.imag[indxPeakPIchan]
    peakFDFrealChan = FDF.real[indxPeakPIchan]
    polAngleChan_deg = 0.5 * np.degrees(np.arctan2(peakFDFimagChan,
                                         peakFDFrealChan)) % 180
    dPolAngleChan_deg = np.degrees(dFDF / (2.0 * ampPeakPIchan))

    # Calculate the derotated polarisation angle and uncertainty
    polAngle0Chan_deg = np.degrees(np.radians(polAngleChan_deg) -
                                  phiPeakPIchan * lam0Sq) % 180
    nChansGood = np.sum(np.where(lamSqArr_m2==lamSqArr_m2, 1.0, 0.0))
    varLamSqArr_m2 = (np.sum(lamSqArr_m2**2.0) -
                      np.sum(lamSqArr_m2)**2.0/nChansGood) / (nChansGood-1)
    dPolAngle0Chan_rad = \
        np.sqrt( dFDF**2.0*nChansGood / (4.0*(nChansGood-2.0)*ampPeakPIchan**2.0) *
                 ((nChansGood-1)/nChansGood + lam0Sq**2.0/varLamSqArr_m2) )
    dPolAngle0Chan_deg = np.degrees(dPolAngle0Chan_rad)
    
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
    if indxPeakPIchan > 0 and indxPeakPIchan < len(FDF)-1:
        phiPeakPIfit, ampPeakPIfit = \
                      calc_parabola_vertex(phiArr[indxPeakPIchan-1],
                                           amp_fdf[indxPeakPIchan-1],
                                           phiArr[indxPeakPIchan],
                                           amp_fdf[indxPeakPIchan],
                                           phiArr[indxPeakPIchan+1],
                                           amp_fdf[indxPeakPIchan+1])
        
        snrPIfit = ampPeakPIfit / dFDF
        
        # Error on fitted Faraday depth (RM) is same as channel, but using fitted PI
        dPhiPeakPIfit = fwhmRMSF * dFDF / (2.0 * ampPeakPIfit)
        
        
        # Correct the peak for polarisation bias (POSSUM report 11)
        ampPeakPIfitEff = ampPeakPIfit
        if snrPIfit >= snrDoBiasCorrect:
            ampPeakPIfitEff = np.sqrt(ampPeakPIfit**2.0 - 2.3 * dFDF**2.0)
            
        # Calculate the polarisation angle from the fitted peak
        # Uncertainty from Eqn A.12 in Brentjens & De Bruyn 2005
        indxPeakPIfit = np.interp(phiPeakPIfit, phiArr,
                                  np.arange(phiArr.shape[-1], dtype='f4'))
        peakFDFimagFit = np.interp(phiPeakPIfit, phiArr, FDF.imag)
        peakFDFrealFit = np.interp(phiPeakPIfit, phiArr, FDF.real)
        polAngleFit_deg = 0.5 * np.degrees(np.arctan2(peakFDFimagFit,
                                                  peakFDFrealFit)) % 180
        dPolAngleFit_deg = np.degrees(dFDF / (2.0 * ampPeakPIfit))

        # Calculate the derotated polarisation angle and uncertainty
        # Uncertainty from Eqn A.20 in Brentjens & De Bruyn 2005
        polAngle0Fit_deg = (np.degrees(np.radians(polAngleFit_deg) -
                                      phiPeakPIfit * lam0Sq)) % 180
        dPolAngle0Fit_rad = \
            np.sqrt( dFDF**2.0*nChansGood / (4.0*(nChansGood-2.0)*ampPeakPIfit**2.0) *
                    ((nChansGood-1)/nChansGood + lam0Sq**2.0/varLamSqArr_m2) )
        dPolAngle0Fit_deg = np.degrees(dPolAngle0Fit_rad)

    # Store the measurements in a dictionary and return
    mDict = {'dFDFcorMAD':       toscalar(dFDFcorMAD),
             'dFDFrms':          toscalar(dFDFrms),
             'phiPeakPIchan_rm2':     toscalar(phiPeakPIchan),
             'dPhiPeakPIchan_rm2':    toscalar(dPhiPeakPIchan),
             'ampPeakPIchan':    toscalar(ampPeakPIchan),
             'ampPeakPIchanEff': toscalar(ampPeakPIchanEff),
             'dAmpPeakPIchan':   toscalar(dFDF),
             'snrPIchan':             toscalar(snrPIchan),
             'indxPeakPIchan':        toscalar(indxPeakPIchan),
             'peakFDFimagChan':       toscalar(peakFDFimagChan),
             'peakFDFrealChan':       toscalar(peakFDFrealChan),
             'polAngleChan_deg':      toscalar(polAngleChan_deg),
             'dPolAngleChan_deg':     toscalar(dPolAngleChan_deg),
             'polAngle0Chan_deg':     toscalar(polAngle0Chan_deg),
             'dPolAngle0Chan_deg':    toscalar(dPolAngle0Chan_deg),
             'phiPeakPIfit_rm2':      toscalar(phiPeakPIfit),
             'dPhiPeakPIfit_rm2':     toscalar(dPhiPeakPIfit),
             'ampPeakPIfit':     toscalar(ampPeakPIfit),
             'ampPeakPIfitEff':  toscalar(ampPeakPIfitEff),
             'dAmpPeakPIfit':    toscalar(dFDF),
             'snrPIfit':              toscalar(snrPIfit),
             'indxPeakPIfit':         toscalar(indxPeakPIfit),
             'peakFDFimagFit':        toscalar(peakFDFimagFit),
             'peakFDFrealFit':        toscalar(peakFDFrealFit),
             'polAngleFit_deg':       toscalar(polAngleFit_deg),
             'dPolAngleFit_deg':      toscalar(dPolAngleFit_deg),
             'polAngle0Fit_deg':      toscalar(polAngle0Fit_deg),
             'dPolAngle0Fit_deg':     toscalar(dPolAngle0Fit_deg)}

    return mDict


#-----------------------------------------------------------------------------#
    
#-----------------------------------------------------------------------------#
def do_rmsynth_planes(dataQ, dataU, lambdaSqArr_m2, phiArr_radm2, 
                      weightArr=None, lam0Sq_m2=None, nBits=64, verbose=False,
                      log=print):
    """Perform RM-synthesis on Stokes Q and U cubes (1,2 or 3D). This version
    of the routine loops through spectral planes and is faster than the pixel-
    by-pixel code. This version also correctly deals with isolated clumps of
    NaN-flagged voxels within the data-cube (unlikely in interferometric cubes,
    but possible in single-dish cubes). Input data must be in standard python
    [z,y,x] order, where z is the frequency axis in ascending order.

    dataQ           ... 1, 2 or 3D Stokes Q data array
    dataU           ... 1, 2 or 3D Stokes U data array
    lambdaSqArr_m2  ... vector of wavelength^2 values (assending freq order)
    phiArr_radm2    ... vector of trial Faraday depth values
    weightArr       ... vector of weights, default [None] is Uniform (all 1s)
    nBits           ... precision of data arrays [32]
    verbose         ... print feedback during calculation [False]
    log             ... function to be used to output messages [print]
    
    """
    
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Set the weight array
    if weightArr is None:
        weightArr = np.ones(lambdaSqArr_m2.shape, dtype=dtFloat)
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)
    
    # Sanity check on array sizes
    if not weightArr.shape  == lambdaSqArr_m2.shape:
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
        log("Err: Data depth does not match lambda^2 vector ({} vs {}).".format(dataQ.shape[0], lambdaSqArr_m2.shape[0]), end=' ')
        log("     Check that data is in [z, y, x] order.")
        return None, None
    
    # Reshape the data arrays to 3 dimensions
    if nDims==1:
        dataQ = np.reshape(dataQ, (dataQ.shape[0], 1, 1))
        dataU = np.reshape(dataU, (dataU.shape[0], 1, 1))
    elif nDims==2:
        dataQ = np.reshape(dataQ, (dataQ.shape[0], dataQ.shape[1], 1))
        dataU = np.reshape(dataU, (dataU.shape[0], dataU.shape[1], 1))
    
    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY, nX]
    pCube = (dataQ + 1j * dataU) * weightArr[:, np.newaxis, np.newaxis]
    
    # Check for NaNs (flagged data) in the cube & set to zero
    mskCube = np.isnan(pCube)
    pCube = np.nan_to_num(pCube)
    
    # If full planes are flagged then set corresponding weights to zero
    mskPlanes =  np.sum(np.sum(~mskCube, axis=1), axis=1)
    mskPlanes = np.where(mskPlanes==0, 0, 1)
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
    weightCube =  np.invert(mskCube) * weightArr[:, np.newaxis, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
        KArr[KArr == np.inf] = 0
        KArr = np.nan_to_num(KArr)
        
    # Do the RM-synthesis on each plane
    if verbose:
        log("Running RM-synthesis by channel.")
        progress(40, 0)
    v_array = freq_array(lambdaSqArr_m2)
    delta_v = bandwidth(v_array)
    for i in range(nPhi):
        if verbose:
            progress(40, ((i+1)*100.0/nPhi))
        cor = np.exp(2j*phiArr_radm2[i]*lam0Sq_m2)
        r_i = rotation_op(delta_v, v_array, phiArr_radm2[i])[:, np.newaxis,np.newaxis]
        arg0 = pCube * cor* np.conj(r_i)
        arg = arg0
        FDFcube[i,:,:] =  KArr * np.sum(arg, axis=0) 
    
    # information to generate theoretical noise, sensitivity
    adjoint_varbs = [delta_v, v_array, phiArr_radm2, K, weightArr]
 
    # Remove redundant dimensions in the FDF array
    FDFcube = np.squeeze(FDFcube)
    return FDFcube, lam0Sq_m2, adjoint_varbs


#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
def run_rmsynth(data, polyOrd=3, phiMax_radm2=None, dPhi_radm2=None,
                nSamples=10.0, weightType="variance", fitRMSF=False,
                noStokesI=False, phiNoise_radm2=1e6, nBits=32, showPlots=False,
                debug=False, verbose=False, log=print,units='Jy/beam'):
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
    dtComplex = "complex" + str(2*nBits)

    # freq_Hz, I, Q, U, dI, dQ, dU
    try:
        if verbose: log("> Trying [freq_Hz, I, Q, U, dI, dQ, dU]", end=' ')
        (freqArr_Hz, IArr, QArr, UArr, dIArr, dQArr, dUArr) = data
        if verbose: log("... success.")
    except Exception:
        if verbose: log("...failed.")
        # freq_Hz, q, u, dq, du
        try:
            if verbose: log("> Trying [freq_Hz, q, u,  dq, du]", end=' ')
            (freqArr_Hz, QArr, UArr, dQArr, dUArr) = data
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
        if verbose: log("Warn: no Stokes I data in use.")
        IArr = np.ones_like(QArr)
        dIArr = np.zeros_like(QArr)

    # Convert to GHz for convenience
    freqArr_GHz = freqArr_Hz / 1e9
    dQUArr = (dQArr + dUArr)/2.0

    # Fit the Stokes I spectrum and create the fractional spectra
    IModArr, qArr, uArr, dqArr, duArr, fitDict = \
             create_frac_spectra(freqArr  = freqArr_GHz,
                                 IArr     = IArr,
                                 QArr     = QArr,
                                 UArr     = UArr,
                                 dIArr    = dIArr,
                                 dQArr    = dQArr,
                                 dUArr    = dUArr,
                                 polyOrd  = polyOrd,
                                 verbose  = True,
                                 debug    = debug)

    # Plot the data and the Stokes I model fit
    if showPlots:
        if verbose: log("Plotting the input data and spectral index fit.")
        freqHirArr_Hz =  np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)
        IModHirArr = poly5(fitDict["p"])(freqHirArr_Hz/1e9)
        specFig = plt.figure(figsize=(12.0, 8))
        plot_Ipqu_spectra_fig(freqArr_Hz     = freqArr_Hz,
                              IArr           = IArr,
                              qArr           = qArr,
                              uArr           = uArr,
                              dIArr          = dIArr,
                              dqArr          = dqArr,
                              duArr          = duArr,
                              freqHirArr_Hz  = freqHirArr_Hz,
                              IModArr        = IModHirArr,
                              fig            = specFig,
                              units          = units)
        
        
        
   

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
            ax.plot(freqArr_Hz/1e9, dQUArr, marker='o', color='k', lw=0.5,
                    label='rms <QU>')
            ax.plot(freqArr_Hz/1e9, dQArr, marker='o', color='b', lw=0.5,
                    label='rms Q')
            ax.plot(freqArr_Hz/1e9, dUArr, marker='o', color='r', lw=0.5,
                    label='rms U')
            xRange = (np.nanmax(freqArr_Hz)-np.nanmin(freqArr_Hz))/1e9
            ax.set_xlim( np.min(freqArr_Hz)/1e9 - xRange*0.05,
                         np.max(freqArr_Hz)/1e9 + xRange*0.05)
            ax.set_xlabel('$\\nu$ (GHz)')
            ax.set_ylabel('RMS '+units)
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
        weightArr = 1.0 / np.power(dQUArr, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    if verbose: log("Weight type is '%s'." % weightType)

    startTime = time.time()

    # Perform RM-synthesis on the spectrum
    dirtyFDF, lam0Sq_m2, adjoint_varbs= do_rmsynth_planes(dataQ           = qArr,
                                            dataU           = uArr,
                                            lambdaSqArr_m2  = lambdaSqArr_m2,
                                            phiArr_radm2    = phiArr_radm2,
                                            weightArr       = weightArr,
                                            nBits           = nBits,
                                            verbose         = verbose,
                                            log             = log)
    
  
    
    # generate adjoint_noise and adjoint__sens
    adjoint_info = adjoint_theory(adjoint_varbs, dQUArr, show_progress=False)
    phiArr_radm2, adjoint_sens, adjoint_noise = adjoint_info 
    
    # calculate peak RM 
    absFDF = np.abs(dirtyFDF)
    rm_fdf = absFDF / adjoint_noise # used for finding peak in RM
    indxPeakPIchan = np.nanargmax(rm_fdf[1:-1])+1
    peak_rm = phiArr_radm2[indxPeakPIchan]

    
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
                        verbose         = verbose,
                        log             = log,
                        peak_rm         = peak_rm)
    fwhmRMSF = float(fwhmRMSFArr)

    # ALTERNATE RM-SYNTHESIS CODE --------------------------------------------#

    #dirtyFDF, [phi2Arr_radm2, RMSFArr], lam0Sq_m2, fwhmRMSF = \
    #          do_rmsynth(qArr, uArr, lambdaSqArr_m2, phiArr_radm2, weightArr)

    #-------------------------------------------------------------------------#

    endTime = time.time()
    cputime = (endTime - startTime)
    if verbose: log("> RM-synthesis completed in %.2f seconds." % cputime)

    # Determine the Stokes I value at lam0Sq_m2 from the Stokes I model
    # Multiply the dirty FDF by Ifreq0 to recover the PI
    freq0_Hz = C / m.sqrt(lam0Sq_m2)
    Ifreq0 = poly5(fitDict["p"])(freq0_Hz/1e9)
    dirtyFDF *= (Ifreq0)    # FDF is in fracpol units initially, convert back to flux

    # Calculate the theoretical noise in the FDF !!Old formula only works for wariance weights!
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)
    dFDFth = np.sqrt( np.sum(weightArr**2 * np.nan_to_num(dQUArr)**2) / (np.sum(weightArr))**2 )


    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties
    
    mDict = measure_FDF_parms(FDF         = dirtyFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = fwhmRMSF,
                              adjoint_sens = adjoint_sens,
                              adjoint_noise = adjoint_noise,
                              dFDF        = dFDFth,
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = lam0Sq_m2)
    mDict["Ifreq0"] = toscalar(Ifreq0)
    mDict["polyCoeffs"] =  ",".join([str(x) for x in fitDict["p"]])
    mDict["IfitStat"] = fitDict["fitStatus"]
    mDict["IfitChiSqRed"] = fitDict["chiSqRed"]
    mDict["lam0Sq_m2"] = toscalar(lam0Sq_m2)
    mDict["freq0_Hz"] = toscalar(freq0_Hz)
    mDict["fwhmRMSF"] = toscalar(fwhmRMSF)
    mDict["dQU"] = toscalar(nanmedian(dQUArr))
    mDict["dFDFth"] = toscalar(dFDFth)
    mDict["units"] = units
   # mDict['dQUArr'] = dQUArr
    
    if fitDict["fitStatus"] >= 128:
        log("WARNING: Stokes I model contains negative values!")
    elif fitDict["fitStatus"] >= 64:
        log("Caution: Stokes I model has low signal-to-noise.")



    #Add information on nature of channels:
    good_channels=np.where(np.logical_and(weightArr != 0,np.isfinite(qArr)))[0]
    mDict["min_freq"]=float(np.min(freqArr_Hz[good_channels]))
    mDict["max_freq"]=float(np.max(freqArr_Hz[good_channels]))
    mDict["N_channels"]=good_channels.size
    mDict["median_channel_width"]=float(np.median(np.diff(freqArr_Hz)))

    # Measure the complexity of the q and u spectra
    mDict["fracPol"] = mDict["ampPeakPIfit"]/(Ifreq0)
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
    aDict["dirtyFDF"]=(dirtyFDF  / adjoint_sens)

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
       log('I freq0 = %.4g %s' % (mDict["Ifreq0"],units))
       log('Peak PI = %.4g (+/-%.4g) %s' % (mDict["ampPeakPIfit"],
                                                mDict["dAmpPeakPIfit"],units))
       log('QU Noise = %.4g %s' % (mDict["dQU"],units))
       log('FDF Noise (theory)   = %.4g %s' % (mDict["dFDFth"],units))
       log('FDF Noise (Corrected MAD) = %.4g %s' % (mDict["dFDFcorMAD"],units))
       log('FDF Noise (rms)   = %.4g %s' % (mDict["dFDFrms"],units))
       log('FDF SNR = %.4g ' % (mDict["snrPIfit"]))
       log('sigma_add(q) = %.4g (+%.4g, -%.4g)' % (mDict["sigmaAddQ"],
                                            mDict["dSigmaAddPlusQ"],
                                            mDict["dSigmaAddMinusQ"]))
       log('sigma_add(u) = %.4g (+%.4g, -%.4g)' % (mDict["sigmaAddU"],
                                            mDict["dSigmaAddPlusU"],
                                            mDict["dSigmaAddMinusU"]))
       log()
       log('-'*80)





    #fix len of RMSFArr
    #RMSFArr = np.reshape(RMSFArr, 1)
    #margin = len(dirtyFDF)-len(RMSFArr)
       
    # Plot the RM Spread Function and dirty FDF
       
   
    
    #RMSFArr = RMSFArr.reshape(-1)
    #phi2Arr_radm2 = phiArr_radm2
    if showPlots:
        plot_adjoint_info(adjoint_info, units=units)
        fdfFig = plt.figure(figsize=(12.0, 8))
        plot_rmsf_fdf_fig(phiArr     = phiArr_radm2,
                          FDF        = (dirtyFDF / adjoint_sens),
                          phi2Arr    = phiArr_radm2,
                          RMSFArr    = RMSFArr,
                          peak_rm    = peak_rm,
                          fwhmRMSF   = fwhmRMSF,
                          vLine      = mDict["phiPeakPIfit_rm2"],
                          fig        = fdfFig,
                          units      = units)
        

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
        #plot_adjoint_info(adjoint_info)
        plt.show()
       # plot_adjoint_info(adjoint_info)
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
    dtComplex = "complex" + str(2*nBits)

    # Output prefix is derived from the input file name


    # Read the data-file. Format=space-delimited, comments="#".
    if verbose: print("Reading the data file '%s':" % dataFile)
    # freq_Hz, I, Q, U, dI, dQ, dU
    try:
        if verbose: print("> Trying [freq_Hz, I, Q, U, dI, dQ, dU]", end=' ')
        (freqArr_Hz, IArr, QArr, UArr,
         dIArr, dQArr, dUArr) = \
         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
        if verbose: print("... success.")
        data=[freqArr_Hz, IArr, QArr, UArr, dIArr, dQArr, dUArr]
    except Exception:
        if verbose: print("...failed.")
        # freq_Hz, q, u, dq, du
        try:
            if verbose: print("> Trying [freq_Hz, q, u,  dq, du]", end=' ')
            (freqArr_Hz, QArr, UArr, dQArr, dUArr) = \
                         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
            if verbose: print("... success.")
            data=[freqArr_Hz, QArr, UArr, dQArr, dUArr]

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


    To get outputs, one or more of the following flags must be set: -S, -p, -v.
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
    #parser.add_argument("-b", dest="bit64", action="store_true",
                        #help="use 64-bit floating point precision [False (uses 32-bit)]")
    parser.add_argument("-p", dest="showPlots", action="store_true",
                        help="show the plots [False].")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="verbose output [False].")
    parser.add_argument("-S", dest="saveOutput", action="store_true",
                        help="save the arrays [False].")
    parser.add_argument("-D", dest="debug", action="store_true",
                        help="turn on debugging messages & plots [False].")
    parser.add_argument("-U", dest="units", type=str, default="Jy/beam",
                        help="Intensity units of the data. [Jy/beam]")
    parser.add_argument("-m", dest="mode_2", action="store_true",
                        help="shows theoretical sensitivity and noise plots only")
    args = parser.parse_args()
    
    # Sanity checks
    if not os.path.exists(args.dataFile[0]):
        print("File does not exist: '%s'." % args.dataFile[0])
        sys.exit()
    prefixOut, ext = os.path.splitext(args.dataFile[0])
    dataDir, dummy = os.path.split(args.dataFile[0])
    # Set the floating point precision
    nBits = 64
    #if args.bit64:
    #    nBits = 64
    verbose=args.verbose
    data = readFile(args.dataFile[0],nBits, verbose=verbose, debug=args.debug)

    mode_2 = args.mode_2
    
    if mode_2 is True:
        depolarization(data           = data,
                phiMax_radm2   = args.phiMax_radm2,
                polyOrd        = args.polyOrd,
                dPhi_radm2     = args.dPhi_radm2,
                nSamples       = args.nSamples,
                weightType     = args.weightType,
                noStokesI      = args.noStokesI,
                verbose        = verbose,
                units          = args.units)
        
    if mode_2 is False:

        # Run RM-synthesis on the spectra
        mDict, aDict = run_rmsynth(data           = data,
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
                    verbose        = verbose,
                    units          = args.units)

        if args.saveOutput:
            saveOutput(mDict, aDict, prefixOut, verbose)


#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
