#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     util_RM.py                                                        #
#                                                                             #
# PURPOSE:  Common procedures used with RM-synthesis scripts.                 #
#                                                                             #
# REQUIRED: Requires the numpy and scipy modules.                             #
#                                                                             #
# MODIFIED: 16-Nov-2018 by J. West                                            #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#  do_rmsynth_planes   ... perform RM-synthesis on Q & U data cubes           #
#  get_rmsf_planes     ... calculate the RMSF for a cube of data              #
#  do_rmclean_hogbom   ... perform Hogbom RM-clean on a dirty FDF             #
#  fits_make_lin_axis  ... create an array of absica values for a lin axis    #
#  extrap              ... interpolate and extrapolate an array               #
#  fit_rmsf            ... fit a Gaussian to the main lobe of the RMSF        #
#  gauss1D             ... return a function to evaluate a 1D Gaussian        #
#  detect_peak         ... detect the extent of a peak in a 1D array          #
#  measure_FDF_parms   ... measure parameters of a Faraday dispersion func    #
#  norm_cdf            ... calculate the CDF of a Normal distribution         #
#  cdf_percentile      ... return the value at the given percentile of a CDF  #
#  calc_sigma_add      ... calculate most likely additional scatter           #
#  calc_normal_tests   ... calculate metrics measuring deviation from normal  #
#  measure_qu_complexity  ... measure the complexity of a q & u spectrum      #
#  measure_fdf_complexity  ... measure the complexity of a clean FDF spectrum #
#                                                                             #
# DEPRECATED CODE ------------------------------------------------------------#
#                                                                             #
#  do_rmsynth          ... perform RM-synthesis on Q & U data by spectrum     #
#  get_RMSF            ... calculate the RMSF for a 1D wavelength^2 array     #
#  do_rmclean          ... perform Hogbom RM-clean on a dirty FDF             #
#  plot_complexity     ... plot the residual, PDF and CDF (deprecated)        #
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
import math as m
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import skewtest
from scipy.stats import kurtosistest
from scipy.stats import anderson
from scipy.stats import kstest
from scipy.stats import norm

from RMutils.mpfit import mpfit
from RMutils.util_misc import progress  
from RMutils.util_misc import toscalar
from RMutils.util_misc import calc_parabola_vertex
from RMutils.util_misc import create_pqu_spectra_burn
from RMutils.util_misc import calc_mom2_FDF
from RMutils.util_misc import MAD
from RMutils.util_misc import nanstd

# Constants
C = 2.99792458e8


#-----------------------------------------------------------------------------#
def do_rmsynth_planes(dataQ, dataU, lambdaSqArr_m2, phiArr_radm2, 
                      weightArr=None, lam0Sq_m2=None, nBits=32, verbose=False,
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
    a = lambdaSqArr_m2 - lam0Sq_m2
    for i in range(nPhi):
        if verbose:
            progress(40, ((i+1)*100.0/nPhi))
        arg = np.exp(-2.0j * phiArr_radm2[i] * a)[:, np.newaxis,np.newaxis]
        FDFcube[i,:,:] =  KArr * np.sum(pCube * arg, axis=0)
        
    # Remove redundant dimensions in the FDF array
    FDFcube = np.squeeze(FDFcube)

    return FDFcube, lam0Sq_m2


#-----------------------------------------------------------------------------#
def get_rmsf_planes(lambdaSqArr_m2, phiArr_radm2, weightArr=None, mskArr=None, 
                    lam0Sq_m2=None, double=True, fitRMSF=False,
                    fitRMSFreal=False, nBits=32, verbose=False,
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

        # Calculate the RMSF
        a = (-2.0 * 1j * phi2Arr).astype(dtComplex)
        b = (lambdaSqArr_m2 - lam0Sq_m2)
        RMSFArr = K * np.sum(weightArr * np.exp( np.outer(a, b) ), 1)
        
        # Fit the RMSF main lobe
        fitStatus = -1
        if fitRMSF:
            if verbose:
                log("Fitting Gaussian to the main lobe.")
            if fitRMSFreal:
                mp = fit_rmsf(phi2Arr, RMSFcube.real)
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

    # Calculate the RMSF at each pixel
    else:
        if verbose:
            log("Calculating RMSF by channel.")

        # The K value used to scale each RMSF must take into account
        # isolated flagged voxels data in the datacube
        weightCube =  np.invert(mskArr) * weightArr[:, np.newaxis, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
            KArr[KArr == np.inf] = 0
            KArr = np.nan_to_num(KArr)

        # Calculate the RMSF for each plane
        if verbose:
            progress(40, 0)
        a = (lambdaSqArr_m2 - lam0Sq_m2)
        for i in range(nPhi):
            if verbose:
                progress(40, ((i+1)*100.0/nPhi))
#            arg = np.exp(-2.0j * phi2Arr[i] * a)[:, np.newaxis, np.newaxis]
#            RMSFcube[i,:,:] =  KArr * np.sum(uCube * arg, axis=0)
            arg = np.exp(-2.0j * phi2Arr[i] * a)[:, np.newaxis, np.newaxis]
            RMSFcube[i,:,:] =  KArr * np.sum(weightCube * arg, axis=0)


        # Default to the analytical RMSF
        fwhmRMSFArr = np.ones((nY, nX), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nY, nX), dtype="int") * (-1)
    
        # Fit the RMSF main lobe
        if fitRMSF:
            if verbose:
                log("Fitting main lobe in each RMSF spectrum.")
                log("> This may take some time!")
            progress(40, 0)
            k = 0
            for i in range(nX):
                for j in range(nY):
                    k += 1
                    if verbose:
                        progress(40, (k*100.0/nPix))
                    if fitRMSFreal:
                        mp = fit_rmsf(phi2Arr, RMSFcube[:,j,i].real)
                    else:
                        mp = fit_rmsf(phi2Arr, np.abs(RMSFcube[:,j,i]))
                    if not (mp is None or mp.status<1):
                        fwhmRMSFArr[j,i] = mp.params[2]
                        statArr[j,i]  = mp.status
    
    # Remove redundant dimensions
    RMSFcube = np.squeeze(RMSFcube)
    fwhmRMSFArr = np.squeeze(fwhmRMSFArr)
    statArr = np.squeeze(statArr)
    
    return RMSFcube, phi2Arr, fwhmRMSFArr, statArr


#-----------------------------------------------------------------------------#
def do_rmclean_hogbom(dirtyFDF, phiArr_radm2, RMSFArr, phi2Arr_radm2,
                      fwhmRMSFArr, cutoff, maxIter=1000, gain=0.1,
                      mskArr=None, nBits=32, verbose=False, doPlots=False,
                      doAnimate=False):
    """Perform Hogbom CLEAN on a cube of complex Faraday dispersion functions
    given a cube of rotation measure spread functions.

    dirtyFDF       ... 1, 2 or 3D complex FDF array
    phiArr_radm2   ... 1D Faraday depth array corresponding to the FDF
    RMSFArr        ... 1, 2 or 3D complex RMSF array
    phi2Arr_radm2  ... double size 1D Faraday depth array of the RMSF
    fwhmRMSFArr    ... scalar, 1D or 2D array of RMSF main lobe widths
    cutoff         ... clean cutoff (+ve = absolute values, -ve = sigma) [-1]
    maxIter        ... maximun number of CLEAN loop interations [1000]
    gain           ... CLEAN loop gain [0.1]
    mskArr         ... scalar, 1D or 2D pixel mask array [None]
    nBits          ... precision of data arrays [32]
    verbose        ... print feedback during calculation [False]
    doPlots        ... plot the final CLEAN FDF [False]
    doAnimate      ... animate the CLEAN loop plots [False]

    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Function to plot FDF
    def plot_clean_spec(ax1, ax2, phiArr_radm2, dirtyFDF, ccArr, residFDF,
                        cutoff):
        ax1.cla()
        ax2.cla()
        ax1.step(phiArr_radm2, np.abs(dirtyFDF[:, yi, xi]),
                 color="grey",marker="None", mfc="w", mec="g", ms=10,
                 where="mid", label="Dirty FDF")
        ax1.step(phiArr_radm2, np.abs(ccArr[:, yi, xi]), color="g",
                 marker="None", mfc="w", mec="g", ms=10, where="mid",
                 label="Clean Components")
        ax1.step(phiArr_radm2, np.abs(residFDF[:, yi, xi]), color="magenta", 
                 marker="None", mfc="w", mec="g", ms=10, where="mid",
                 label="Residual FDF")
        ax1.step(phiArr_radm2, np.abs(cleanFDF[:, yi, xi]), color="k",
                 marker="None", mfc="w", mec="g", ms=10, where="mid", lw=1.5,
                 label="Clean FDF")
        ax1.axhline(cutoff, color="r", ls="--", label="Clean cutoff")
        ax1.yaxis.set_major_locator(MaxNLocator(4))
        ax1.set_ylabel("Flux Density")
        leg = ax1.legend(numpoints=1, loc='upper right', shadow=False,
                         borderaxespad=0.3, bbox_to_anchor=(1.00, 1.00))  
        for t in leg.get_texts():
            t.set_fontsize('small')       
        leg.get_frame().set_linewidth(0.5)        
        leg.get_frame().set_alpha(0.5)
        [label.set_visible(False) for label in ax1.get_xticklabels()]
        ax2.step(phiArr_radm2, np.abs(residFDF[:, yi, xi]), color="magenta",
                 marker="None", mfc="w", mec="g", ms=10, where="mid",
                 label="Residual FDF")
        ax2.step(phiArr_radm2, np.abs(ccArr[:, yi, xi]), color="g",
                 marker="None", mfc="w", mec="g", ms=10, where="mid",
                 label="Clean Components")
        ax2.axhline(cutoff, color="r", ls="--", label="Clean cutoff")
        ax2.set_ylim(0, cutoff*3.0)
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.set_ylabel("Flux Density")
        ax2.set_xlabel("$\phi$ rad m$^{-2}$")
        leg = ax2.legend(numpoints=1, loc='upper right', shadow=False,
                         borderaxespad=0.3, bbox_to_anchor=(1.00, 1.00))
        for t in leg.get_texts():
            t.set_fontsize('small') 
        leg.get_frame().set_linewidth(0.5)        
        leg.get_frame().set_alpha(0.5)
        ax2.autoscale_view(True,True,True)
        plt.draw()

    if doAnimate:
        doPlots = True
    
    # Sanity checks on array sizes
    nPhi = phiArr_radm2.shape[0]
    if nPhi != dirtyFDF.shape[0]:
        print("Err: 'phi2Arr_radm2' and 'dirtyFDF' are not the same length.")
        return None, None, None
    nPhi2 = phi2Arr_radm2.shape[0]
    if not nPhi2 == RMSFArr.shape[0]:
        print("Err: missmatch in 'phi2Arr_radm2' and 'RMSFArr' length.")
        return None, None, None
    if not (nPhi2 >= 2 * nPhi):
        print("Err: the Faraday depth of the RMSF must be twice the FDF.")
        return None, None, None
    nDims = len(dirtyFDF.shape)
    if not nDims <= 3:
        print("Err: FDF array dimensions must be <= 3.")
        return None, None, None
    if not nDims == len(RMSFArr.shape):
        print("Err: the input RMSF and FDF must have the same number of axes.")
        return None, None, None
    if not RMSFArr.shape[1:]==dirtyFDF.shape[1:]:
        print("Err: the xy dimesions of the RMSF and FDF must match.")
        return None, None, None
    if mskArr is not None:
        if not mskArr.shape==dirtyFDF.shape[1:]:
            print("Err: pixel mask must match xy dimesnisons of FDF cube.")
            print("     FDF[z,y,z] = {:}, Mask[y,x] = {:}.".format(dirtyFDF.shape, mskArr.shape), end=' ')
            
            return None, None, None
    else:
        mskArr = np.ones(dirtyFDF.shape[1:], dtype="bool") 
        
    # Reshape the FDF & RMSF array to 3 dimensions and mask array to 2
    if nDims==1:
        dirtyFDF = np.reshape(dirtyFDF, (dirtyFDF.shape[0], 1, 1))
        RMSFArr = np.reshape(RMSFArr, (RMSFArr.shape[0], 1, 1))
        mskArr = np.reshape(mskArr,  (1, 1))
        fwhmRMSFArr = np.reshape(fwhmRMSFArr,  (1, 1))
    elif nDims==2:
        dirtyFDF = np.reshape(dirtyFDF,list(dirtyFDF.shape[:2])+[1])
        RMSFArr = np.reshape(RMSFArr,list(RMSFArr.shape[:2])+[1])
        mskArr = np.reshape(mskArr,  (dirtyFDF.shape[1],1))
        fwhmRMSFArr = np.reshape(fwhmRMSFArr,  (dirtyFDF.shape[1],1))
    iterCountArr = np.zeros_like(mskArr, dtype="int")

    # Determine which pixels have components above the cutoff
    absFDF = np.abs(np.nan_to_num(dirtyFDF))
    mskCutoff = np.where(np.max(absFDF, axis=0)>=cutoff,1, 0)
    xyCoords = np.rot90(np.where(mskCutoff>0))

    # Feeback to user
    if verbose:
        nPix = dirtyFDF.shape[-1]* dirtyFDF.shape[-2]
        nCleanPix = len(xyCoords)
        print("Cleaning {:}/{:} spectra.".format(nCleanPix, nPix), end=' ')
        
    
    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    residFDF = dirtyFDF.copy()
    ccArr = np.zeros(dirtyFDF.shape, dtype=dtComplex)
    cleanFDF = np.zeros_like(dirtyFDF)
    
    # Plotting
    if doPlots:
        
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
        
        # Setup the figure to track the clean
        fig = plt.figure(figsize=(12.0, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        
        fig.show()

    # Loop through the pixels containing a polarised signal
    j = 0
    if verbose:
        pass  
        #progress(40, 0)  #This is currently broken...
    for yi, xi in xyCoords:
        if verbose:
            j += 1
            #progress(40, ((j)*100.0/nCleanPix))  #This is currently broken...

        # Find the index of the peak of the RMSF
        indxMaxRMSF = np.nanargmax(RMSFArr[:, yi, xi])

        # Calculate the padding in the sampled RMSF
        # Assumes only integer shifts and symmetric
        nPhiPad = int((len(phi2Arr_radm2)-len(phiArr_radm2))/2)

        # Main CLEAN loop
        iterCount = 0
        while ( np.max(np.abs(residFDF[:, yi, xi])) >= cutoff
                and iterCount <= maxIter ):
        
            # Get the absolute peak channel, values and Faraday depth
            indxPeakFDF = np.argmax(np.abs(residFDF[:, yi, xi]))
            peakFDFval = residFDF[indxPeakFDF, yi, xi]
            phiPeak = phiArr_radm2[indxPeakFDF]
        
            # A clean component is "loop-gain * peakFDFval
            CC = gain * peakFDFval
            ccArr[indxPeakFDF, yi, xi] += CC
        
            # At which channel is the CC located at in the RMSF?
            indxPeakRMSF = indxPeakFDF + nPhiPad
            
            # Shift the RMSF & clip so that its peak is centred above this CC
            shiftedRMSFArr = np.roll(RMSFArr[:, yi, xi],
                                 indxPeakRMSF-indxMaxRMSF)[nPhiPad:-nPhiPad]
        
            # Subtract the product of the CC shifted RMSF from the residual FDF
            residFDF[:, yi, xi] -= CC * shiftedRMSFArr
 
            # Restore the CC * a Gaussian to the cleaned FDF
            cleanFDF[:, yi, xi] += \
                gauss1D(CC, phiPeak, fwhmRMSFArr[yi, xi])(phiArr_radm2)
            iterCount += 1
            iterCountArr[yi, xi] = iterCount
            
            # Plot the progress of the clean
            if doAnimate:
                plot_clean_spec(ax1,
                                ax2,
                                phiArr_radm2,
                                dirtyFDF,
                                ccArr,
                                residFDF,
                                cutoff)


        if doPlots:
            plot_clean_spec(ax1,
                            ax2,
                            phiArr_radm2,
                            dirtyFDF,
                            ccArr,
                            residFDF,
                            cutoff)
            ax1.lines[2].remove()
            plt.draw()

    # Restore the residual to the CLEANed FDF (moved outside of loop: 
        #will now work for pixels/spectra without clean components)
    cleanFDF += residFDF


    # Remove redundant dimensions
    cleanFDF = np.squeeze(cleanFDF)
    ccArr = np.squeeze(ccArr)
    iterCountArr = np.squeeze(iterCountArr)
    
    return cleanFDF, ccArr, iterCountArr


#-----------------------------------------------------------------------------#
def fits_make_lin_axis(head, axis=0, dtype="f4"):
    """Create an array containing the axis values, assuming a simple linear
    projection scheme. Axis selection is zero-indexed."""
    
    axis = int(axis)    
    if head['NAXIS'] < axis + 1:
        return []
    
    i = str(int(axis) + 1)
    start = head['CRVAL' + i] + (1 - head['CRPIX' + i]) * head['CDELT' + i]
    stop = (head['CRVAL' + i] + (head['NAXIS' + i] - head['CRPIX' + i]) * 
            head['CDELT' + i])
    nChan = int(abs(start - stop)/head['CDELT' + i] +1)
    
    return np.linspace(start, stop, nChan).astype(dtype)
 

#-----------------------------------------------------------------------------#
def extrap(x, xp, yp):
    """
    Wrapper to allow np.interp to linearly extrapolate at function ends.
    
    np.interp function with linear extrapolation
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate
    -give-a-an-extrapolated-result-beyond-the-input-ran

    """
    y = np.interp(x, xp, yp)
    y = np.where(x < xp[0], yp[0]+(x-xp[0])*(yp[0]-yp[1])/(xp[0]-xp[1]), y)
    y = np.where(x > xp[-1], yp[-1]+(x-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2]),
                 y)
    return y


#-----------------------------------------------------------------------------#
def fit_rmsf(xData, yData, thresh=0.3, ampThresh=0.5):
    """
    Fit the main lobe of the RMSF with a Gaussian function. 
    """

    try:
        
        # Detect the peak and mask off the sidelobes
        msk1 = detect_peak(yData, thresh)
        msk2 = np.where(yData<ampThresh, 0.0, msk1)
        if sum(msk2)<4:
            msk2 = msk1
        validIndx = np.where(msk2==1.0)
        xData = xData[validIndx]
        yData = yData[validIndx]
        
        # Estimate starting parameters
        a = 1.0
        b = xData[np.argmax(yData)]
        w = np.nanmax(xData)-np.nanmin(xData)
        inParms=[ {'value': a, 'fixed':False, 'parname': 'amp'},
                  {'value': b, 'fixed':False, 'parname': 'offset'},
                  {'value': w, 'fixed':False, 'parname': 'width'}]

        # Function which returns another function to evaluate a Gaussian
        def gauss(p):
            a, b, w = p
            gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
            s = w / gfactor    
            def rfunc(x):
                y = a * np.exp(-(x-b)**2.0 /(2.0 * s**2.0))
                return y
            return rfunc
    
        # Function to evaluate the difference between the model and data.
        # This is minimised in the least-squared sense by the fitter
        def errFn(p, fjac=None):
            status = 0
            return status, gauss(p)(xData) - yData
    
        # Use mpfit to perform the fitting
        mp = mpfit(errFn, parinfo=inParms, quiet=True)

        return mp
    
    except Exception:
        return None


#-----------------------------------------------------------------------------#
def gauss1D(amp=1.0, mean=0.0, fwhm=1.0):
    """Function which returns another function to evaluate a Gaussian"""

    gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
    sigma = fwhm / gfactor    
    def rfunc(x):
        return amp * np.exp(-(x-mean)**2.0 /(2.0 * sigma**2.0))
    return rfunc


#-----------------------------------------------------------------------------#
def detect_peak(a, thresh=0.3):
    """Detect the extent of the peak in the array by moving away, in both
    directions, from the peak channel amd looking for where the slope changes
    to some shallow value. The triggering slope is 'thresh*max(slope)'.
    Returns a mask array like the input array with 1s over the extent of the
    peak and 0s elsewhere."""

    # Find the peak and take the 1st derivative
    iPkL= np.argmax(a)  # If the peak is flat, this is the left index
    g1 = np.abs(np.gradient(a))

    # Specify a threshold for the 1st derivative. Channels between the peak
    # and the first crossing point will be included in the mask.
    threshPos = np.nanmax(g1) * thresh

    # Determine the right-most index of flat peaks
    iPkR = iPkL
    d = np.diff(a)
    flatIndxLst = np.argwhere(d[iPkL:]==0).flatten()
    if len(flatIndxLst)>0:
        iPkR += (np.max(flatIndxLst)+1)
        
    # Search for the left & right crossing point
    iL = np.max(np.argwhere(g1[:iPkL]<=threshPos).flatten())
    iR = iPkR + np.min(np.argwhere(g1[iPkR+1:]<=threshPos).flatten()) + 2
    msk = np.zeros_like(a)
    msk[iL:iR] = 1
    
    # DEBUG PLOTTING
    if False:
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(np.arange(len(a)),a, where="mid", label="arr")
        ax.step(np.arange(len(g1)), np.abs(g1), where="mid", label="g1")
        ax.step(np.arange(len(msk)), msk*0.5, where="mid", label="msk")
        ax.axhline(0, color='grey')
        ax.axvline(iPkL, color='k', linewidth=3.0)
        ax.axhline(threshPos, color='magenta', ls="--")
        ax.set_xlim([iPkL-20, iPkL+20])
        leg = ax.legend(numpoints=1, loc='upper right', shadow=False,
                        borderaxespad=0.3, ncol=1,
                        bbox_to_anchor=(1.00, 1.00))
        fig.show()
        input()

    return msk


#-----------------------------------------------------------------------------#
def measure_FDF_parms(FDF, phiArr, fwhmRMSF, dFDF=None, lamSqArr_m2=None,
                      lam0Sq=None, snrDoBiasCorrect=5.0):
    """
    Measure standard parameters from a complex Faraday Dispersion Function.
    Currently this function assumes that the noise levels in the Stokes Q
    and U spectra are the same.
    """
    
    # Determine the peak channel in the FDF, its amplitude and index
    absFDF = np.abs(FDF)
    ampPeakPIchan = np.nanmax(absFDF)
    indxPeakPIchan = np.nanargmax(absFDF)

    # Measure the RMS noise in the spectrum after masking the peak
    dPhi = np.nanmin(np.diff(phiArr))
    fwhmRMSF_chan = np.ceil(fwhmRMSF/dPhi)
    iL = int(max(0, indxPeakPIchan-fwhmRMSF_chan*2))
    iR = int(min(len(absFDF), indxPeakPIchan+fwhmRMSF_chan*2))
    absFDFmsked = absFDF.copy()
    absFDFmsked[iL:iR] = np.nan
    absFDFmsked = absFDFmsked[np.where(absFDFmsked==absFDFmsked)]
    if float(len(absFDFmsked))/len(absFDF)<0.3:
        dFDFcorMAD_Jybm = MAD(absFDF)
        dFDFrms_Jybm = np.sqrt( np.mean(absFDF**2) )
    else:
        dFDFcorMAD_Jybm = MAD(absFDFmsked)
        dFDFrms_Jybm = np.sqrt( np.mean(absFDFmsked**2) )

    # Default to using the measured FDF if a noise value has not been provided
    if dFDF is None:
        dFDF = dFDFcorMAD_Jybm
    
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
                                         peakFDFrealChan))
    dPolAngleChan_deg = np.degrees(dFDF / (2.0 * ampPeakPIchan))

    # Calculate the derotated polarisation angle and uncertainty
    polAngle0Chan_deg = np.degrees(np.radians(polAngleChan_deg) -
                                  phiPeakPIchan * lam0Sq)
    nChansGood = np.sum(np.where(lamSqArr_m2==lamSqArr_m2, 1.0, 0.0))
    varLamSqArr_m2 = (np.sum(lamSqArr_m2**2.0) -
                      np.sum(lamSqArr_m2)**2.0/nChansGood) / (nChansGood-1)
    dPolAngle0Chan_rad = \
        np.sqrt( dFDF**2.0 / (4.0*(nChansGood-2.0)*ampPeakPIchan**2.0) *
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
                                           absFDF[indxPeakPIchan-1],
                                           phiArr[indxPeakPIchan],
                                           absFDF[indxPeakPIchan],
                                           phiArr[indxPeakPIchan+1],
                                           absFDF[indxPeakPIchan+1])
        
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
                                                  peakFDFrealFit))
        dPolAngleFit_deg = np.degrees(dFDF / (2.0 * ampPeakPIfit))

        # Calculate the derotated polarisation angle and uncertainty
        # Uncertainty from Eqn A.20 in Brentjens & De Bruyn 2005
        polAngle0Fit_deg = (np.degrees(np.radians(polAngleFit_deg) -
                                      phiPeakPIfit * lam0Sq)) % 180.0
        dPolAngle0Fit_rad = \
            np.sqrt( dFDF**2.0 / (4.0*(nChansGood-2.0)*ampPeakPIfit**2.0) *
                    ((nChansGood-1)/nChansGood + lam0Sq**2.0/varLamSqArr_m2) )
        dPolAngle0Fit_deg = np.degrees(dPolAngle0Fit_rad)

    # Store the measurements in a dictionary and return
    mDict = {'dFDFcorMAD_Jybm':       toscalar(dFDFcorMAD_Jybm),
             'dFDFrms_Jybm':          toscalar(dFDFrms_Jybm),
             'phiPeakPIchan_rm2':     toscalar(phiPeakPIchan),
             'dPhiPeakPIchan_rm2':    toscalar(dPhiPeakPIchan),
             'ampPeakPIchan_Jybm':    toscalar(ampPeakPIchan),
             'ampPeakPIchanEff_Jybm': toscalar(ampPeakPIchanEff),
             'dAmpPeakPIchan_Jybm':   toscalar(dFDF),
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
             'ampPeakPIfit_Jybm':     toscalar(ampPeakPIfit),
             'ampPeakPIfitEff_Jybm':  toscalar(ampPeakPIfitEff),
             'dAmpPeakPIfit_Jybm':    toscalar(dFDF),
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
def norm_cdf(mean=0.0, std=1.0, N=50, xArr=None):
    """Return the CDF of a normal distribution between -6 and 6 sigma, or at
    the values of an input array."""
    
    if xArr is None:
        x = np.linspace(-6.0*std, 6.0*std, N)
    else:
        x = xArr
    y = norm.cdf(x, loc=mean, scale=std)
    
    return x, y


#-----------------------------------------------------------------------------#
def cdf_percentile(x, p, q=50.0):
    """Return the value at a given percentile of a cumulative distribution
    function."""

    # Determine index where cumulative percentage is achieved
    i = np.where(p>q/100.0)[0][0]

    # If at extremes of the distribution, return the limiting value
    if i==0 or i==len(x):
        return x[i]

    # or interpolate between the two bracketing values in the CDF
    else:
        m = (p[i]-p[i-1])/(x[i]-x[i-1])
        c = p[i] - m*x[i]
        return (q/100.0-c)/m
    

#-----------------------------------------------------------------------------#
def calc_sigma_add(xArr, yArr, dyArr, yMed=None, noise=None, nSamp=1000,
                   suffix=""):
    """Calculate the most likely value of additional scatter, assuming the
    input data is drawn from a normal distribution. The total uncertainty on
    each data point Y_i is modelled as dYtot_i**2 = dY_i**2 + dYadd**2."""
    
    # Measure the median and MADFM of the input data if not provided.
    # Used to overplot a normal distribution when debugging.
    if yMed is None:
        yMed = np.median(yArr)
    if noise is None:
        noise = MAD(yArr)

    # Sample the PDF of the additional noise term from a limit near zero to
    # a limit of the range of the data, including error bars
    yRng = np.nanmax(yArr+dyArr) - np.nanmin(yArr-dyArr)
    sigmaAddArr = np.linspace(yRng/nSamp, yRng, nSamp)
    
    # Model deviation from Gaussian as an additional noise term.
    # Loop through the range of i additional noise samples and calculate
    # chi-squared and sum(ln(sigma_total)), used later to calculate likelihood.
    nData = len(xArr)
    chiSqArr = np.zeros_like(sigmaAddArr)
    lnSigmaSumArr = np.zeros_like(sigmaAddArr)
    for i, sigmaAdd in enumerate(sigmaAddArr):
        sigmaSqTot = dyArr**2.0 + sigmaAdd**2.0
        lnSigmaSumArr[i] = np.nansum(np.log(np.sqrt(sigmaSqTot)))
        chiSqArr[i] = np.nansum((yArr-yMed)**2.0/sigmaSqTot)
    dof = nData-1
    chiSqRedArr = chiSqArr/dof

    # Calculate the PDF in log space and normalise the peak to 1
    lnProbArr = (-np.log(sigmaAddArr) -nData * np.log(2.0*np.pi)/2.0
                 -lnSigmaSumArr -chiSqArr/2.0)
    lnProbArr -= np.nanmax(lnProbArr)
    probArr = np.exp(lnProbArr)
    
    # Normalise the area under the PDF to be 1
    A = np.nansum(probArr * np.diff(sigmaAddArr)[0])
    probArr /= A
    
    # Calculate the cumulative PDF
    CPDF = np.cumsum(probArr)/np.nansum(probArr)

    # Calculate the mean of the distribution and the +/- 1-sigma limits
    sigmaAdd = cdf_percentile(x=sigmaAddArr, p=CPDF, q=50.0)
    sigmaAddMinus = cdf_percentile(x=sigmaAddArr, p=CPDF, q=15.72)
    sigmaAddPlus = cdf_percentile(x=sigmaAddArr, p=CPDF, q=84.27)
    mDict = {"sigmaAdd" + suffix:  toscalar(sigmaAdd),
             "dSigmaAddMinus" + suffix: toscalar(sigmaAdd - sigmaAddMinus),
             "dSigmaAddPlus" + suffix: toscalar(sigmaAddPlus - sigmaAdd)}

    # Return the curves to be plotted in a separate dictionary
    pltDict = {"sigmaAddArr" + suffix: sigmaAddArr,
               "chiSqRedArr" + suffix: chiSqRedArr,
               "probArr" + suffix: probArr,
               "xArr" + suffix: xArr,
               "yArr" + suffix: yArr,
               "dyArr" + suffix: dyArr}
    
    # DEBUG PLOTS
    if False:

        # Setup for the figure
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18.0, 10.0))
        
        # Plot the data and the +/- 1-sigma levels
        ax1 = fig.add_subplot(231)
        ax1.errorbar(x=xArr, y=yArr, yerr=dyArr, ms=4, fmt='o')
        ax1.axhline(yMed, color='grey', zorder=10)
        ax1.axhline(yMed+noise, color='r', linestyle="--", zorder=10)
        ax1.axhline(yMed-noise, color='r', linestyle="--", zorder=10)
        ax1.set_title(r'Input Data')
        ax1.set_xlabel(r'$\lambda^2$')
        ax1.set_ylabel('Amplitude')

        # Plot the histogram of the data overlaid by the normal distribution
        H = 1.0/ np.sqrt(2.0 * np.pi * noise**2.0)
        xNorm = np.linspace(yMed-3*noise, yMed+3*noise, 1000)
        yNorm = H * np.exp(-0.5 * ((xNorm-yMed)/noise)**2.0)
        fwhm = noise * (2.0 * np.sqrt(2.0 * np.log(2.0)))
        ax2 = fig.add_subplot(232)
        nBins = 15
        n, b, p = ax2.hist(yArr, nBins, normed=1, histtype='step')
        ax2.plot(xNorm, yNorm, color='k', linestyle="--", linewidth=2)
        ax2.axvline(yMed, color='grey', zorder=11)
        ax2.axvline(yMed+fwhm/2.0, color='r', linestyle="--", zorder=11)
        ax2.axvline(yMed-fwhm/2.0, color='r', linestyle="--", zorder=11)
        ax2.set_title(r'Distribution of Data Compared to Normal')
        ax2.set_xlabel(r'Amplitude')
        ax2.set_ylabel(r'Normalised Counts')
    
        # Plot the ECDF versus a normal CDF
        ecdfArr = np.array(list(range(nData)))/float(nData)
        ySrtArr = np.sort(yArr)
        ax3 = fig.add_subplot(233)
        ax3.step(ySrtArr, ecdfArr, where="mid")
        x, y = norm_cdf(mean=yMed, std=noise, N=1000)
        ax3.plot(x, y, color='k', linewidth=2, linestyle="--", zorder=1)
        ax3.set_title(r'CDF of Data Compared to Normal')
        ax3.set_xlabel(r'Amplitude')
        ax3.set_ylabel(r'Normalised Counts')

        # Plot reduced chi-squared
        ax4 = fig.add_subplot(234)
        ax4.step(x=sigmaAddArr, y=chiSqRedArr, linewidth=1.5, where="mid")
        ax4.axhline(1.0, color='r', linestyle="--")
        ax4.set_title(r'$\chi^2_{\rm reduced}$ vs $\sigma_{\rm additional}$')
        ax4.set_xlabel(r'$\sigma_{\rm additional}$')
        ax4.set_ylabel(r'$\chi^2_{\rm reduced}$')

        # Plot the probability distribution function
        ax5 = fig.add_subplot(235)
        ax5.step(x=sigmaAddArr, y=probArr, linewidth=1.5, where="mid")
        ax5.axvline(sigmaAdd, color='grey', linestyle="-", linewidth=1.5)
        ax5.axvline(sigmaAddMinus, color='r', linestyle="--", linewidth=1.0)
        ax5.axvline(sigmaAddPlus, color='r', linestyle="--", linewidth=1.0)
        ax5.set_title('Relative Likelihood')
        ax5.set_xlabel(r"$\sigma_{\rm additional}$")
        ax5.set_ylabel(r"P($\sigma_{\rm additional}$|data)")

        # Plot the CPDF
        ax6 = fig.add_subplot(236)
        ax6.step(x=sigmaAddArr, y=CPDF, linewidth=1.5, where="mid")
        ax6.set_ylim(0, 1.05)
        ax6.axvline(sigmaAdd, color='grey', linestyle="-", linewidth=1.5)
        ax6.axhline(0.5, color='grey', linestyle="-", linewidth=1.5)
        ax6.axvline(sigmaAddMinus, color='r', linestyle="--", linewidth=1.0)
        ax6.axvline(sigmaAddPlus, color='r', linestyle="--", linewidth=1.0)
        ax6.set_title('Cumulative Likelihood')
        ax6.set_xlabel(r"$\sigma_{\rm additional}$")
        ax6.set_ylabel(r"Cumulative Likelihood")

        # Zoom in
        ax5.set_xlim(0, sigmaAdd + (sigmaAddPlus-sigmaAdd)*4.0)
        ax6.set_xlim(0, sigmaAdd + (sigmaAddPlus-sigmaAdd)*4.0)
        
        # Show the figure
        fig.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.94,
                            wspace=0.25, hspace=0.25)
        fig.show()
        
        # Feedback to user
        print("sigma_add(q) = %.4g (+%3g, -%3g)" %
              (mDict["sigmaAddQ"], mDict["dSigmaAddPlusQ"],
               mDict["dSigmaAddMinusQ"]))
        print("sigma_add(u) = %.4g (+%3g, -%3g)" %
              (mDict["sigmaAddU"], mDict["dSigmaAddPlusU"],
               mDict["dSigmaAddMinusU"]))
        input()
            
    return mDict, pltDict


#-----------------------------------------------------------------------------#
def calc_normal_tests(inArr, suffix=""):
    """Calculate metrics measuring deviation of an array from Normal."""

    # Perfrorm the KS-test
    KS_z, KS_p = kstest(inArr, "norm")
        
    # Calculate the Anderson test
    AD_z, AD_crit, AD_sig = anderson(inArr, "norm")
    
    # Calculate the skewness (measure of symmetry)
    # abs(skewness) < 0.5 =  approx symmetric
    skewVal = skew(inArr)
    SK_z, SK_p = skewtest(inArr)
    
    # Calculate the kurtosis (tails compared to a normal distribution)
    kurtosisVal = kurtosis(inArr)
    KUR_z, KUR_p = kurtosistest(inArr)

    # Return dictionary
    mDict = {"KSz" + suffix:  toscalar(KS_z),
             "KSp" + suffix:  toscalar(KS_p),
             #"ADz" + suffix: toscalar(AD_z),
             #"ADcrit" + suffix: toscalar(AD_crit),
             #"ADsig" + suffix: toscalar(AD_sig),
             "skewVal" + suffix: toscalar(skewVal),
             "SKz" + suffix: toscalar(SK_z),
             "SKp" + suffix: toscalar(SK_p),
             "kurtosisVal" + suffix: toscalar(kurtosisVal),
             "KURz" + suffix: toscalar(KUR_z),
             "KURp" + suffix: toscalar(KUR_p)
    }

    return mDict


#-----------------------------------------------------------------------------#
def measure_qu_complexity(freqArr_Hz, qArr, uArr, dqArr, duArr, fracPol,
                          psi0_deg, RM_radm2, specF=1):
    
    # Create a RM-thin model to subtract
    pModArr, qModArr, uModArr = \
             create_pqu_spectra_burn(freqArr_Hz   = freqArr_Hz,
                                     fracPolArr   = [fracPol],
                                     psi0Arr_deg  = [psi0_deg],
                                     RMArr_radm2  = [RM_radm2])
    lamSqArr_m2 = np.power(C/freqArr_Hz, 2.0)
    ndata = len(lamSqArr_m2)
    
    # Subtract the RM-thin model to create a residual q & u
    qResidArr = qArr - qModArr
    uResidArr = uArr - uModArr

    # Calculate value of additional scatter term for q & u (max likelihood)
    mDict = {}
    pDict = {}
    m1D, p1D = calc_sigma_add(xArr=lamSqArr_m2[:int(ndata/specF)],
                              yArr=(qResidArr/dqArr)[:int(ndata/specF)],
                              dyArr=(dqArr/dqArr)[:int(ndata/specF)],
                              yMed=0.0,
                              noise=1.0,
                              suffix="Q")
    mDict.update(m1D)
    pDict.update(p1D)    
    m2D, p2D = calc_sigma_add(xArr=lamSqArr_m2[:int(ndata/specF)],
                              yArr=(uResidArr/duArr)[:int(ndata/specF)],
                              dyArr=(duArr/duArr)[:int(ndata/specF)],
                              yMed=0.0,
                              noise=1.0,
                              suffix="U")
    mDict.update(m2D)
    pDict.update(p2D)
    
    # Calculate the deviations statistics
    # Done as a test for the paper, not usually offered to user.
    #mDict.update( calc_normal_tests(qResidArr/dqArr, suffix="Q") )
    #mDict.update( calc_normal_tests(uResidArr/duArr, suffix="U") )
    
    return mDict, pDict


#-----------------------------------------------------------------------------#
def measure_fdf_complexity(phiArr, FDF):

    # Second moment of clean component spectrum
    mom2FDF = calc_mom2_FDF(FDF, phiArr)

    return toscalar(mom2FDF)


#=============================================================================#
#                         DEPRECATED CODE BELOW                               #
#=============================================================================#


#-----------------------------------------------------------------------------#
def do_rmsynth(dataQ, dataU, lamSqArr, phiArr, weight=None, dtype='float32'):
    """Perform RM-synthesis on Stokes Q and U cubes. This version operates on
    data in spectral order, i.e., [zyx] => [yxz] (np.transpose(dataQ, (1,2,0)).

    *** Depricated by the faster 'do_rmsynth_planes' routine, above. ***
    
    """
    
    # Parse the weight argument
    if weight is None:
        weightArr = np.ones(lamSqArr.shape, dtype=dtype)
    else:
        weightArr = np.array(weight, dtype=dtype)
    
    # Sanity check on array sizes
    if not weightArr.shape  == lamSqArr.shape:
        print("Err: Lambda^2 and weight arrays must be the same shape.")
        return None, [None, None], None, None
    if not dataQ.shape == dataU.shape:
        print("Err: Stokes Q and U data arrays must be the same shape.")
        return None, [None, None], None, None
    nDims = len(dataQ.shape)
    if not nDims <= 3:
        print("Err: data-dimensions must be <= 3.")
        return None, [None, None], None, None
    if not dataQ.shape[-1] == lamSqArr.shape[-1]:
        print("Err: The Stokes Q and U arrays mush be in spectral order.")
        print("     # Stokes = %d, # Lamda = %d.", end=' ')
        (dataQ.shape[-1], lamSqArr.shape[-1])
        return None, [None, None], None, None

    # Reshape data arrays to allow the same recipies to work on all
    if nDims==1:
        dataQ = np.reshape(dataQ, (1, 1, dataQ.shape[-1]))
        dataU = np.reshape(dataU, (1, 1, dataU.shape[-1]))
    elif nDims==2:
        dataQ = np.reshape(dataQ, (1, dataQ.shape[-2], dataQ.shape[-1]))
        dataU = np.reshape(dataU, (1, dataU.shape[-2], dataU.shape[-1]))

    # Create a blanking mask assuming NaNs are blanked in the input data
    # Set the weight = 0 in fully blanked planes
    dataMsk = np.where(np.isnan(dataQ) + np.isnan(dataU), 0, 1)
    dataMsk = np.where(np.sum(np.sum(dataMsk, 0), 0)==0, 1, 0)
    weightArr = np.where(dataMsk==1, 0.0, weightArr)
    del dataMsk
    
    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Cube has dimensions (nY, nX, nFreq)
    pCube = (dataQ + 1j * dataU) * weightArr
    
    # Initialise the complex Faraday Dispersion Function (FDF) cube
    # Remember, python index order is reversed [2,1,0] = [y,x,phi]
    nY = dataQ.shape[0]
    nX = dataQ.shape[1]
    nPhi = phiArr.shape[0]
    FDFcube = np.ndarray((nY, nX, nPhi), dtype='complex')
    
    # B&dB equations (24) and (38) give the inverse sum of the weights
    # Get the weighted mean of the LambdaSq distribution (B&dB Eqn. 32)
    K = 1.0 / np.nansum(weightArr)
    lam0Sq_m2 = K * np.nansum(weightArr * lamSqArr)
    
    # Mininize the number of inner-loop operations by calculating the
    # argument of the EXP term in B&dB Eqns. (25) and (36) for the FDF
    # Returned array has dimensions (nPhi x nFreq)
    a = (-2.0 * 1j * phiArr)
    b = (lamSqArr - lam0Sq_m2) 
    arg = np.exp( np.outer(a, b) )

    # Do the synthesis at each pixel of the image
    nPix = nX * nY
    j = 0
    progress(40, 0)
    for k in range(nY):
        for i in range(nX):
            j += 1
            progress(40, ((j)*100.0/nPix))
                
            # Calculate the FDF, B&dB Eqns. (25) and (36)
            # B&dB Eqns. (25) and (36)
            FDFcube[k,i,:] = K * np.nansum(pCube[k,i,:] * arg, axis=1)
            
    # Calculate the complex Rotation Measure Spread Function
    RMSFArr, phiSamp, fwhmRMSF = get_RMSF(lamSqArr, phiArr, weightArr,
                                          lam0Sq_m2)

    # Reshape the data and FDF cube back to their original shapes
    if nDims==1:
        dataQ = np.reshape(dataQ, (dataQ.shape[-1]))
        dataU = np.reshape(dataU, (dataU.shape[-1]))
        FDFcube = np.reshape(FDFcube, (FDFcube.shape[-1]))
    elif nDims==2:
        dataQ = np.reshape(dataQ, (dataQ.shape[-2], dataQ.shape[-1]))
        dataU = np.reshape(dataU, (dataU.shape[-2], dataU.shape[-1]))
        FDFcube = np.reshape(FDFcube, (FDFcube.shape[-2], FDFcube.shape[-1]))
    
    return FDFcube, [phiSamp, RMSFArr], lam0Sq_m2, fwhmRMSF


#-----------------------------------------------------------------------------#
def get_RMSF(lamSqArr, phiArr, weightArr=None, lam0Sq_m2=None, double=True,
             fitRMSFreal=False, dtype="float32"):
    """Calculate the RMSF from 1D wavelength^2 and Faraday depth arrays.
    
    *** Depricated by the faster 'get_rmsf_planes' routine, above. ***

    """

    # Set the weight array
    if weightArr is None:
        uniformWt = True
        weightArr = np.ones(lamSqArr.shape, dtype=dtype)
    else:
        uniformWt = False
        weightArr = np.array(weightArr, dtype=dtype)
            
    # lam0Sq is the weighted mean of the LambdaSq distribution (B&dB Eqn. 32)
    K = 1.0 / np.nansum(weightArr)
    if lam0Sq_m2 is None:
        lam0Sq_m2 = K * np.nansum(weightArr * lamSqArr)

    # For cleaning the RMSF should extend by 1/2 on each side in phi-space
    if double:
        nPhi = phiArr.shape[0]
        nExt = np.ceil(nPhi/2.0)
        resampIndxArr = np.arange(2.0 * nExt + nPhi) - nExt
        phi2Arr = extrap(resampIndxArr, np.arange(nPhi, dtype='int'), phiArr)
    else:
        phi2Arr = phiArr
        
    # Calculate the RM spread function
    a = (-2.0 * 1j * phi2Arr)
    b = (lamSqArr - lam0Sq_m2) 
    RMSFArr = K * np.nansum(weightArr * np.exp( np.outer(a, b) ), 1)

    # Calculate (B&dB Equation 61) or fit the main-lobe FWHM of the RMSF
    fwhmRMSF = 2.0 * m.sqrt(3.0)/(np.nanmax(lamSqArr) - np.nanmin(lamSqArr))
    if not uniformWt:
        if fitRMSFreal:
            mp = fit_rmsf(phi2Arr, RMSFArr.real)
        else:
            mp = fit_rmsf(phi2Arr, np.abs(RMSFArr))
        if mp is None or mp.status<1:
            pass
            print("Err: failed to fit the RMSF.")
            print("Defaulting to analytical value in uniform case.")
        else:
            fwhmRMSF = mp.params[2]
            
    return RMSFArr, phi2Arr, fwhmRMSF


#-----------------------------------------------------------------------------#
def do_rmclean(dirtyFDF, phiArr, lamSqArr, cutoff, maxIter=1000, gain=0.1,
               weight=None, RMSFArr=None, RMSFphiArr=None, fwhmRMSF=None,
               fitRMSFreal=False, dtype='float32', doPlots=True):
    """Perform Hogbom (Heald) clean on a single RM spectrum.

        *** Depricated by the 'do_rmsynth_hogbom' routine, above. ***"""
    
    # Initial sanity checks --------------------------------------------------#
   
    # Check that dirtyFDF is ID and get its length
    if len(dirtyFDF.shape) != 1:
        print("Err: the dirty FDF is not a 1D array.")
        sys.exit(1)
    nFDF = dirtyFDF.shape[0]

    # Check that dirtyFDF is a complex array
    if not np.iscomplexobj(dirtyFDF):
        print("Err: the dirty FDF is not a complex array.")
        sys.exit(1)
        
    # Check that phiArr is 1D and get its length
    if len(phiArr.shape) != 1:
        print("Err: the phi array is not a 1D array.")
        sys.exit(1)
    nPhi = phiArr.shape[0]

    # Check that the lamSqArr is 1D and get its length
    if len(lamSqArr.shape) != 1:
        print("Err: the lamSqArr array is not a 1D array.")
        sys.exit(1)
    nlamSq = lamSqArr.shape[0]

    # Check that phiArr and FDF arrays are the same length
    if nPhi != nFDF:
        print('Err: the phiArr and dirty FDF are not the same length.')
        sys.exit(1)
    
    # If the RMSF has been passed in then check for correct formatting:
    #  - Twice the number of channels as dirtyFDF
    #  - Must be complex
    if not RMSFArr is None:

        # Check 1D
        if len(RMSFArr.shape) != 1:
            print("Err: input RMSF must be a 1D array.")
            sys.exit(1)
        nRMSF = RMSFArr.shape[0]
        
        # Check complex
        if not np.iscomplexobj(RMSFArr):
            print("Err: the RMSF is not a complex array.")
            sys.exit(1)
    
        # Check RMSF is at least double the FDF spectrum
        if not (nRMSF >= 2 * nFDF):
            print('Err: the RMSF must be twice the length of the FDF.')
            sys.exit(1)

        # Check that phiSampArr is also present and the same length
        if RMSFphiArr is None:
            print('Err: the phi sampling array must be passed with the RMSF.')
            sys.exit(1)
        nRMSFphi = RMSFphiArr.shape[0]
        if not nRMSF==nRMSFphi:
            print('Err: the RMSF and phi sampling array must be equal length.')
            sys.exit(1)
            
        # Calculate or fit the main-lobe FWHM of the RMSF
        # B&dB Equation (61)
        fwhmRMSF = 2.0 * m.sqrt(3.0)/(np.nanmax(lamSqArr) -
                                  np.nanmin(lamSqArr))
        if fitRMSFreal:
            mp = fit_rmsf(RMSFphiArr, RMSFArr.real)
        else:
            mp = fit_rmsf(RMSFphiArr, np.abs(RMSFArr))
        if mp is None or mp.status<1:
            pass
            print('Err: failed to fit the RMSF.')
            print("Defaulting to analytical value in uniform case.")
        else:
            fwhmRMSF = mp.params[2]
    
    # If the weight array has been passed in ...
    if not weight is None:
        
        uniformWt = False
        weightArr = np.array(weight, dtype=dtype)
    
        # Check weightArr and lamSqArr have the same length
        if not weightArr.shape[0] == lamSqArr.shape[0]:
            print('Err: the lamSqArr and weightArr are not the same length.')
            sys.exit(1)
            
    # or else use uniform weighting
    else:
        uniformWt = True
        weightArr = np.ones(lamSqArr.shape, dtype=dtype)

    if doPlots:
        
        from matplotlib import pyplot as plt
        
        # Setup the figure to track the clean
        fig = plt.figure(figsize=(12.0, 8))
        ax = fig.add_subplot(111)
        yMaxPlot = np.nanmax(np.abs(dirtyFDF))
        yMinPlot = np.nanmin(np.abs(dirtyFDF))
        yRangePlot = yMaxPlot - yMinPlot
        yMaxPlot +=  yRangePlot * 0.05
        yMinPlot -=  yRangePlot * 0.05
        ax.set_ylim([yMinPlot, yMaxPlot])
        fig.show()
        
    # Prerequisite calculations ----------------------------------------------#
    
    # Calculate the normalisation constant.
    # BdB Equations (24) and (38) give the inverse sum of the weights.
    K = 1.0 / np.nansum(weightArr)
    
    # Calculate the default lambda_0^2:
    # the weighted mean of the LambdaSq distribution (B&dB Eqn. 32).
    lam0Sq_m2 = K * np.nansum(weightArr * lamSqArr)

    # Calculate the RMSF if it has not been passed in 
    # Equation (26) OF BdB05
    if RMSFArr is None:
        RMSFArr, phi2Arr, fwhmRMSF= get_RMSF(lamSqArr, phiArr, weightArr,
                                             lam0Sq_m2)
    else:
        phi2Arr = RMSFphiArr
    
    # Find the index of the peak of the RMSF
    indxMaxRMSF = np.nanargmax(RMSFArr)
    
    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    residFDF = dirtyFDF.copy()
    ccArr = np.zeros(phiArr.shape, dtype='complex')
    cleanFDF = np.zeros(phiArr.shape, dtype='complex')

    # HOGBOM CLEAN -----------------------------------------------------------#

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric
    phiMin = np.min(phiArr)
    nPhiPad = np.argwhere(phi2Arr==phiMin)[0][0]

    # Main CLEAN loop
    iterCount = 0
    while ( np.max(np.abs(residFDF)) >= cutoff and iterCount < maxIter ):
        print(np.max(np.abs(residFDF)), cutoff)
        # Get the absolute peak channel and values of the residual FDF at peak
        indxPeakFDF = np.nanargmax(np.abs(residFDF))
        peakFDFvals = residFDF[indxPeakFDF]

        # What is the faraday depth at this channel?
        phiPeak = phiArr[indxPeakFDF]
        
        # A clean component (CC) is the max absolute amplitude * loop-gain
        #cc = gain * maxAbsResidFDF
        cc = gain * peakFDFvals
        ccArr[indxPeakFDF] += cc
        
        # At which channel is the CC located at in the RMSF?
        indxPeakRMSF = np.argwhere(phi2Arr==phiPeak)[0][0]
        
        # Shift the RMSF in Faraday Depth so that its peak is centred above
        # this CC
        shiftedRMSFArr = np.roll(RMSFArr, indxPeakRMSF-indxMaxRMSF)
        
        # Clip the shifted RMSF to correspond to our FD range
        shiftedRMSFArr = shiftedRMSFArr[nPhiPad:-nPhiPad]
        
        # Subtract the product of the CC shifted RMSF from the residual FDF
        residFDF -= cc * shiftedRMSFArr
 
        # Restore the CC * a Gaussian 'main peak' RMSF to the cleaned FDF
        cleanFDF += cc * np.exp(-2.77258872224 *
                                np.power( (phiArr - phiPeak)/fwhmRMSF, 2.0)) 
        
        # Plot the progress of the clean
        if doPlots:
            ax.cla()
            ax.step(phiArr, np.abs(ccArr), color='g',marker='None',mfc='w',
                    mec='g', ms=10, label='none')
            ax.step(phiArr, np.abs(residFDF), color='r',marker='None',mfc='w',
                    mec='g', ms=10, label='none')
            ax.step(phiArr, np.abs(shiftedRMSFArr), color='b',marker='None',
                    mfc='w', mec='g', ms=10, label='none')
            ax.step(phiArr, np.abs(cleanFDF), color='k',marker='None',mfc='w',
                    mec='g', ms=10, label='none')
        
            ax.set_ylim(yMinPlot, yMaxPlot)
            plt.draw()

        # Iterate ...
        iterCount += 1
        print("Iteration %d", end=' ')
        iterCount

    # End clean loop ---------------------------------------------------------#

    # Restore the final residuals to the cleaned FDF
    cleanFDF += residFDF
    
    # Plot the final spectrum
    if doPlots:
        ax.cla()
        ax.step(phiArr, np.abs(dirtyFDF), color='grey',marker='None',mfc='w',
                mec='g', ms=10, label='none')
        ax.step(phiArr, np.abs(cleanFDF), color='k',marker='None',mfc='w',
                mec='g', ms=10, label='none')
        ax.step(phiArr, np.abs(ccArr), color='g',marker='None',mfc='w',
                mec='g', ms=10, label='none')
        ax.step(phiArr, cleanFDF.real, color='r',marker='None',mfc='w',
                mec='g', ms=10, label='none')
        ax.step(phiArr, cleanFDF.imag, color='b',marker='None',mfc='w',
                mec='g', ms=10, label='none')
            
        ax.set_xlabel('phi')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(yMinPlot, yMaxPlot)
        fig.show()
        input()

    return cleanFDF, ccArr, fwhmRMSF, iterCount


#-----------------------------------------------------------------------------#
def plot_complexity(freqArr_Hz, qArr, uArr, dqArr, duArr, fracPol, psi0_deg,
                    RM_radm2):
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from .util_plotTk import plot_pqu_vs_lamsq_ax
    
    lamSqArr_m2 = np.power(C/freqArr_Hz, 2.0)

    # Create a RM-thin model to subtract
    pModArr, qModArr, uModArr = \
             create_pqu_spectra_burn(freqArr_Hz   = freqArr_Hz,
                                     fracPolArr   = [fracPol],
                                     psi0Arr_deg  = [psi0_deg],
                                     RMArr_radm2  = [RM_radm2])
        
    # Subtract the RM-thin model to create a residual q & u
    qResidArr = qArr - qModArr
    uResidArr = uArr - uModArr
    qResidNorm = qResidArr/dqArr
    uResidNorm = uResidArr/duArr
    
    # High resolution models
    freqHirArr_Hz =  np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000) 
    lamSqHirArr_m2 = np.power(C/freqHirArr_Hz, 2.0)
    pModArr, qModArr, uModArr = \
             create_pqu_spectra_burn(freqArr_Hz   = freqHirArr_Hz,
                                     fracPolArr   = [fracPol],
                                     psi0Arr_deg  = [psi0_deg],
                                     RMArr_radm2  = [RM_radm2])
    
    # Plot the fractional spectra
    fig = plt.figure(figsize=(12.0, 10.0))        
    ax1 = fig.add_subplot(221)
    plot_pqu_vs_lamsq_ax(ax=ax1,
                         lamSqArr_m2 = lamSqArr_m2,
                         qArr        = qArr,
                         uArr        = uArr,
                         dqArr       = dqArr,
                         duArr       = duArr,                             
                         qModArr     = qModArr,
                         uModArr     = uModArr,
                         lamSqHirArr_m2 = lamSqHirArr_m2)
    
    # Plot the residual in lambda-sq space
    ax2 = fig.add_subplot(222)
    ax2.errorbar(x=lamSqArr_m2, y=qResidArr, mec='none', mfc='b', ms=4,
                 fmt='o', label='q Residual')
    ax2.errorbar(x=lamSqArr_m2, y=uResidArr, mec='none', mfc='r', ms=4,
                 fmt='o', label='u Residual')
    ax2.axhline(0, color='grey')
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.set_xlabel('$\\lambda^2$ (m$^2$)')
    #ax2.set_ylabel('Residual ($\sigma$)')
    ax2.set_ylabel('Residual (fractional polarisation)')

    # Plot the distribution of the residual
    ax3 = fig.add_subplot(223)
    nBins = 30
    n, b, p = ax3.hist(qResidNorm, nBins, normed=1, edgecolor="b",
                       histtype='step', linewidth=1.0, zorder=2)
    n, b, p = ax3.hist(uResidNorm, nBins, normed=1, edgecolor="r",
                       histtype='step', linewidth=1.0, zorder=2)
    
    # Overlay a Gaussian
    H = 1.0/m.sqrt(2.0*m.pi)                  # Normalised height
    FWHM = 2.0 * m.sqrt(2.0 * m.log(2.0))     # 1-sigma
    
    x = np.linspace(b[0], b[-1], 1000)
    g = gauss1D(amp=H, mean=0.0, fwhm=FWHM)(x)
    ax3.plot(x, g, color='k', linewidth=2, linestyle="--", zorder=1)
    ax3.set_ylabel('Normalised Units')
    ax3.set_xlabel('Residual ($\sigma$)')

    # Plot the cumulative distribution function
    N = len(qResidNorm)
    cumArr = np.array(list(range(N)))/float(N)
    qResidNormSrt = np.sort(qResidNorm)
    uResidNormSrt = np.sort(uResidNorm)
    ax4 = fig.add_subplot(224)
    ax4.step(qResidNormSrt, cumArr, color='b')
    ax4.step(uResidNormSrt, cumArr, color='r')
    x, y = norm_cdf(mean=0.0, std=1.0, N=1000)
    ax4.step(x, y, color='k', linewidth=2, linestyle="--", zorder=1)
    ax4.set_ylabel('CDF')
    ax4.set_xlabel('Residual ($\sigma$)')

    return fig



def threeDnoise_do_rmsynth_planes(dataQ, dataU, lambdaSqArr_m2, phiArr_radm2, 
                      weightArr=None, lam0Sq_m2=None, nBits=32, verbose=False,log=print):
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
    weightArr       ... 1, 2 or 3D array of weights, default [None] is Uniform (all 1s)
    nBits           ... precision of data arrays [32]
    verbose         ... print feedback during calculation [False]
    
    Returns:
        FDFcube     ... 1, 2, or 3D complex array of Faraday spectra
        lam0Sq_m2   ... 0, 1, or 2D array of lambda_0^2 values.
    """
    
    log('3D noise functions are still in prototype stage! Proper function is not guaranteed!')
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Set the weight array
    if weightArr is None:
        weightArr = np.ones(dataQ.shape, dtype=dtFloat)
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)
    
    # Sanity check on array sizes
    if (not weightArr.shape  == lambdaSqArr_m2.shape) and (not weightArr.shape == dataQ.shape):
        log("Err: Weight array must have same size as lambda^2 or Q/U arrays.")
        return None, None
    if not dataQ.shape == dataU.shape:
        log("Err: Stokes Q and U data arrays must be the same shape.")
        return None, None
    nDims = len(dataQ.shape)
    if not dataQ.shape[0] == lambdaSqArr_m2.shape[0]:
        log("Wavelength array and first axis of data must have same length.")
        return None, None
    if not nDims <= 3:
        log("Err: data dimensions must be <= 3.")
        return None, None
    if not dataQ.shape[0] == lambdaSqArr_m2.shape[0]:
        log("Err: Data depth does not match lambda^2 vector (%d vs %d).", end=' ')
        (dataQ.shape[0], lambdaSqArr_m2.shape[0])
        print("     Check that data is in [z, y, x] order.")
        return None, None
    
    # Reshape the data arrays to 3 dimensions
    if nDims==1:
        dataQ = np.reshape(dataQ, (dataQ.shape[0], 1, 1))
        dataU = np.reshape(dataU, (dataU.shape[0], 1, 1))
    elif nDims==2:
        dataQ = np.reshape(dataQ, (dataQ.shape[0], dataQ.shape[1], 1))
        dataU = np.reshape(dataU, (dataU.shape[0], dataU.shape[1], 1))
    
    #Check shape of weight array and pad if necessary:
    if weightArr.ndim < 3:
        weightArr=np.repeat(weightArr,dataQ.shape)
    
    
    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY, nX]
    pCube = (dataQ + 1j * dataU) * weightArr
    
    # Check for NaNs (flagged data) in the cube & set to zero
    mskCube = np.isnan(pCube)
    pCube = np.nan_to_num(pCube)
    
    # If full planes are flagged then set corresponding weights to zero
    mskPlanes =  np.sum(np.sum(~mskCube, axis=1), axis=1)
    mskPlanes = np.where(mskPlanes==0, 0, 1)
    weightArr *= mskPlanes[:,np.newaxis,np.newaxis]
    
    # Initialise the complex Faraday Dispersion Function cube
    nX = dataQ.shape[-1]
    nY = dataQ.shape[-2]
    nPhi = phiArr_radm2.shape[0]
    FDFcube = np.zeros((nPhi, nY, nX), dtype=dtComplex)

    # lam0Sq_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a per-pixel lam0Sq_m2 value, ignoring isolated flagged voxels
    K = 1.0 / np.sum(weightArr,axis=0)
    if lam0Sq_m2 is None:
        lam0Sq_m2 = K * np.sum(weightArr * lambdaSqArr_m2[:,np.newaxis,np.newaxis],axis=0)
    
    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weightCube =  np.invert(mskCube) * weightArr
    with np.errstate(divide='ignore', invalid='ignore'):
        KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
        KArr[KArr == np.inf] = 0
        KArr = np.nan_to_num(KArr)
        
    # Do the RM-synthesis on each plane
    if verbose:
        log("Running RM-synthesis by channel.")
        progress(40, 0)
    a = lambdaSqArr_m2[:,np.newaxis,np.newaxis] -lam0Sq_m2[np.newaxis,:,:]
    for i in range(nPhi):
#        if verbose:
#            progress(40, ((i+1)*100.0/nPhi))
        arg = np.exp(-2.0j * phiArr_radm2[i] * a)
        FDFcube[i,:,:] =  KArr * np.sum(pCube * arg, axis=0)
        
    # Remove redundant dimensions in the FDF array
    FDFcube = np.squeeze(FDFcube)
    lam0Sq_m2=np.squeeze(lam0Sq_m2)

    return FDFcube, lam0Sq_m2


#-----------------------------------------------------------------------------#
#3D noise functions are still in prototype stage! Proper function is not guaranteed!
def threeDnoise_get_rmsf_planes(lambdaSqArr_m2, phiArr_radm2, weightArr=None, 
                    lam0Sq_m2=None, double=True, fitRMSF=False,
                    fitRMSFreal=False, nBits=32, verbose=False,log=print):
    """Calculate the Rotation Measure Spread Function from inputs. This version
    returns a cube (1, 2 or 3D) of RMSF spectra based on the shape of a
    boolean mask and/or weight array, where flagged data are True and unflagged data False.
    If only whole planes (wavelength channels) are flagged then the RMSF is the
    same for all pixels and the calculation is done once and replicated to the
    dimensions of the mask. If some isolated voxels are flagged then the RMSF
    is calculated by looping through each wavelength plane, which can take some
    time. By default the routine returns the analytical width of the RMSF main
    lobe but can also use MPFIT to fit a Gaussian.
    This version assumes that masking for bad voxels has already been applied to the
    weight array (as zeros or NaNs).
    If no weight array is supplied, uniform (equal) weights are assumed.
    
    lambdaSqArr_m2  ... vector of wavelength^2 values (assending freq order)
    phiArr_radm2    ... vector of trial Faraday depth values
    weightArr       ... array (1/2/3D) of weights, default [None] is no weighting    
    lam0Sq_m2       ... force a reference lambda^2 value (def=calculate) [None]
    double          ... pad the Faraday depth to double-size [True]
    fitRMSF         ... fit the main lobe of the RMSF with a Gaussian [False]
    fitRMSFreal     ... fit RMSF.real, rather than abs(RMSF) [False]
    nBits           ... precision of data arrays [32]
    verbose         ... print feedback during calculation [False]
    
    """
    log('3D noise functions are still in prototype stage! Proper function is not guaranteed!')
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
    nDims = weightArr.ndim



    # Set the mask array (default to 1D, no masked channels)
    
    
    # Sanity checks on array sizes
#    if not weightArr.shape  == lambdaSqArr_m2.shape:
#        print("Err: wavelength^2 and weight arrays must be the same shape.")
#        return None, None, None, None
    if not nDims <= 3:
        log("Err: mask dimensions must be <= 3.")
        return None, None, None, None
    if not weightArr.shape[0] == lambdaSqArr_m2.shape[0]:
        log("Error: Weight depth does not match lambda^2 vector ({} vs {}).".format(weightArr.shape[0], lambdaSqArr_m2.shape[-1]))

        log("Check that the mask is in [z, y, x] order.")
        return None, None, None, None
    

    #Adjust dimensions of weightArr to be 3D no matter what, for convenience/generality below
    if nDims == 1:
        weightArr=weightArr[:,np.newaxis,np.newaxis]
    elif nDims == 2:
        weightArr=weightArr[:,:,np.newaxis]
    
    
    # Initialise the complex RM Spread Function cube
    nX = weightArr.shape[-1]
    nY = weightArr.shape[-2]
    nPix = nX * nY
    nPhi = phi2Arr.shape[0]
    RMSFcube = np.ones((nPhi, nY, nX), dtype=dtComplex)

    # If full planes are flagged then set corresponding weights to zero
    #Check if weights are uniform across the image plane:
    pixel_variation=np.sum(np.std(weightArr,axis=(1,2)))
    if pixel_variation == 0.:
        do1Dcalc=True
    else:
        do1Dcalc=False
    
    
    # lam0Sq is the weighted mean of LambdaSq distribution (B&dB Eqn. 32)
    # Calculate a lam0Sq_m2 value per pixel, ignoring isolated flagged voxels
    K = 1.0 / np.nansum(weightArr,axis=0)
    lam0Sq_m2 = K * np.nansum(weightArr * lambdaSqArr_m2[:,np.newaxis,np.newaxis],axis=0)

    # Calculate the analytical FWHM width of the main lobe    
    fwhmRMSF = 2.0 * m.sqrt(3.0)/(np.nanmax(lambdaSqArr_m2) -
                                  np.nanmin(lambdaSqArr_m2))

    # Do a simple 1D calculation and replicate along X & Y axes
    if do1Dcalc:
        if verbose:
            log("Calculating 1D RMSF and replicating along X & Y axes.")

        # Calculate the RMSF
        a = (-2.0 * 1j * phi2Arr).astype(dtComplex)
        b = (lambdaSqArr_m2 - lam0Sq_m2[0,0])
        RMSFArr = K[0,0] * np.sum(weightArr[:,0,0] * np.exp( np.outer(a, b) ), 1)
        
        # Fit the RMSF main lobe
        fitStatus = -1
        if fitRMSF:
            if verbose:
                log("Fitting Gaussian to the main lobe.")
            if fitRMSFreal:
                mp = fit_rmsf(phi2Arr, RMSFcube.real)
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

    # Calculate the RMSF at each pixel
    else:
        if verbose:
            log("Calculating RMSF in 3D.")

        # The K value used to scale each RMSF must take into account
        # isolated flagged voxels data in the datacube
        with np.errstate(divide='ignore', invalid='ignore'):
            KArr = np.true_divide(1.0, np.sum(weightArr, axis=0))
            KArr[KArr == np.inf] = 0
            KArr = np.nan_to_num(KArr)

        # Calculate the RMSF for each plane
        if verbose:
            progress(40, 0)
        a = lambdaSqArr_m2[:,np.newaxis,np.newaxis] -lam0Sq_m2[np.newaxis,:,:]
        for i in range(nPhi):
            if verbose:
                progress(40, ((i+1)*100.0/nPhi))
            arg = np.exp(-2.0j * phi2Arr[i] * a)
#            RMSFcube[i,:,:] =  KArr * np.sum(uCube * arg, axis=0)
            RMSFcube[i,:,:] =  KArr * np.sum(weightArr * arg, axis=0)

        # Default to the analytical RMSF
        fwhmRMSFArr = np.ones((nY, nX), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nY, nX), dtype="int") * (-1)
    
        # Fit the RMSF main lobe
        if fitRMSF:
            if verbose:
                log("Fitting main lobe in each RMSF spectrum.")
                log("> This may take some time!")
            progress(40, 0)
            k = 0
            for i in range(nX):
                for j in range(nY):
                    k += 1
                    if verbose:
                        progress(40, (k*100.0/nPix))
                    if fitRMSFreal:
                        mp = fit_rmsf(phi2Arr, RMSFcube[:,j,i].real)
                    else:
                        mp = fit_rmsf(phi2Arr, np.abs(RMSFcube[:,j,i]))
                    if not (mp is None or mp.status<1):
                        fwhmRMSFArr[j,i] = mp.params[2]
                        statArr[j,i]  = mp.status
    
    # Remove redundant dimensions
    RMSFcube = np.squeeze(RMSFcube)
    fwhmRMSFArr = np.squeeze(fwhmRMSFArr)
    statArr = np.squeeze(statArr)
    
    return RMSFcube, phi2Arr, fwhmRMSFArr, statArr

    
