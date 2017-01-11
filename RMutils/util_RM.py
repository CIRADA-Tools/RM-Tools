#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     util_RM.py                                                        #
#                                                                             #
# PURPOSE:  Common procedures used with RM-synthesis scripts.                 #
#                                                                             #
# REQUIRED: Requires the numpy and scipy modules.                             #
#                                                                             #
# MODIFIED: 11-Jan-2017 by C.Purcell.                                         #
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
#  measure_qu_complexity  ... measure the complexity of a q & u spectrum      #
#  measure_fdf_complexity  ... measure the complexity of a clean FDF spectrum #
#                                                                             #
# DEPRECATED CODE ------------------------------------------------------------#
#                                                                             #
#  do_rmsynth          ... perform RM-synthesis on Q & U data by spectrum     #
#  get_RMSF            ... calculate the RMSF for a 1D wavelength^2 array     #
#  do_rmclean          ... perform Hogbom RM-clean on a dirty FDF             #
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
import math as m
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import skewtest
from scipy.stats import kurtosistest

from mpfit import mpfit
from util_misc import progress  
from util_misc import toscalar
from util_misc import calc_parabola_vertex
from util_misc import create_pqu_spectra_burn
from util_misc import calc_mom2_FDF

# Constants
C = 2.99792458e8


#-----------------------------------------------------------------------------#
def do_rmsynth_planes(dataQ, dataU, lambdaSqArr_m2, phiArr_radm2, 
                      weightArr=None, nBits=32, verbose=False):
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
    weightArr       ... vector of weights, default [None] is Natural (all 1s)
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
        print "Err: Lambda^2 and weight arrays must be the same shape."
        return None, None
    if not dataQ.shape == dataU.shape:
        print "Err: Stokes Q and U data arrays must be the same shape."
        return None, None
    nDims = len(dataQ.shape)
    if not nDims <= 3:
        print "Err: data dimensions must be <= 3."
        return None, None
    if not dataQ.shape[0] == lambdaSqArr_m2.shape[0]:
        print "Err: Data depth does not match lambda^2 vector (%d vs %d)." \
              % (dataQ.shape[0], lambdaSqArr_m2.shape[0])
        print "     Check that data is in [z, y, x] order."
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
        print "Running RM-synthesis by channel."
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
                    fitRMSFreal=False, nBits=32, verbose=False):
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
        print "Err: wavelength^2 and weight arrays must be the same shape."
        return None, None, None, None
    if not nDims <= 3:
        print "Err: mask dimensions must be <= 3."
        return None, None, None, None
    if not mskArr.shape[0] == lambdaSqArr_m2.shape[0]:
        print "Err: mask depth does not match lambda^2 vector (%d vs %d)." \
              % (mskArr.shape[0], lambdaSqArr_m2.shape[-1])
        print "     Check that the mask is in [z, y, x] order."
        return None, None, None, None
    
    # Reshape the mask array to 3 dimensions
    if nDims==1:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], 1, 1))
    elif nDims==2:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], mskArr.shape[1], 1))
    
    # Create a unit cube for use in RMSF calculation (negative of mask)
    uCube = np.invert(mskArr).astype(dtComplex)
    
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
            print "Calculating 1D RMSF and replicating along X & Y axes."

        # Calculate the RMSF
        a = (-2.0 * 1j * phi2Arr).astype(dtComplex)
        b = (lambdaSqArr_m2 - lam0Sq_m2)
        RMSFArr = K * np.sum(weightArr * np.exp( np.outer(a, b) ), 1)
        
        # Fit the RMSF main lobe
        fitStatus = -1
        if fitRMSF:
            if verbose:
                print "Fitting Gaussian to the main lobe."
            if fitRMSFreal:
                mp = fit_rmsf(phi2Arr, RMSFcube.real)
            else:
                mp = fit_rmsf(phi2Arr, np.abs(RMSFArr))
            if mp is None or mp.status<1:
                print "Err: failed to fit the RMSF."
                print "     Defaulting to analytical value."
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
            print "Calculating RMSF by channel."

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
            arg = np.exp(-2.0j * phi2Arr[i] * a)[:, np.newaxis, np.newaxis]
            RMSFcube[i,:,:] =  KArr * np.sum(uCube * arg, axis=0)

        # Default to the analytical RMSF
        fwhmRMSFArr = np.ones((nY, nX), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nY, nX), dtype="int") * (-1)
    
        # Fit the RMSF main lobe
        if fitRMSF:
            if verbose:
                print "Fitting main lobe in each RMSF spectrum."
                print "> This may take some time!"
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
        print "Err: 'phi2Arr_radm2' and 'dirtyFDF' are not the same length."
        return None, None, None
    nPhi2 = phi2Arr_radm2.shape[0]
    if not nPhi2 == RMSFArr.shape[0]:
        print "Err: missmatch in 'phi2Arr_radm2' and 'RMSFArr' length."
        return None, None, None
    if not (nPhi2 >= 2 * nPhi):
        print "Err: the Faraday depth of the RMSF must be twice the FDF."
        return None, None, None
    nDims = len(dirtyFDF.shape)
    if not nDims <= 3:
        print "Err: FDF array dimensions must be <= 3."
        return None, None, None
    if not nDims == len(RMSFArr.shape):
        print "Err: the input RMSF and FDF must have the same number of axes."
        return None, None, None
    if not RMSFArr.shape[1:]==dirtyFDF.shape[1:]:
        print "Err: the xy dimesions of the RMSF and FDF must match."
        return None, None, None
    if mskArr is not None:
        if not mskArr.shape==dirtyFDF.shape[1:]:
            print "Err: pixel mask must match xy dimesnisons of FDF cube."
            print "     FDF[z,y,z] = %s, Mask[y,x] = %s." % (dirtyFDF.shape,
                                                             mskArr.shape)
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
        dirtyFDF = np.reshape(list(dirtyFDF.shape[:2])+[1])
        RMSFArr = np.reshape(list(RMSFArr.shape[:2])+[1])
        mskArr = np.reshape(mskArr,  (1, dirtyFDF.shape[1]))
        fwhmRMSFArr = np.reshape(fwhmRMSFArr,  (1, dirtyFDF.shape[1]))
    iterCountArr = np.zeros_like(mskArr, dtype="int")

    # Determine which pixels have components above the cutoff
    absFDF = np.abs(np.nan_to_num(dirtyFDF))
    mskCutoff = np.where(np.max(absFDF, axis=0)>=cutoff,1, 0)
    xyCoords = np.rot90(np.where(mskCutoff>0))

    # Feeback to user
    if verbose:
        nPix = dirtyFDF.shape[-1]* dirtyFDF.shape[-2]
        nCleanPix = len(xyCoords)
        print "Cleaning %d/%d spectra." % (nCleanPix, nPix)
    
    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    residFDF = dirtyFDF.copy()
    ccArr = np.zeros(dirtyFDF.shape, dtype=dtFloat)
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
        progress(40, 0)
    for yi, xi in xyCoords:
        if verbose:
            j += 1
            progress(40, ((j)*100.0/nCleanPix))

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
            ccArr[indxPeakFDF, yi, xi] += np.abs(CC)
        
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

        # Restore the residual to the CLEANed FDF
        cleanFDF[:, yi, xi] += residFDF[:, yi, xi]

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
        raw_input()

    return msk


#-----------------------------------------------------------------------------#
def measure_FDF_parms(FDF, phiArr, fwhmRMSF, dQU, lamSqArr_m2=None,
                      lam0Sq=None, snrDoBiasCorrect=5.0):
    """
    Measure standard parameters from a complex Faraday Dispersion Function.
    Currently this function assumes that the noise levels in the Stokes Q
    and U spectra are the same.
    """
    
    # Determine the peak channel in the FDF, its amplitude and RM
    absFDF = np.abs(FDF)
    ampPeakPIchan = np.nanmax(absFDF)
    indxPeakPIchan = np.nanargmax(absFDF)
    phiPeakPIchan = phiArr[indxPeakPIchan]
    dPhiPeakPIchan = fwhmRMSF * dQU / (2.0 * ampPeakPIchan)
    snrPIchan = ampPeakPIchan / dQU
    dPhi = np.nanmin(np.diff(phiArr))
    
    # Correct the peak for polarisation bias (POSSUM report 11)
    ampPeakPIchanEff = ampPeakPIchan
    if snrPIchan >= snrDoBiasCorrect:
        ampPeakPIchanEff = np.sqrt(ampPeakPIchan**2.0 - 2.3 * dQU**2.0)

    # Calculate the polarisation angle from the channel
    peakFDFimagChan = FDF.imag[indxPeakPIchan]
    peakFDFrealChan = FDF.real[indxPeakPIchan]
    polAngleChan_deg = 0.5 * np.degrees(np.arctan2(peakFDFimagChan,
                                         peakFDFrealChan))
    dPolAngleChan_deg = np.degrees(dQU**2.0 / (4.0 * ampPeakPIchan**2.0))

    # Calculate the derotated polarisation angle and uncertainty
    polAngle0Chan_deg = np.degrees(np.radians(polAngleChan_deg) -
                                  phiPeakPIchan * lam0Sq)
    nChansGood = np.sum(np.where(lamSqArr_m2==lamSqArr_m2, 1.0, 0.0))
    varLamSqArr_m2 = (np.sum(lamSqArr_m2**2.0) -
                      np.sum(lamSqArr_m2)**2.0/nChansGood) / (nChansGood-1)
    dPolAngle0Chan_rad = \
        np.sqrt( dQU**2.0 / (4.0*(nChansGood-2.0)*ampPeakPIchan**2.0) *
                 ((nChansGood-1)/nChansGood + lam0Sq**2.0/varLamSqArr_m2) )
    dPolAngle0Chan_deg = np.degrees(dPolAngle0Chan_rad)

    # Determine the peak in the FDF, its amplitude and Phi using a
    # 3-point parabolic interpolation
    phiPeakPIfit = None
    dPhiPeakPIfit = None
    ampPeakPIfit = None
    dAmpPeakPIfit = None
    snrPIfit = None
    ampPeakPIfitEff = None
    indxPeakPIfit = None
    peakFDFimagFit = None 
    peakFDFrealFit = None 
    polAngleFit_deg = None
    dPolAngleFit_deg = None
    polAngle0Fit_deg = None
    dPolAngle0Fit_deg = None
    
    if indxPeakPIchan > 0 and indxPeakPIchan < len(FDF)-1:
        phiPeakPIfit, ampPeakPIfit = \
                      calc_parabola_vertex(phiArr[indxPeakPIchan-1],
                                           absFDF[indxPeakPIchan-1],
                                           phiArr[indxPeakPIchan],
                                           absFDF[indxPeakPIchan],
                                           phiArr[indxPeakPIchan+1],
                                           absFDF[indxPeakPIchan+1])
        
        snrPIfit = ampPeakPIfit / dQU
        
        # Error on fitted Faraday depth (RM) from Eqn 4b in Landman 1982
        # Parabolic interpolation is approximately equivalent to a Gaussian fit
        dPhiPeakPIfit = (np.sqrt(fwhmRMSF * dPhi) /
                         np.power(2.0*np.pi*np.log(2.0), 0.25) / snrPIfit)
        
        # Error on fitted peak intensity (PI) from Eqn 4a in Landman 1982
        dAmpPeakPIfit = (np.power(18.0*np.log(2.0)/(np.pi), 0.25) *
                         np.sqrt(dPhi) * dQU / np.sqrt(fwhmRMSF))
        
        # Correct the peak for polarisation bias (POSSUM report 11)
        ampPeakPIfitEff = ampPeakPIfit
        if snrPIfit >= snrDoBiasCorrect:
            ampPeakPIfitEff = np.sqrt(ampPeakPIfit**2.0 - 2.3 * dQU**2.0)
            
        # Calculate the polarisation angle from the fitted peak
        # Uncertainty from Eqn A.12 in Brentjens & De Bruyn 2005
        indxPeakPIfit = np.interp(phiPeakPIfit, phiArr,
                                  np.arange(phiArr.shape[-1], dtype='f4'))
        peakFDFimagFit = np.interp(phiPeakPIfit, phiArr, FDF.imag)
        peakFDFrealFit = np.interp(phiPeakPIfit, phiArr, FDF.real)
        polAngleFit_deg = 0.5 * np.degrees(np.arctan2(peakFDFimagFit,
                                                  peakFDFrealFit))
        dPolAngleFit_deg = np.degrees(dQU**2.0 / (4.0 * ampPeakPIfit**2.0))

        # Calculate the derotated polarisation angle and uncertainty
        # Uncertainty from Eqn A.20 in Brentjens & De Bruyn 2005
        polAngle0Fit_deg = np.degrees(np.radians(polAngleFit_deg) -
                                     phiPeakPIfit * lam0Sq)
        dPolAngle0Fit_rad = \
            np.sqrt( dQU**2.0 / (4.0*(nChansGood-2.0)*ampPeakPIfit**2.0) *
                    ((nChansGood-1)/nChansGood + lam0Sq**2.0/varLamSqArr_m2) )
        dPolAngle0Fit_deg = np.degrees(dPolAngle0Fit_rad)

    # Store the measurements in a dictionary and return
    mDict = {'phiPeakPIchan_rm2':     toscalar(phiPeakPIchan),
             'dPhiPeakPIchan_rm2':    toscalar(dPhiPeakPIchan),
             'ampPeakPIchan_Jybm':    toscalar(ampPeakPIchan),
             'ampPeakPIchanEff_Jybm': toscalar(ampPeakPIchanEff),
             'dAmpPeakPIchan_Jybm':   toscalar(dQU),
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
             'dAmpPeakPIfit_Jybm':    toscalar(dQU),
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
def measure_qu_complexity(freqArr_Hz, qArr, uArr, dqArr, duArr, fracPol,
                          psi0_deg, RM_radm2, doPlots=True):
    
    # Fractional polarised intensity
    pArr = np.sqrt(qArr**2.0 + uArr**2.0 )
    dpArr = np.sqrt(dqArr**2.0 + duArr**2.0 )
    
    # Create a RM-thin model to subtract
    pModArr, qModArr, uModArr = \
             create_pqu_spectra_burn(freqArr_Hz   = freqArr_Hz,
                                     fracPolArr   = [fracPol],
                                     psi0Arr_deg  = [psi0_deg],
                                     RMArr_radm2  = [RM_radm2])

    # Subtract the RM-thin model to create a residual q & u
    qResidArr = qArr - qModArr
    uResidArr = uArr - uModArr
    pResidArr = pArr - pModArr
    #pResidArr = np.sqrt(qResidArr**2.0 + uResidArr**2.0)

    # DEBUG
    if False:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from util_plotTk import plot_pqu_vs_lamsq_ax
        fig = plt.figure(figsize=(18.0, 8))
        
        lamSqArr_m2 = np.power(C/freqArr_Hz, 2.0)

        # Plot the Fractional spectra
        ax1 = fig.add_subplot(131)
        plot_pqu_vs_lamsq_ax(ax=ax1,
                             lamSqArr_m2 = lamSqArr_m2,
                             qArr        = qArr,
                             uArr        = uArr,
                             dqArr       = dqArr,
                             duArr       = duArr,
                             qModArr     = qModArr,
                             uModArr     = uModArr)

        # Plot the residual
        ax2 = fig.add_subplot(132)
        ax2.errorbar(x=lamSqArr_m2, y=pResidArr/dpArr, mec='k', mfc='k', ms=4,
                fmt='D', ecolor='k', label='Residual P')
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.xaxis.set_major_locator(MaxNLocator(4))
        
        # Plot the distribution of the residual
        ax3 = fig.add_subplot(133)
        n, b, p = ax3.hist(pResidArr/dpArr, 50, normed=1, facecolor='green',
                           alpha=0.75)
        
        g = gauss1D(amp=0.6, mean=0.0, fwhm=1.00)(b)
        ax3.step(b, g, color='k', linewidth=5)

        # Calculate the skewness (measure of symmetry)
        skewVal = skew(n)
        print "SKEW:", skewVal
#        skewTestVal = skewtest(pResidArr/dpArr)
#        print "SKEWTEST:", skewTestVal


        # Calculate the kurtosis (tails compared to a normal distribution)
        kurtosisVal = kurtosis(n)
        print "KURTOSIS:", kurtosisVal
#        kurtosisTestVal = kurtosistest(pResidArr/dpArr)
#        print "KURTOSISTEST:", kurtosisTestVal
        


        
        """
        pSrtArr = np.sort(pResidArr)
        N = len(pSrtArr)
        pCum = np.array(range(N))/float(N)
        
        g = gauss1D(amp=1.0, mean=0.0, fwhm=0.05)(pSrtArr)
        gSrtArr = np.sort(g)
        N = len(gSrtArr)
        gCum = np.array(range(N))/float(N)



        ax3 = fig.add_subplot(133)
        ax3.plot(pSrtArr, pCum)
        ax3.plot(gSrtArr, gCum)
        """
        
        fig.show()
        raw_input()



    # Complexity metric 1
    M1 = ( np.nansum(np.power(pResidArr, 2.0) / np.power(dpArr, 2))
           / (len(pResidArr)-1) )
    
    # Complexity metric 2
    #M2 = (np.sum(np.power(qResidArr, 2.0) + np.power(uResidArr, 2.0))
    #      /(len(pModArr)-1) )
    M2 = ( np.nansum(np.power(qResidArr, 2.0) / np.power(qArr, 2) +
                     np.power(uResidArr, 2.0) / np.power(qArr, 2))
           / 2*(len(qResidArr)-2) )
    
#    M2 = ( np.nansum(np.power(qResidArr, 2.0) + np.power(uResidArr, 2.0))
#          / 2*(len(pModArr)-2) )
            
            
    return M1, M2
    
    
#-----------------------------------------------------------------------------#
def measure_fdf_complexity(phiArr, ccFDF):

    # Second moment of clean component spectrum
    M3 = calc_mom2_FDF(ccFDF, phiArr)

    return M3


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
        print "Err: Lambda^2 and weight arrays must be the same shape."
        return None, [None, None], None, None
    if not dataQ.shape == dataU.shape:
        print "Err: Stokes Q and U data arrays must be the same shape."
        return None, [None, None], None, None
    nDims = len(dataQ.shape)
    if not nDims <= 3:
        print "Err: data-dimensions must be <= 3."
        return None, [None, None], None, None
    if not dataQ.shape[-1] == lamSqArr.shape[-1]:
        print "Err: The Stokes Q and U arrays mush be in spectral order."
        print "     # Stokes = %d, # Lamda = %d." % (dataQ.shape[-1],
                                                     lamSqArr.shape[-1])
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
        naturalWt = True
        weightArr = np.ones(lamSqArr.shape, dtype=dtype)
    else:
        naturalWt = False
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
    if not naturalWt:
        if fitRMSFreal:
            mp = fit_rmsf(phi2Arr, RMSFArr.real)
        else:
            mp = fit_rmsf(phi2Arr, np.abs(RMSFArr))
        if mp is None or mp.status<1:
            print "Err: failed to fit the RMSF."
            print "Defaulting to analytical value in natural case."
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
        print "Err: the dirty FDF is not a 1D array."
        sys.exit(1)
    nFDF = dirtyFDF.shape[0]

    # Check that dirtyFDF is a complex array
    if not np.iscomplexobj(dirtyFDF):
        print "Err: the dirty FDF is not a complex array."
        sys.exit(1)
        
    # Check that phiArr is 1D and get its length
    if len(phiArr.shape) != 1:
        print "Err: the phi array is not a 1D array."
        sys.exit(1)
    nPhi = phiArr.shape[0]

    # Check that the lamSqArr is 1D and get its length
    if len(lamSqArr.shape) != 1:
        print "Err: the lamSqArr array is not a 1D array."
        sys.exit(1)
    nlamSq = lamSqArr.shape[0]

    # Check that phiArr and FDF arrays are the same length
    if nPhi != nFDF:
        print 'Err: the phiArr and dirty FDF are not the same length.'
        sys.exit(1)
    
    # If the RMSF has been passed in then check for correct formatting:
    #  - Twice the number of channels as dirtyFDF
    #  - Must be complex
    if not RMSFArr is None:

        # Check 1D
        if len(RMSFArr.shape) != 1:
            print "Err: input RMSF must be a 1D array."
            sys.exit(1)
        nRMSF = RMSFArr.shape[0]
        
        # Check complex
        if not np.iscomplexobj(RMSFArr):
            print "Err: the RMSF is not a complex array."
            sys.exit(1)
    
        # Check RMSF is at least double the FDF spectrum
        if not (nRMSF >= 2 * nFDF):
            print 'Err: the RMSF must be twice the length of the FDF.'
            sys.exit(1)

        # Check that phiSampArr is also present and the same length
        if RMSFphiArr is None:
            print 'Err: the phi sampling array must be passed with the RMSF.'
            sys.exit(1)
        nRMSFphi = RMSFphiArr.shape[0]
        if not nRMSF==nRMSFphi:
            print 'Err: the RMSF and phi sampling array must be equal length.'
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
            print 'Err: failed to fit the RMSF.'
            print "Defaulting to analytical value in natural case."
        else:
            fwhmRMSF = mp.params[2]
    
    # If the weight array has been passed in ...
    if not weight is None:
        
        naturalWt = False
        weightArr = np.array(weight, dtype=dtype)
    
        # Check weightArr and lamSqArr have the same length
        if not weightArr.shape[0] == lamSqArr.shape[0]:
            print 'Err: the lamSqArr and weightArr are not the same length.'
            sys.exit(1)
            
    # or else use natural weighting
    else:
        naturalWt = True
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
        print np.max(np.abs(residFDF)), cutoff
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
        print "Iteration %d" % iterCount

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
        raw_input()

    return cleanFDF, ccArr, fwhmRMSF, iterCount

