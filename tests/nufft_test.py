#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the NUFFT sections"""

import logging
from time import time
from typing import NamedTuple

import numpy as np
from astropy.constants import c as speed_of_light
from tqdm import trange

from RMutils.util_RM import do_rmsynth_planes, extrap, fit_rmsf, get_rmsf_planes

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FakeData(NamedTuple):
    """Fake data for testing RM-synthesis"""

    freq: np.ndarray
    """Frequency array (Hz)"""
    lsq: np.ndarray
    """Wavelength^2 array (m^2)"""
    rm: float
    """Rotation Measure (rad/m^2)"""
    stokes_q: np.ndarray
    """Stokes Q array"""
    stokes_u: np.ndarray
    """Stokes U array"""
    weights: np.ndarray
    """Weights array"""
    lsq_0: float
    """Weighted mean of lsq"""
    phis: np.ndarray
    """Faraday depth array (rad/m^2)"""


def do_rmsynth_planes_old(
    dataQ,
    dataU,
    lambdaSqArr_m2,
    phiArr_radm2,
    weightArr=None,
    lam0Sq_m2=None,
    nBits=32,
    verbose=False,
    log=print,
):
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
    dtComplex = "complex" + str(2 * nBits)

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
    if not np.isfinite(lam0Sq_m2):  # Can happen if all channels are NaNs/zeros
        lam0Sq_m2 = 0.0

    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weightCube = np.invert(mskCube) * weightArr[:, np.newaxis, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
        KArr[KArr == np.inf] = 0
        KArr = np.nan_to_num(KArr)

    # Do the RM-synthesis on each plane
    a = lambdaSqArr_m2 - lam0Sq_m2
    for i in trange(nPhi, desc="Running RM-synthesis by channel", disable=not verbose):
        arg = np.exp(-2.0j * phiArr_radm2[i] * a)[:, np.newaxis, np.newaxis]
        FDFcube[i, :, :] = KArr * np.sum(pCube * arg, axis=0)

    # Check for pixels that have Re(FDF)=Im(FDF)=0. across ALL Faraday depths
    # These pixels will be changed to NaN in the output
    zeromap = np.all(FDFcube == 0.0, axis=0)
    zeropxlist = np.where(zeromap)
    if np.shape(zeropxlist)[1] != 0:
        FDFcube[:, zeropxlist[0], zeropxlist[1]] = np.nan + 1.0j * np.nan

    # Remove redundant dimensions in the FDF array
    FDFcube = np.squeeze(FDFcube)

    return FDFcube, lam0Sq_m2


# -----------------------------------------------------------------------------#
def get_rmsf_planes_old(
    lambdaSqArr_m2,
    phiArr_radm2,
    weightArr=None,
    mskArr=None,
    lam0Sq_m2=None,
    double=True,
    fitRMSF=False,
    fitRMSFreal=False,
    nBits=32,
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
    if not weightArr.shape == lambdaSqArr_m2.shape:
        log("Err: wavelength^2 and weight arrays must be the same shape.")
        return None, None, None, None
    if not nDims <= 3:
        log("Err: mask dimensions must be <= 3.")
        return None, None, None, None
    if not mskArr.shape[0] == lambdaSqArr_m2.shape[0]:
        log(
            f"Err: mask depth does not match lambda^2 vector ({mskArr.shape[0]} vs {lambdaSqArr_m2.shape[-1]}).",
            end=" ",
        )
        log("     Check that the mask is in [z, y, x] order.")
        return None, None, None, None

    # Reshape the mask array to 3 dimensions
    if nDims == 1:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], 1, 1))
    elif nDims == 2:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], mskArr.shape[1], 1))

    # Create a unit cube for use in RMSF calculation (negative of mask)
    # CVE: unit cube removed: it wasn't accurate for non-uniform weights, and was no longer used

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
    do1Dcalc = True
    if len(flagTotals) > 0:
        do1Dcalc = False

    # lam0Sq is the weighted mean of LambdaSq distribution (B&dB Eqn. 32)
    # Calculate a single lam0Sq_m2 value, ignoring isolated flagged voxels
    K = 1.0 / np.nansum(weightArr)
    if lam0Sq_m2 is None:
        lam0Sq_m2 = K * np.nansum(weightArr * lambdaSqArr_m2)

    # Calculate the analytical FWHM width of the main lobe
    fwhmRMSF = 3.8 / (np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2))

    # Do a simple 1D calculation and replicate along X & Y axes
    if do1Dcalc:
        if verbose:
            log("Calculating 1D RMSF and replicating along X & Y axes.")

        # Calculate the RMSF
        a = (-2.0 * 1j * phi2Arr).astype(dtComplex)
        b = lambdaSqArr_m2 - lam0Sq_m2
        RMSFArr = K * np.sum(weightArr * np.exp(np.outer(a, b)), 1)

        # Fit the RMSF main lobe
        fitStatus = -1
        if fitRMSF:
            if verbose:
                log("Fitting Gaussian to the main lobe.")
            mp = fit_rmsf(phi2Arr, RMSFArr.real if fitRMSFreal else np.abs(RMSFArr))
            if mp is None or mp.status < 1:
                log("Err: failed to fit the RMSF.")
                log("     Defaulting to analytical value.")
            else:
                fwhmRMSF = mp.params[2]
                fitStatus = mp.status

        # Replicate along X and Y axes
        RMSFcube = np.tile(RMSFArr[:, np.newaxis, np.newaxis], (1, nY, nX))
        fwhmRMSFArr = np.ones((nY, nX), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nY, nX), dtype="int") * fitStatus

    # Calculate the RMSF at each pixel
    else:
        if verbose:
            log()

        # The K value used to scale each RMSF must take into account
        # isolated flagged voxels data in the datacube
        weightCube = np.invert(mskArr) * weightArr[:, np.newaxis, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
            KArr[KArr == np.inf] = 0
            KArr = np.nan_to_num(KArr)

        # Calculate the RMSF for each plane
        a = lambdaSqArr_m2 - lam0Sq_m2
        for i in trange(nPhi, desc="Calculating RMSF by channel", disable=not verbose):
            arg = np.exp(-2.0j * phi2Arr[i] * a)[:, np.newaxis, np.newaxis]
            RMSFcube[i, :, :] = KArr * np.sum(weightCube * arg, axis=0)

        # Default to the analytical RMSF
        fwhmRMSFArr = np.ones((nY, nX), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nY, nX), dtype="int") * (-1)

        # Fit the RMSF main lobe
        if fitRMSF:
            if verbose:
                log("Fitting main lobe in each RMSF spectrum.")
                log("> This may take some time!")
            k = 0
            for i in trange(nX, desc="Fitting RMSF by pixel", disable=not verbose):
                for j in range(nY):
                    k += 1
                    if fitRMSFreal:
                        mp = fit_rmsf(phi2Arr, RMSFcube[:, j, i].real)
                    else:
                        mp = fit_rmsf(phi2Arr, np.abs(RMSFcube[:, j, i]))
                    if not (mp is None or mp.status < 1):
                        fwhmRMSFArr[j, i] = mp.params[2]
                        statArr[j, i] = mp.status

    # Remove redundant dimensions
    RMSFcube = np.squeeze(RMSFcube)
    fwhmRMSFArr = np.squeeze(fwhmRMSFArr)
    statArr = np.squeeze(statArr)

    return RMSFcube, phi2Arr, fwhmRMSFArr, statArr


def make_fake_data() -> FakeData:
    # Set up
    freq = np.arange(744, 1032, 1) * 1e6
    lsq = (speed_of_light.value / freq) ** 2
    rm = -100

    # Make fake data
    quArr = 1 * np.exp(2j * (np.radians(0) + rm * lsq))
    stokes_q = quArr.real
    stokes_u = quArr.imag
    weights = np.ones_like(freq)
    lsq_0 = np.average(lsq, weights=weights)
    phis = np.linspace(-500, 500, 10_000)

    return FakeData(
        freq=freq,
        lsq=lsq,
        rm=rm,
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        weights=weights,
        lsq_0=lsq_0,
        phis=phis,
    )


def test_rmsynth() -> None:
    """Test the NUFFT RM-synthesis routine agaist DFT."""
    fake_data = make_fake_data()
    for eps in [1e-4, 1e-5, 1e-6, 1e-8]:
        tick = time()
        FDFcube, lam0Sq_m2 = do_rmsynth_planes(
            dataQ=fake_data.stokes_q,
            dataU=fake_data.stokes_u,
            lambdaSqArr_m2=fake_data.lsq,
            phiArr_radm2=fake_data.phis,
            weightArr=fake_data.weights,
            lam0Sq_m2=fake_data.lsq_0,
            eps=eps,
        )
        tock = time()
        logger.info(f"Time taken for NUFFT: {(tock - tick)*1000:0.2f} ms")

        tick = time()
        FDFcube_old, lam0Sq_m2_old = do_rmsynth_planes_old(
            dataQ=fake_data.stokes_q,
            dataU=fake_data.stokes_u,
            lambdaSqArr_m2=fake_data.lsq,
            phiArr_radm2=fake_data.phis,
            weightArr=fake_data.weights,
            lam0Sq_m2=fake_data.lsq_0,
        )
        tock = time()
        logger.info(f"Time taken for DFT: {(tock - tick)*1000:0.2f} ms")

        # fiNUFFT can't go below 1e-8 in precision!
        if eps == 1e-8:
            assert np.allclose(FDFcube, FDFcube_old, rtol=eps * 10, atol=eps * 10)
            assert np.allclose(lam0Sq_m2, lam0Sq_m2_old, rtol=eps * 10, atol=eps * 10)
        else:
            assert np.allclose(FDFcube, FDFcube_old, rtol=eps, atol=eps)
            assert np.allclose(lam0Sq_m2, lam0Sq_m2_old, rtol=eps, atol=eps)


def test_rmsf():
    """Test the NUFFT RMSF routine agaist DFT."""
    fake_data = make_fake_data()
    for eps in [1e-4, 1e-5, 1e-6, 1e-8]:
        tick = time()
        RMSFcube, phi2Arr, fwhmRMSFArr, statArr, _ = get_rmsf_planes(
            lambdaSqArr_m2=fake_data.lsq,
            phiArr_radm2=fake_data.phis,
            weightArr=fake_data.weights,
            lam0Sq_m2=fake_data.lsq_0,
            eps=eps,
        )
        tock = time()
        logger.info(f"Time taken for NUFFT: {(tock - tick)*1000:0.2f} ms")

        tick = time()
        RMSFcube_old, phi2Arr_old, fwhmRMSFArr_old, statArr_old = get_rmsf_planes_old(
            lambdaSqArr_m2=fake_data.lsq,
            phiArr_radm2=fake_data.phis,
            weightArr=fake_data.weights,
            lam0Sq_m2=fake_data.lsq_0,
        )
        tock = time()
        logger.info(f"Time taken for DFT: {(tock - tick)*1000:0.2f} ms")

        # fiNUFFT can't go below 1e-8 in precision!
        for new, old in zip(
            [RMSFcube, phi2Arr, fwhmRMSFArr, statArr],
            [RMSFcube_old, phi2Arr_old, fwhmRMSFArr_old, statArr_old],
        ):
            if eps == 1e-8:
                assert np.allclose(new, old, rtol=eps * 10, atol=eps * 10)
            else:
                assert np.allclose(new, old, rtol=eps, atol=eps)
