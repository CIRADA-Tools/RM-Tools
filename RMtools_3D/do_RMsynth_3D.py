#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     do_RMsynth_3D.py                                                  #
#                                                                             #
# PURPOSE:  Run RM-synthesis on a Stokes Q & U cubes.                         #
#                                                                             #
# MODIFIED: 7-March-2019 by J. West                                           #
# MODIFIED: 23-October-2019 by A. Thomson                                     #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2016 Cormac R. Purcell                                        #
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

import gc
import math as m
import os
import sys
import time

import astropy.io.fits as pf
import numpy as np
from astropy.constants import c as speed_of_light

from RMutils.util_misc import interp_images, remove_header_third_fourth_axis
from RMutils.util_RM import do_rmsynth_planes, get_rmsf_planes

if sys.version_info.major == 2:
    print("RM-tools will no longer run with Python 2! Please use Python 3.")
    exit()

# -----------------------------------------------------------------------------#


def run_rmsynth(
    dataQ,
    dataU,
    freqArr_Hz,
    dataI=None,
    rmsArr=None,
    phiMax_radm2=None,
    dPhi_radm2=None,
    nSamples=10.0,
    weightType="uniform",
    fitRMSF=False,
    nBits=32,
    verbose=True,
    not_rmsynth=False,
    not_rmsf=False,
    log=print,
    super_resolution=False,
):
    """Run RM-synthesis on 2/3D data.

    Args:
      dataQ (array_like): Stokes Q intensity cube.
      dataU (array_like): Stokes U intensity cube.
      freqArr_Hz (array_like): Frequency of each channel in Hz.

    Kwargs:
        dataI (array_like): Model cube of Stokes I spectra (see do_fitIcube).
        rmsArr (array_like): Cube of uncertainty spectra.
        phiMax_radm2 (float): Maximum absolute Faraday depth (rad/m^2).
        dPhi_radm2 (float): Faraday depth channel size (rad/m^2).
        nSamples (float): Number of samples across the RMSF.
           weightType (str): Can be "variance" or "uniform"
            "variance" -- Weight by inverse variance.
            "uniform" -- Weight uniformly (i.e. with 1s)
        fitRMSF (bool): Fit a Gaussian to the RMSF?
        nBits (int): Precision of floating point numbers.
        verbose (bool): Verbosity.
        not_rmsynth (bool): Just do RMSF and ignore RM synthesis?
        not_rmsf (bool): Just do RM synthesis and ignore RMSF? -- one of these must be False
        log (function): Which logging function to use.

    Returns:
      dataArr (list): FDF and RMSF information
        if not_rmsf:
            dataArr = [FDFcube, phiArr_radm2, lam0Sq_m2, lambdaSqArr_m2]

        else:
            dataArr = [FDFcube, phiArr_radm2, RMSFcube, phi2Arr_radm2, fwhmRMSFCube,fitStatArr, lam0Sq_m2, lambdaSqArr_m2]


    """
    if not_rmsynth and not_rmsf:
        log(
            "Err: both RM synthesis and RMSF computation not requested?\n"
            + "Please make sure either not_rmsynth or not_rmsf is False"
        )
        sys.exit()

    # Sanity check on header dimensions

    if not str(dataQ.shape) == str(dataU.shape):
        log(
            "Err: unequal dimensions: Q = "
            + str(dataQ.shape)
            + ", U = "
            + str(dataU.shape)
            + "."
        )
        sys.exit()

    # Check dimensions of Stokes I cube, if present
    if dataI is not None:
        if not str(dataI.shape) == str(dataQ.shape):
            log(
                "Err: unequal dimensions: Q = "
                + str(dataQ.shape)
                + ", I = "
                + str(dataI.shape)
                + "."
            )
            sys.exit()

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    lambdaSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)

    dFreq_Hz = np.nanmin(np.abs(np.diff(freqArr_Hz)))
    lambdaSqRange_m2 = np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2)
    dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
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
    nChanRM = int(round(abs((phiMax_radm2 - 0.0) / dPhi_radm2))) * 2 + 1
    startPhi_radm2 = -(nChanRM - 1.0) * dPhi_radm2 / 2.0
    stopPhi_radm2 = +(nChanRM - 1.0) * dPhi_radm2 / 2.0
    phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, int(nChanRM))
    phiArr_radm2 = phiArr_radm2.astype(dtFloat)
    if verbose:
        log(
            "PhiArr = %.2f to %.2f by %.2f (%d chans)."
            % (phiArr_radm2[0], phiArr_radm2[-1], float(dPhi_radm2), nChanRM)
        )

    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weightType == "variance" and rmsArr is not None:
        weightArr = 1.0 / np.power(rmsArr, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    if verbose:
        log("Weight type is '%s'." % weightType)

    startTime = time.time()

    # Read the Stokes I model and divide into the Q & U data
    if dataI is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            qArr = np.true_divide(dataQ, dataI)
            uArr = np.true_divide(dataU, dataI)
    else:
        qArr = dataQ
        uArr = dataU

    # Perform RM-synthesis on the cube
    if not not_rmsynth:
        FDFcube, lam0Sq_m2 = do_rmsynth_planes(
            dataQ=qArr,
            dataU=uArr,
            lambdaSqArr_m2=lambdaSqArr_m2,
            phiArr_radm2=phiArr_radm2,
            weightArr=weightArr,
            nBits=32,
            lam0Sq_m2=0 if super_resolution else None,
        )
    else:
        # need lambda0 for RMSF calculation
        lam0Sq_m2 = 0 if super_resolution else None

    # Calculate the Rotation Measure Spread Function cube
    if not not_rmsf:
        RMSFcube, phi2Arr_radm2, fwhmRMSFCube, fitStatArr, lam0Sq_m2 = get_rmsf_planes(
            lambdaSqArr_m2=lambdaSqArr_m2,
            phiArr_radm2=phiArr_radm2,
            weightArr=weightArr,
            mskArr=~np.isfinite(dataQ),
            lam0Sq_m2=lam0Sq_m2,
            double=True,
            fitRMSF=fitRMSF or super_resolution,
            fitRMSFreal=super_resolution,
            nBits=32,
            verbose=verbose,
            log=log,
        )
    endTime = time.time()
    cputime = endTime - startTime
    if verbose:
        log("> RM-synthesis completed in %.2f seconds." % cputime)

    # Determine the Stokes I value at lam0Sq_m2 from the Stokes I model
    # Note: the Stokes I model MUST be continuous throughout the cube,
    # i.e., no NaNs as the amplitude at freq0_Hz is interpolated from the
    # nearest two planes.
    with np.errstate(divide="ignore", invalid="ignore"):
        freq0_Hz = (
            np.true_divide(speed_of_light.value, m.sqrt(lam0Sq_m2))
            if lam0Sq_m2 > 0
            else np.nanmean(freqArr_Hz)
        )

    if dataI is not None and not not_rmsynth:  # if we created an FDF cube
        idx = np.abs(freqArr_Hz - freq0_Hz).argmin()
        if freqArr_Hz[idx] < freq0_Hz:
            Ifreq0Arr = interp_images(dataI[idx, :, :], dataI[idx + 1, :, :], f=0.5)
        elif freqArr_Hz[idx] > freq0_Hz:
            Ifreq0Arr = interp_images(dataI[idx - 1, :, :], dataI[idx, :, :], f=0.5)
        else:
            Ifreq0Arr = dataI[idx, :, :]

        # Multiply the dirty FDF by Ifreq0 to recover the PI
        FDFcube *= Ifreq0Arr

    if not_rmsf:  # only RMsynth
        dataArr = [FDFcube, phiArr_radm2, lam0Sq_m2, lambdaSqArr_m2]
    elif not_rmsynth:  # only RMSF
        dataArr = [RMSFcube, phi2Arr_radm2, fwhmRMSFCube, fitStatArr, lam0Sq_m2]
    else:  # both have been computed
        dataArr = [
            FDFcube,
            phiArr_radm2,
            RMSFcube,
            phi2Arr_radm2,
            fwhmRMSFCube,
            fitStatArr,
            lam0Sq_m2,
            lambdaSqArr_m2,
        ]

    return dataArr


def writefits(
    dataArr,
    headtemplate,
    fitRMSF=False,
    prefixOut="",
    outDir="",
    nBits=32,
    write_seperate_FDF=True,
    not_rmsynth=False,
    not_rmsf=False,
    do_peakmaps=True,
    verbose=False,
    log=print,
):
    """Write data to disk in FITS

    Args:
      dataArr (list): FDF and/or RMSF information
        if not_rmsf: # only RMsynth
            dataArr = [FDFcube, phiArr_radm2, lam0Sq_m2, lambdaSqArr_m2]
        elif not_rmsynth: # only RMSF
            dataArr = [RMSFcube, phi2Arr_radm2, fwhmRMSFCube, fitStatArr, lam0Sq_m2]
        else: # both
            dataArr = [FDFcube, phiArr_radm2, RMSFcube, phi2Arr_radm2, fwhmRMSFCube,fitStatArr, lam0Sq_m2, lambdaSqArr_m2]

        headtemplate: FITS header template

    Kwargs:
        fitRMSF (bool): Fit a Gaussian to the RMSF?
        prefixOut (str): Prefix for filenames.
        outDir (str): Directory to save files.
        write_seperate_FDF (bool): Write Q, U, and PI separately?
        verbose (bool): Verbosity.
        not_rmsynth (bool): Just do RMSF and ignore RM synthesis?
        not_rmsf (bool): Just do RM synthesis and ignore RMSF? -- one of these must be False
        do_peakmaps (bool): Compute and write peak RM and peak intensity?
        log (function): Which logging function to use.


    Output files:
        Default:
        FDF_maxPI.fits: 2D map of peak polarized intensity per pixel.
        FDF_peakRM.fits: 2D map of Faraday depth of highest peak, per pixel.

        write_seperate_FDF=True: [default]
        FDF_dirty.fits is split into three constituent components:
            FDF_real_dirty.fits: Stokes Q
            FDF_im_dirty.fits: Stokes U
            FDF_tot_dirty.fits: Polarizd Intensity (sqrt(Q^2+U^2))
            RMSF_real.fits: Real/Stokes Q component of RMSF
            RMSF_im.fits: Imaginary/Stokes U component of RMSF
            RM_tot.fits: polarized intensity view of RMSF
            RMSF_FWHM.fits: 2D map of width of RMSF main lobe

        write_seperate_FDF=False:
            FDF_dirty.fits: FDF, in 3 extensions: Q,U, and PI.
            RMSF.fits: 4 extensions; first 3 are RMSF cubes [Q, U, PI]
                                 4th is 2D map of RMSF FWHM.


    """
    if not_rmsynth and not_rmsf:
        log(
            "Err: both RM synthesis and RMSF computation not done?\n"
            + "Please make sure either not_rmsynth or not_rmsf is False"
        )
        sys.exit()

    if not_rmsf:
        FDFcube, phiArr_radm2, lam0Sq_m2, lambdaSqArr_m2 = dataArr
        if verbose:
            log("Saving the dirty FDF and ancillary FITS files.")
    elif not_rmsynth:
        RMSFcube, phi2Arr_radm2, fwhmRMSFCube, fitStatArr, lam0Sq_m2 = dataArr
        if verbose:
            log("Saving the RMSF and ancillary FITS files.")
    else:
        (
            FDFcube,
            phiArr_radm2,
            RMSFcube,
            phi2Arr_radm2,
            fwhmRMSFCube,
            fitStatArr,
            lam0Sq_m2,
            lambdaSqArr_m2,
        ) = dataArr
        if verbose:
            log("Saving the dirty FDF, RMSF and ancillary FITS files.")

    # Default data typess
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    # Make a copy of the Q header and alter frequency-axis as Faraday depth
    header = headtemplate.copy()
    Ndim = header["NAXIS"]
    freq_axis = Ndim  # If frequency axis not found, assume it's the last one.
    # Check for frequency axes. Because I don't know what different formatting
    # I might get ('FREQ' vs 'OBSFREQ' vs 'Freq' vs 'Frequency'), convert to
    # all caps and check for 'FREQ' anywhere in the axis name.
    for i in range(1, Ndim + 1):
        try:
            if "FREQ" in header["CTYPE" + str(i)].upper():
                freq_axis = i
        except:
            pass  # The try statement is needed for if the FITS header does not
            # have CTYPE keywords.

    header["CTYPE" + str(freq_axis)] = ("FDEP", "Faraday depth (linear)")
    header["CUNIT" + str(freq_axis)] = "rad/m^2"
    if not np.isfinite(lam0Sq_m2):
        lam0Sq_m2 = 0.0
    header["LAMSQ0"] = (lam0Sq_m2, "Lambda^2_0, in m^2")
    if "DATAMAX" in header:
        del header["DATAMAX"]
    if "DATAMIN" in header:
        del header["DATAMIN"]

    if outDir == "":  # To prevent code breaking if file is in current directory
        outDir = "."

    # Re-add any initially removed degenerate axes (to match with FITS header)
    # NOTE THIS HAS NOT BEEN RIGOROUSLY TESTED!!!
    output_axes = []
    for i in range(1, Ndim + 1):
        output_axes.append(header["NAXIS" + str(i)])  # Get FITS dimensions
    del output_axes[
        freq_axis - 1
    ]  # Remove frequency axis (since it's first in the array)
    output_axes.reverse()  # To get into numpy order.

    if "BUNIT" in header:
        header["BUNIT"] = header["BUNIT"] + "/RMSF"

    # Save the FDF
    if not not_rmsynth:
        header["NAXIS" + str(freq_axis)] = phiArr_radm2.size
        header["CDELT" + str(freq_axis)] = (
            np.diff(phiArr_radm2)[0],
            "[rad/m^2] Coordinate increment at reference point",
        )
        header["CRPIX" + str(freq_axis)] = phiArr_radm2.size // 2 + 1
        header["CRVAL" + str(freq_axis)] = (
            phiArr_radm2[phiArr_radm2.size // 2],
            "[rad/m^2] Coordinate value at reference point",
        )

        # Put frequency axis first, and reshape to add degenerate axes:
        FDFcube = np.reshape(FDFcube, [FDFcube.shape[0]] + output_axes)
        # Move Faraday depth axis to appropriate position to match header.
        FDFcube = np.moveaxis(FDFcube, 0, Ndim - freq_axis)

        if write_seperate_FDF:  # more memory efficient as well
            header = _setStokes(header, "Q")
            hdu0 = pf.PrimaryHDU(FDFcube.real.astype(dtFloat), header)
            fitsFileOut = outDir + "/" + prefixOut + "FDF_real_dirty.fits"
            if verbose:
                log("> %s" % fitsFileOut)
            hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)
            del hdu0
            gc.collect()

            header = _setStokes(header, "U")
            hdu1 = pf.PrimaryHDU(FDFcube.imag.astype(dtFloat), header)
            fitsFileOut = outDir + "/" + prefixOut + "FDF_im_dirty.fits"
            if verbose:
                log("> %s" % fitsFileOut)
            hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)
            del hdu1
            gc.collect()

            header = _setStokes(
                header, "PI"
            )  # Sets Stokes axis to zero, which is a non-standard value.
            del header["STOKES"]
            hdu2 = pf.PrimaryHDU(np.abs(FDFcube).astype(dtFloat), header)
            fitsFileOut = outDir + "/" + prefixOut + "FDF_tot_dirty.fits"
            if verbose:
                log("> %s" % fitsFileOut)
            hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)
            del hdu2
            gc.collect()

        else:
            header = _setStokes(header, "Q")
            hdu0 = pf.PrimaryHDU(FDFcube.real.astype(dtFloat), header)
            header = _setStokes(header, "U")
            hdu1 = pf.ImageHDU(FDFcube.imag.astype(dtFloat), header)
            header = _setStokes(
                header, "PI"
            )  # Sets Stokes axis to zero, which is a non-standard value.
            del header["STOKES"]
            hdu2 = pf.ImageHDU(np.abs(FDFcube).astype(dtFloat), header)

            # Save the dirty FDF
            fitsFileOut = outDir + "/" + prefixOut + "FDF_dirty.fits"
            if verbose:
                log("> %s" % fitsFileOut)
            hduLst = pf.HDUList([hdu0, hdu1, hdu2])
            hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
            hduLst.close()

    # Save the RMSF: real, im, tot (4D) and FWHM (2D)
    if not not_rmsf:
        # Create header for RMSF_{real,imag,tot}.fits

        # Put frequency axis first, and reshape to add degenerate Stokes axis:
        RMSFcube = np.reshape(RMSFcube, [RMSFcube.shape[0]] + output_axes)
        # Move Faraday depth axis to appropriate position to match header.
        RMSFcube = np.moveaxis(RMSFcube, 0, Ndim - freq_axis)

        # Header for outputs that are RMSF
        header["NAXIS" + str(freq_axis)] = phi2Arr_radm2.size
        header["CDELT" + str(freq_axis)] = (
            np.diff(phi2Arr_radm2)[0],
            "[rad/m^2] Coordinate increment at reference point",
        )
        header["CRPIX" + str(freq_axis)] = phi2Arr_radm2.size // 2 + 1
        header["CRVAL" + str(freq_axis)] = (
            phi2Arr_radm2[phi2Arr_radm2.size // 2],
            "[rad/m^2] Coordinate value at reference point",
        )
        header["BUNIT"] = ""

        # Create header for RMSF_FHWM.fits
        rmsffwhm_header = header.copy()
        rmsffwhm_header["BUNIT"] = "rad/m^2"
        rmsffwhm_header.pop("BTYPE", None)
        # Remove 3rd and 4th axis, RMSF_FWHM is a plane, like the 2D peak maps
        rmsffwhm_header = remove_header_third_fourth_axis(rmsffwhm_header)

        if write_seperate_FDF:  # more memory efficient as well
            header = _setStokes(header, "Q")
            hdu0 = pf.PrimaryHDU(RMSFcube.real.astype(dtFloat), header)
            fitsFileOut = outDir + "/" + prefixOut + "RMSF_real.fits"
            if verbose:
                log("> %s" % fitsFileOut)
            hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)
            del hdu0
            gc.collect()

            header = _setStokes(header, "U")
            hdu1 = pf.PrimaryHDU(RMSFcube.imag.astype(dtFloat), header)
            fitsFileOut = outDir + "/" + prefixOut + "RMSF_im.fits"
            if verbose:
                log("> %s" % fitsFileOut)
            hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)
            del hdu1
            gc.collect()

            header = _setStokes(
                header, "PI"
            )  # Sets Stokes axis to zero, which is a non-standard value.
            del header["STOKES"]
            hdu2 = pf.PrimaryHDU(np.abs(RMSFcube).astype(dtFloat), header)
            fitsFileOut = outDir + "/" + prefixOut + "RMSF_tot.fits"
            if verbose:
                log("> %s" % fitsFileOut)
            hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)
            del hdu2
            gc.collect()

            hdu3 = pf.PrimaryHDU(
                # np.expand_dims(fwhmRMSFCube.astype(dtFloat), axis=0), rmsffwhm_header
                fwhmRMSFCube.astype(dtFloat),
                rmsffwhm_header,
            )
            fitsFileOut = outDir + "/" + prefixOut + "RMSF_FWHM.fits"
            if verbose:
                log("> %s" % fitsFileOut)
            hdu3.writeto(fitsFileOut, output_verify="fix", overwrite=True)

        else:
            header = _setStokes(header, "Q")
            hdu0 = pf.PrimaryHDU(RMSFcube.real.astype(dtFloat), header)
            header = _setStokes(header, "U")
            hdu1 = pf.ImageHDU(RMSFcube.imag.astype(dtFloat), header)
            header = _setStokes(
                header, "PI"
            )  # Sets Stokes axis to zero, which is a non-standard value.
            del header["STOKES"]
            hdu2 = pf.ImageHDU(np.abs(RMSFcube).astype(dtFloat), header)
            hdu3 = pf.ImageHDU(
                # np.expand_dims(fwhmRMSFCube.astype(dtFloat), axis=0), rmsffwhm_header
                fwhmRMSFCube.astype(dtFloat),
                rmsffwhm_header,
            )

            fitsFileOut = outDir + "/" + prefixOut + "RMSF.fits"
            hduLst = pf.HDUList([hdu0, hdu1, hdu2, hdu3])
            if verbose:
                log("> %s" % fitsFileOut)
            hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
            hduLst.close()

    if not not_rmsynth and do_peakmaps:
        ## Note that peaks are computed from the sampled functions
        ## might be better to fit the FDF and compute the peak.
        ## See RMpeakfit_3D.py

        # Because there can be problems with different axes having different FITS keywords,
        # don't try to remove the FD axis, but just make it degenerate.
        # Also requires np.expand_dims to set the correct NAXIS.
        # Generate peak maps:

        ## Erik: THIS NOW INCONSISTENT WITH RMSF_FWHM (2D). PEAK MAPS STILL USE np.expand_dims
        ## but since do_peakmaps shouldnt be used anyways, I did not update the code below
        ## it still produces OK data though, just inconsistent in dimensions
        log("WARNING: dimensions of these peak maps are not 2D")

        maxPI, peakRM = create_peak_maps(FDFcube, phiArr_radm2, Ndim - freq_axis)
        # Save a maximum polarised intensity map
        if "BUNIT" in headtemplate:
            header["BUNIT"] = headtemplate["BUNIT"]
        header["NAXIS" + str(freq_axis)] = 1
        header["CTYPE" + str(freq_axis)] = (
            "DEGENERATE",
            "Axis left in to avoid FITS errors",
        )
        header["CUNIT" + str(freq_axis)] = ""

        # Header for output that are RM maps (peakRM, RMSF_FWHM, maxPI)
        header["NAXIS" + str(freq_axis)] = 1
        header["CRVAL" + str(freq_axis)] = (
            phiArr_radm2[0],
            "[rad/m^2] Coordinate value at reference point",
        )
        if "DATAMAX" in header:
            del header["DATAMAX"]
        if "DATAMIN" in header:
            del header["DATAMIN"]

        # Generate peak maps:

        maxPI, peakRM = create_peak_maps(FDFcube, phiArr_radm2, Ndim - freq_axis)
        # Save a maximum polarised intensity map
        if "BUNIT" in headtemplate:
            header["BUNIT"] = headtemplate["BUNIT"]
        header["NAXIS" + str(freq_axis)] = 1
        header["CTYPE" + str(freq_axis)] = (
            "DEGENERATE",
            "Axis left in to avoid FITS errors",
        )
        header["CUNIT" + str(freq_axis)] = ""

        stokes_axis = None
        for axis in range(1, header["NAXIS"] + 1):
            if "STOKES" in header[f"CTYPE{axis}"]:
                stokes_axis = axis
        if stokes_axis is not None:
            header[f"CTYPE{stokes_axis}"] = (
                "DEGENERATE",
                "Axis left in to avoid FITS errors",
            )

        fitsFileOut = outDir + "/" + prefixOut + "FDF_maxPI.fits"
        if verbose:
            log("> %s" % fitsFileOut)
        pf.writeto(
            fitsFileOut,
            np.expand_dims(maxPI.astype(dtFloat), axis=0),
            header,
            overwrite=True,
            output_verify="fix",
        )
        # Save a peak RM map
        fitsFileOut = outDir + "/" + prefixOut + "FDF_peakRM.fits"
        header["BUNIT"] = "rad/m^2"
        header["BTYPE"] = "FDEP"
        if verbose:
            log("> %s" % fitsFileOut)
        pf.writeto(
            fitsFileOut,
            np.expand_dims(peakRM, axis=0),
            header,
            overwrite=True,
            output_verify="fix",
        )

    #   #Cameron: I've removed the moment 1 map for now because I don't think it's properly/robustly defined.
    #    # Save an RM moment-1 map
    #    fitsFileOut = outDir + "/" + prefixOut + "FDF_mom1.fits"
    #    header["BUNIT"] = "rad/m^2"
    #    mom1FDFmap = (np.nansum(np.moveaxis(np.abs(FDFcube),FDFcube.ndim-freq_axis,FDFcube.ndim-1) * phiArr_radm2, FDFcube.ndim-1)
    #                  /np.nansum(np.abs(FDFcube), FDFcube.ndim-freq_axis))
    #    mom1FDFmap = mom1FDFmap.astype(dtFloat)
    #    if(verbose): log("> %s" % fitsFileOut)
    #    pf.writeto(fitsFileOut, mom1FDFmap, header, overwrite=True,
    #               output_verify="fix")


def _setStokes(header, stokes):
    """Check if header has Stokes axis. If so, set to correct numerical value
    (if IQUV). If not a valid Stokes parameter, sets to zero.
    Adds Stokes keyword regardless. Returns new, updated header.
    """
    stokes_dict = {"I": 1, "Q": 2, "U": 3, "V": 4}

    outheader = header.copy()
    stokes_axis = None
    for axis in range(1, header["NAXIS"] + 1):
        if "STOKES" in header[f"CTYPE{axis}"]:
            stokes_axis = axis

    if stokes_axis is not None:
        outheader[f"CRPIX{stokes_axis}"] = 1.0
        outheader[f"CRVAL{stokes_axis}"] = stokes_dict.get(stokes.upper(), 0)
    outheader["STOKES"] = stokes

    return outheader


def create_peak_maps(FDFcube, phiArr_radm2, phi_axis=0):
    """Finds the location and amplitude of the highest peak in the FDF (pixelwise)
    and returns maps of those parameters. Does not fit the peak, only finds
    the location in terms of the quantized Faraday depth slices of the cube.
    Used to produce the maxPI and peakRM maps.
    Inputs:
        FDFcube: output cube from run_rmsynth
        phiArr_radm2: array of Faraday depth values, from run_rmsynth
        phi_axis (int): number of the axis for Faraday depth (in python order,
                         not FITS order). Defaults to zero (first axis).
    Returns:
        maxPI: array of same dimensions as FDFcube exceppt collapsed along
                first (Faraday depth) axis, containing the maximum polarized
                intensity for each pixel
        peakRM: as maxPI, but with the Faraday depth location of the peak
    """

    maxPI = np.max(np.abs(FDFcube), axis=phi_axis)
    peakRM_indices = np.argmax(np.abs(FDFcube), axis=phi_axis)
    peakRM = phiArr_radm2[peakRM_indices]
    # Check for pixels with all NaNs across FD
    # Write peakRM as NaN (otherwise it takes the first entry in phiArr)
    nan_mask = np.all(np.isnan(FDFcube), axis=phi_axis)
    peakRM[nan_mask] = np.nan

    return maxPI, peakRM


def find_freq_axis(header):
    """Finds the frequency axis in a FITS header.
    Input: header: a Pyfits header object.
    Returns the axis number (as recorded in the FITS file, **NOT** in numpy ordering.)
    Returns 0 if the frequency axis cannot be found.
    """
    freq_axis = 0  # Default for 'frequency axis not identified'
    # Check for frequency axes. Because I don't know what different formatting
    # I might get ('FREQ' vs 'OBSFREQ' vs 'Freq' vs 'Frequency'), convert to
    # all caps and check for 'FREQ' anywhere in the axis name.
    for i in range(1, header["NAXIS"] + 1):  # Check each axis in turn.
        try:
            if "FREQ" in header["CTYPE" + str(i)].upper():
                freq_axis = i
        except:
            pass  # The try statement is needed for if the FITS header does not
            # have CTYPE keywords.
    return freq_axis


def readFitsCube(file, verbose, log=print):
    """The old version of this function could only accept 3 or 4 axis input
    (and implicitly assumed that in the 4 axis case that axis 3 was degenerate).
    I'm trying to somewhat generalize this, so that it will accept NAXIS=1..3
    cases and automatically try to identify which axis is the frequency axis,
    and will try to remove the degenerate axis in the 4D case.
    Where it can't find the correct frequency axis, it will assume it is the
    last one. It assumes any fourth or higher dimensions are degenerate (length 1)
    and will remove them. If the higher dimensions are NOT degenerate (e.g., a
    cube with all 4 Stokes), the code will fail (support may be added later?).
    -Cameron (3 April 2019)
    """
    if not os.path.exists(file):
        log("Err: File not found")

    if verbose:
        log("Reading " + file + " ...")
    data = pf.getdata(file)
    head = pf.getheader(file)
    if verbose:
        log("done.")

    N_dim = head["NAXIS"]  # Get number of axes
    if verbose:
        print("Dimensions of the input cube are: ", end=" ")
        for i in range(1, N_dim + 1):
            print("NAXIS{} = {}".format(i, head["NAXIS" + str(i)]), end="  ")
        print()

    freq_axis = find_freq_axis(head)
    # If the frequency axis isn't the last one, rotate the array until it is.
    # Recall that pyfits reverses the axis ordering, so we want frequency on
    # axis 0 of the numpy array.
    if freq_axis != 0 and freq_axis != N_dim:
        data = np.moveaxis(data, N_dim - freq_axis, 0)

    if N_dim >= 4:
        data = np.squeeze(data)  # Remove degenerate axes

    if verbose:
        print("Dimensions of the input array are: ", data.shape)

    if data.ndim > 3:
        raise Exception("Data cube has too many (non-degenerate) axes!")

    return head, data


def readFreqFile(file, verbose, log=print):
    # Read the frequency vector and wavelength sampling
    freqArr_Hz = np.loadtxt(file, dtype=float)
    return freqArr_Hz


# -----------------------------------------------------------------------------#
def main():
    import argparse

    """
    Start the function to perform RM-synthesis if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run RM-synthesis on a pair of Stokes Q and U cubes (3D). This script
    correctly deals with isolated clumps of flagged voxels in the cubes (NaNs).
    Saves cubes containing the complex Faraday dispersion function (FDF), a
    cube of double-size Rotation Measure Spread Functions, a peak Faraday
    depth map, a first-moment map and a maximum polarised intensity map.
    """

    epilog_text = """
    Output files:
        Default:
        FDF_real_dirty.fits: real (Stokes Q) component of the FDF
        FDF_im_dirty.fits: imaginary (Stokes U) component of the FDF
        FDF_tot_dirty.fits: polarized intnsity (Stokes P) component of the FDF
        FDF_maxPI.fits: 2D map of peak polarized intensity per pixel.
        FDF_peakRM.fits: 2D map of Faraday depth of highest peak, per pixel.
        RMSF_real_dirty.fits: real (Stokes Q) component of the RMSF
        RMSF_im_dirty.fits: imaginary (Stokes U) component of the RMSF
        RMSF_tot_dirty.fits: polarized intnsity (Stokes P) component of the RMSF
        RMSF_FWHM: 2D map of RMSF FWHM per pixel.

        With -f flag, the 3 FDF cubes are combined in a single file with 3 extensions.
            and the RMSF files are combined in a single file with 4 extensions.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr,
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "fitsQ",
        metavar="StokesQ.fits",
        nargs=1,
        help="FITS cube containing Stokes Q data.",
    )
    parser.add_argument(
        "fitsU",
        metavar="StokesU.fits",
        nargs=1,
        help="FITS cube containing Stokes U data.",
    )
    parser.add_argument(
        "freqFile",
        metavar="freqs_Hz.dat",
        nargs=1,
        help="ASCII file containing the frequency vector.",
    )
    parser.add_argument(
        "-i",
        dest="fitsI",
        default=None,
        help="FITS cube containing Stokes I model [None].",
    )
    parser.add_argument(
        "-n",
        dest="noiseFile",
        default=None,
        help="Text file containing channel noise values [None].",
    )
    parser.add_argument(
        "-w",
        dest="weightType",
        default="uniform",
        help="weighting ['uniform'] (all 1s) or 'variance'.",
    )
    parser.add_argument(
        "-t",
        dest="fitRMSF",
        action="store_true",
        help="Fit a Gaussian to the RMSF [False]",
    )
    parser.add_argument(
        "-l",
        dest="phiMax_radm2",
        type=float,
        default=None,
        help="Absolute max Faraday depth sampled (overrides NSAMPLES) [Auto].",
    )
    parser.add_argument(
        "-d",
        dest="dPhi_radm2",
        type=float,
        default=None,
        help="Width of Faraday depth channel [Auto].",
    )
    parser.add_argument(
        "-o",
        dest="prefixOut",
        default="",
        help="Prefix to prepend to output files [None].",
    )
    parser.add_argument(
        "-s",
        dest="nSamples",
        type=float,
        default=5,
        help="Number of samples across the FWHM RMSF.",
    )
    parser.add_argument(
        "-f",
        dest="write_seperate_FDF",
        action="store_false",
        help="Store different Stokes as FITS extensions [False, store as separate files].",
    )
    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="Verbose [False]."
    )
    parser.add_argument(
        "-R",
        dest="not_RMSF",
        action="store_true",
        help="Skip calculation of RMSF? [False]",
    )
    parser.add_argument(
        "-r",
        "--super-resolution",
        action="store_true",
        help="Optimise the resolution of the RMSF (as per Rudnick & Cotton). ",
    )
    args = parser.parse_args()

    # Sanity checks
    for f in args.fitsQ + args.fitsU:
        if not os.path.exists(f):
            print("File does not exist: '%s'." % f)
            sys.exit()
    dataDir, dummy = os.path.split(args.fitsQ[0])
    verbose = args.verbose
    if args.fitsI is not None:
        dataI = readFitsCube(args.fitsI, verbose)[1]
    else:
        dataI = None
    if args.noiseFile is not None:
        rmsArr = readFreqFile(args.noiseFile, verbose)
    else:
        rmsArr = None

    header, dataQ = readFitsCube(args.fitsQ[0], verbose)

    # Run RM-synthesis on the cubes
    dataArr = run_rmsynth(
        dataQ=dataQ,
        dataU=readFitsCube(args.fitsU[0], verbose)[1],
        freqArr_Hz=readFreqFile(args.freqFile[0], verbose),
        dataI=dataI,
        rmsArr=rmsArr,
        phiMax_radm2=args.phiMax_radm2,
        dPhi_radm2=args.dPhi_radm2,
        nSamples=args.nSamples,
        weightType=args.weightType,
        fitRMSF=args.fitRMSF,
        nBits=32,
        verbose=verbose,
        not_rmsf=args.not_RMSF,
        super_resolution=args.super_resolution,
    )

    # Write to files
    writefits(
        dataArr,
        headtemplate=header,
        fitRMSF=False,
        prefixOut=args.prefixOut,
        outDir=dataDir,
        write_seperate_FDF=args.write_seperate_FDF,
        not_rmsf=args.not_RMSF,
        nBits=32,
        verbose=verbose,
    )


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
