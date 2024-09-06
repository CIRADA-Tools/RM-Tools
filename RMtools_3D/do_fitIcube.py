#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     do_fitIcube.py                                                    #
#                                                                             #
# PURPOSE:  Make a model Stokes I cube and a noise vector.                    #
#                                                                             #
# MODIFIED: 26-Feb-2017 by C. Purcell
# MODIFIED: 18 January 2023 by Lerato Baidoo  (re-structured and optimized)   #
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

import argparse
import multiprocessing as mp
import os
import sys
import time
from functools import partial

import astropy.io.fits as pf
import numpy as np
from tqdm.contrib.concurrent import process_map

from RMtools_3D.do_RMsynth_3D import readFitsCube
from RMtools_3D.make_freq_file import get_freq_array
from RMutils.util_FITS import strip_fits_dims
from RMutils.util_misc import (
    MAD,
    calculate_StokesI_model,
    fit_StokesI_model,
    remove_header_third_fourth_axis,
)

# -----------------------------------------------------------------------------#


def main():
    """
    Start the make_model_I function if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Create a model Stokes I dataset by fitting a polynomial to emitting regions
    above a cutoff threshold in the Stokes I cube. Also outputs a noise spectrum
    with the Stokes I noise per channel.

    NOTE: Each pixel is fit independently, so there are no protections in place
    to ensure smoothness across the image-plane. Noise levels are estimated
    per-channel using 2 passes of the MAD.
    The source mask, if applied, is calculated per-pixel by looking for values
    above a threshold (either an absolute intensity threshold or a multiple of
    the noise) in any channel.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "fitsI",
        metavar="StokesI.fits",
        nargs=1,
        help="FITS cube containing Stokes I data.",
    )
    parser.add_argument(
        "-freqFile",
        dest="freq_file",
        default="",
        type=str,
        help="Path + ASCII file containing the frequency vector. If not provided,\nfrequencies are derived from fits header.",
    )
    parser.add_argument(
        "-f",
        dest="fit_function",
        type=str,
        default="log",
        help="Stokes I fitting function: 'linear' or ['log'] polynomials.",
    )
    parser.add_argument(
        "-p",
        dest="polyOrd",
        type=int,
        default=2,
        help="polynomial order to fit to I spectrum: 0-5 supported, 2 is default.\nSet to negative number to enable dynamic order selection.",
    )
    parser.add_argument(
        "-n",
        dest="num_cores",
        type=int,
        default=1,
        help="Number of cores to use for multiprocessing. Default is 1.",
    )
    parser.add_argument(
        "-m",
        dest="apply_mask",
        action="store_true",
        help="Apply masking before spectral fitting. Default is False.",
    )
    parser.add_argument(
        "-c",
        dest="chunk_size",
        type=int,
        default=1,
        help="Chunk size for multiprocessing. Default is 1.",
    )
    parser.add_argument(
        "-t",
        dest="threshold",
        type=float,
        default=-3,
        help="Source masking threshold in flux units (if positive) or factors of per-channel sigma (if negative). Default is -3 (i.e. 3 sigma mask).",
    )
    parser.add_argument(
        "-o",
        dest="prefixOut",
        default="",
        help="Prefix to use for to output file names.",
    )
    parser.add_argument(
        "-odir",
        dest="outDir",
        default="",
        help="Output directory to save output files. If none, save inside input directory",
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="turn on verbose messages [False].",
    )
    args = parser.parse_args()

    # Sanity checks
    for f in [args.fitsI[0]]:
        if not os.path.exists(f):
            print("File does not exist: '%s'." % f)
            sys.exit()
    out_directory = args.outDir
    if not out_directory:
        out_directory, dummy = os.path.split(args.fitsI[0])

    I_filename = args.fitsI[0]
    datacube, headI = open_datacube(fitsI=I_filename, verbose=args.verbose)

    # if frequency file is not provided, extract frequency information from the input fits header.
    if args.freq_file:
        freqArr_Hz = get_frequencies(
            datacube=datacube, header=headI, freqFile=args.freq_file
        )
    else:
        print("Frequency file not provided. Deriving frequencies from the fits header.")
        freqArr_Hz = get_freq_array(I_filename)

    # Run polynomial fitting on the spectra
    make_model_I(
        datacube=datacube,
        header=headI,
        freqArr_Hz=freqArr_Hz,
        polyOrd=args.polyOrd,
        prefixOut=args.prefixOut,
        outDir=out_directory,
        nBits=32,
        threshold=args.threshold,
        apply_mask=args.apply_mask,
        num_cores=args.num_cores,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        fit_function=args.fit_function,
    )


# -----------------------------------------------------------------------------#


def open_datacube(fitsI, verbose=False):
    """Reads the image fits

    Parameters:
    fitsI : Input Stokes I cube fits image
    verbose: If true, write logs

    Returns:
    datacube: Image data
    header: Fits header
    """

    # Default data type

    # Sanity check on header dimensions
    print("Reading FITS cube header from '%s':" % fitsI)
    header, datacube = readFitsCube(fitsI, verbose)

    nDim = datacube.ndim
    if nDim < 3 or nDim > 4:
        print("Err: only 3 or 4 dimensions supported: D = %d." % header["NAXIS"])
        sys.exit()

    # freq_axis=find_freq_axis(header)
    # #If the frequency axis isn't the last one, rotate the array until it is.
    # #Recall that pyfits reverses the axis ordering, so we want frequency on
    # #axis 0 of the numpy array.
    # if freq_axis != 0 and freq_axis != nDim:
    #     datacube=np.moveaxis(datacube,nDim-freq_axis,0)

    return datacube, header


def get_frequencies(datacube, header, freqFile):
    """Reads a frequency file

    Parameters:
    datacube: Image cube data
    header: A header of the input datacube
    freqFile: A frequency file in text format

    Returns:
    freqArr_Hz: frequency array
    """

    nBits = np.abs(header["BITPIX"])
    dtFloat = "float" + str(nBits)

    nChan = datacube.shape[0]  # for now, assumes frequency is the first axis

    # Read the frequency vector
    print("Reading frequency vector from '%s'." % freqFile)
    freqArr_Hz = np.loadtxt(freqFile, dtype=dtFloat)
    if nChan != len(freqArr_Hz):
        print("Err: frequency vector and frequency axis of cube unequal length.")
        sys.exit()

    return freqArr_Hz


def cube_noise(datacube, header, freqArr_Hz, threshold=-5):
    """Estimates noise of each channel in an image cube.

    Parameters:
    datacube: Input cube data
    header : Header of a cube image
    frequency: Frequency values of a cube image in Hz
    threshold: Threshold to use for masking off pixels.
            If the value is +ve, then it is taken as absolute value, and
            -ve as sigma. E.g. threshold=-5, means consider pixels
            with emission > noise_median - 5 * rms_noise.
            Where the noise_median and rms_noise are obtained by computing
            the median and MAD of the pixel emission > image median + 3 * image MAD.

    Returns:
    rms_Arr: An array containing rms values of each channel
    mskSrc:  A 2D image data containing masking values (0s and 1s)
    """
    nBits = np.abs(header["BITPIX"])
    dtFloat = "float" + str(nBits)
    nChan = datacube.shape[0]

    if nChan != len(freqArr_Hz):
        print("Err: frequency vector and frequency axis of cube unequal length.")
        sys.exit()

    # Measure the RMS spectrum using 2 passes of MAD on each plane
    # Determine which pixels have emission above the threshold
    print("Measuring the per-channel noise and creating an emission mask")
    rmsArr = np.zeros_like(freqArr_Hz)
    medSky = np.zeros_like(freqArr_Hz)
    mskSrc = np.zeros((header["NAXIS2"], header["NAXIS1"]), dtype=dtFloat)

    # start = time.time()
    for i in range(nChan):
        dataPlane = datacube[i]
        if np.isnan(dataPlane).all():
            # If this plane is fully flagged, dont have to calculate
            medSky[i] = np.nan
            rmsArr[i] = np.nan

        else:
            if threshold > 0:
                idxSky = np.where(
                    dataPlane < threshold
                )  # replaced cutoff with threshold
            else:
                idxSky = np.where(dataPlane)

            # Pass 1
            rmsTmp = MAD(dataPlane[idxSky])
            medTmp = np.nanmedian(dataPlane[idxSky])

            # Pass 2: use a fixed 3-sigma cutoff to mask off emission
            idxSky = np.where(dataPlane < medTmp + rmsTmp * 3)
            medSky[i] = np.nanmedian(dataPlane[idxSky])
            rmsArr[i] = MAD(dataPlane[idxSky])

            # When building final emission mask treat +ve threshold as absolute
            # values and negative threshold as sigma values
            if threshold > 0:
                idxSrc = np.where(dataPlane > threshold)
            else:
                idxSrc = np.where(dataPlane > medSky[i] - 1 * rmsArr[i] * threshold)

            mskSrc[idxSrc] += 1

    # end = time.time()
    # print(' For loop masking takes %.3fs'%(end-start))
    return rmsArr, mskSrc


def savefits_mask(data, header, outDir, prefixOut, dtFloat):
    """Save the derived mask to a fits file

    data:  2D data defining the mask.
    header: header to describe the mask
    outDir: directory to save the mask fits data
    prefixOut: prefix to use on the output name
    dtFloat: type to use for output file
    """

    headMask = remove_header_third_fourth_axis(header=header)
    headMask["DATAMAX"] = 1
    headMask["DATAMIN"] = 0
    if "BUNIT" in headMask:
        del headMask["BUNIT"]

    mskArr = np.where(data > 0, 1.0, np.nan).astype(dtFloat)
    MaskfitsFile = os.path.join(outDir, prefixOut + "mask.fits")
    pf.writeto(MaskfitsFile, mskArr, headMask, output_verify="fix", overwrite=True)


def savefits_Coeffs(data, dataerr, header, polyOrd, outDir, prefixOut):
    """Save the derived coefficients to a fits file

    data: 2D planes containing coeffs values.
    dataerr: 2D planes containing error in coeffs values.
    header: header to describe the coeffs and error in coeffs
    polyOrd: the order of polynomial to fit
    outDir: directory to save the (errors) coeffs fits data
    prefixOut: prefix to use on the output name
    """

    header["BUNIT"] = ""
    if "BTYPE" in header:
        del header["BTYPE"]

    for i in range(np.abs(polyOrd) + 1):
        outname = os.path.join(outDir, prefixOut + "coeff" + str(i) + ".fits")
        pf.writeto(outname, data[i], header, overwrite=True)

        outname = os.path.join(outDir, prefixOut + "coeff" + str(i) + "err.fits")
        pf.writeto(outname, dataerr[i], header, overwrite=True)


def savefits_model_I(data, header, outDir, prefixOut):
    """Save the derived Stokes cube model

    data:  Stokes I cube model data
    header: header to describe the model cube.
    outDir: directory to save the model cube fits data
    prefixOut: prefix to use on the output name
    """

    nDim = data.ndim
    nBits = np.abs(header["BITPIX"])

    headModelCube = strip_fits_dims(header=header, minDim=nDim)
    headModelCube["NAXIS1"] = header["NAXIS1"]
    headModelCube["NAXIS2"] = header["NAXIS2"]
    headModelCube["NAXIS3"] = header["NAXIS3"]

    nVoxels = header["NAXIS1"] * header["NAXIS2"] * header["NAXIS3"]
    if nDim == 4:
        headModelCube["NAXIS4"] = header["NAXIS4"]
        nVoxels *= header["NAXIS4"]
    while len(headModelCube) < (36 * 4 - 1):
        headModelCube.append()

    fitsModelFile = os.path.join(outDir, prefixOut + "model.i.fits")
    headModelCube.tofile(fitsModelFile, overwrite=True)
    with open(fitsModelFile, "rb+") as f:
        f.seek(len(headModelCube.tostring()) + (nVoxels * int(nBits / 8)) - 1)
        f.write(b"\0")
    HDULst = pf.open(fitsModelFile, "update", memmap=True)
    HDULst[0].data = data
    HDULst.close()


def fit_spectra_I(
    Ispectrum, freqArr_Hz, rms_Arr, polyOrd, fit_function, nDetectPix, verbose=False
):
    """Fits polynomial function to Stokes I data

    xy: Position of pixel to fit (in pixels).
        The xy consists of pixel number, x and y pixel position.
    datacube:  Stokes I data cube to model.
    freqArr_Hz: Frequency array in Hz.
    rms_Arr: An array containing rms values of each channel.
    polyOrd: the order of polynomial to fit.
    fit_function: A type of function to fit.
         It can be log or linear.
    nDetectPix:  the total number of pixels to be fit.
    """

    pix_fit_result = fit_StokesI_model(
        freqArr_Hz, Ispectrum, rms_Arr, polyOrd=polyOrd, fit_function=fit_function
    )

    pixImodel = calculate_StokesI_model(pix_fit_result, freqArr_Hz)

    outs = dict()

    outs["I"] = pixImodel.astype("float32")
    outs["coeffs"] = pix_fit_result.params.astype("float32")
    outs["coeffs_err"] = pix_fit_result.perror.astype("float32")
    outs["chiSq"] = pix_fit_result.chiSq
    outs["chiSqRed"] = pix_fit_result.chiSqRed
    outs["nIter"] = pix_fit_result.nIter
    outs["AIC"] = pix_fit_result.AIC
    outs["covar"] = pix_fit_result.pcov
    outs["reference_frequency_Hz"] = pix_fit_result.reference_frequency_Hz

    return outs


def make_model_I(
    datacube,
    header,
    freqArr_Hz,
    polyOrd=2,
    nBits=32,
    threshold=3,
    num_cores=1,
    chunk_size=1,
    verbose=False,
    fit_function="log",
    apply_mask=False,
    outDir=None,
    prefixOut=None,
):
    """Fits a polynomial function to Stokes I data, derives coefficients,
       predicts model I, and save the respective fits file.

    datacube:  Stokes I data cube
    header: header of the data cube
    freqArr_Hz: frequency values of the cube in Hz
    polyOrd: the order of the polynomial to fit. 0-5 supported, 2 is default
    fit_function: fit log or linear

    num_cores: Number of cores to use for parallel processing
    chunk_size: Chunk size for multiprocessing
    verbose: Write to log
    apply_mask: If true, a mask will be applied
    threshold: Threshold to use for masking off pixels

            If the value is +ve, then it is taken as absolute value, and
            -ve as sigma. E.g. threshold=-5, means consider pixels
            with emission > noise_median - 5 * rms_noise.
            Where the noise_median and rms_noise are obtained by computing
            the median and MAD of the pixel emission > image median + 3 * image MAD
    outDir: Directory to save all outputs
    prefixOut: Prefix name to use in all output names

    Returns:
    ModelIcube: Model I cube data array
    Mask fits data
    Coefficients fits data
    Error in coefficients fits data
    """
    dtFloat = "float" + str(nBits)

    rms_Arr, mskSrc = cube_noise(datacube, header, freqArr_Hz, threshold=threshold)

    mskArr = np.where(mskSrc > 0, 1.0, np.nan)

    if not apply_mask:
        mskSrc = np.ones((header["naxis2"], header["naxis1"]), dtype=dtFloat)
        mskArr = np.where(mskSrc > 0, 1.0, np.nan)

    srcCoords = np.rot90(np.where(mskSrc > 0))

    nPix = mskSrc.shape[-1] * mskSrc.shape[-2]
    nDetectPix = len(srcCoords)

    if verbose and apply_mask:
        print(
            "Emission present in %d spectra (%.1f percent)."
            % (nDetectPix, (nDetectPix * 100.0 / nPix))
        )

    modelIcube = np.zeros_like(datacube)
    modelIcube[:] = np.nan
    results = []

    coeffs = np.array([mskArr] * 6, dtype=dtFloat)
    coeffs_error = np.array([mskArr] * 6, dtype=dtFloat)
    reffreq = np.squeeze(np.array([mskArr], dtype=dtFloat))

    covars = np.array([[mskArr] * 6] * 6, dtype=dtFloat)
    datacube = np.squeeze(datacube)

    # Select only the spectra with emission
    srcData = np.rot90(datacube[:, mskSrc > 0])

    # Inform user job magnitude
    startTime = time.time()

    xy = list(zip(np.arange(0, len(srcCoords)), srcCoords[:, 0], srcCoords[:, 1]))

    if verbose:
        print("Fitting %d/%d spectra." % (nDetectPix, nPix))
    if verbose:
        print(
            f"Using {num_cores} cores and chunksize {chunk_size} for parallel processing."
        )

    func = partial(
        fit_spectra_I,
        freqArr_Hz=freqArr_Hz,
        rms_Arr=rms_Arr,
        polyOrd=polyOrd,
        fit_function=fit_function,
        nDetectPix=nDetectPix,
        verbose=verbose,
    )

    # Send each spectrum to a different core
    if verbose:  # Note that 'verbose' is not compatible with Prefect
        results = process_map(
            func,
            srcData,
            max_workers=num_cores,
            chunksize=chunk_size,
            disable=not verbose,
            desc="Fitting spectra",
            total=nDetectPix,
        )
    else:
        mp.set_start_method("spawn", force=True)
        args_list = [d for d in srcData]
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(func, args_list)

    headcoeff = remove_header_third_fourth_axis(header=header.copy())
    del headcoeff["BUNIT"]

    endTime = time.time()
    cputime = endTime - startTime
    print("Fitting completed in %.2f seconds." % cputime)

    for _, an in enumerate(xy):
        i, x, y = an

        modelIcube[:, x, y] = results[i]["I"]
        reffreq[x, y] = results[i]["reference_frequency_Hz"]
        covars[:, :, x, y] = results[i]["covar"]

        for k, j, l in zip(
            range(len(coeffs)), results[i]["coeffs"], results[i]["coeffs_err"]
        ):
            coeffs[5 - k, x, y] = j
            coeffs_error[5 - k, x, y] = l

    headcoeff["HISTORY"] = "Stokes I model fitted by RM-Tools"
    if polyOrd < 0:
        headcoeff["HISTORY"] = (
            f"Fit model is dynamic order {fit_function}-polynomial, max order {-polyOrd}"
        )
    else:
        headcoeff["HISTORY"] = f"Fit model is {polyOrd}-order {fit_function}-polynomial"

    if verbose:
        print("Saving mask image.")
    savefits_mask(
        data=mskSrc,
        header=headcoeff,
        outDir=outDir,
        prefixOut=prefixOut,
        dtFloat=dtFloat,
    )

    if verbose:
        print("Saving model I coefficients.")
    savefits_Coeffs(
        data=coeffs,
        dataerr=coeffs_error,
        header=headcoeff,
        polyOrd=polyOrd,
        outDir=outDir,
        prefixOut=prefixOut,
    )

    # Save frequency map
    head_freq = headcoeff.copy()
    head_freq["BUNIT"] = "Hz"
    if "BTYPE" in headcoeff:
        del headcoeff["BTYPE"]

    outname = os.path.join(outDir, prefixOut + "reffreq.fits")
    pf.writeto(outname, reffreq, head_freq, overwrite=True)

    # Save covariance maps -- these are necessary if/when converting the model
    # reference frequency.
    # Structure will be a single file as a 4D cube, with the 3rd and 4th dimensions
    # iterating over the two axes of the covariance matrix.
    head_covar = headcoeff.copy()
    head_covar["NAXIS"] = 4
    head_covar["NAXIS3"] = 6
    head_covar["NAXIS4"] = 6
    head_covar["CTYPE3"] = "INDEX"
    head_covar["CTYPE4"] = "INDEX"
    head_covar["CRVAL3"] = 0
    head_covar["CRVAL4"] = 0
    head_covar["CDELT3"] = 1
    head_covar["CDELT4"] = 1
    head_covar["CRPIX3"] = 1
    head_covar["CRPIX4"] = 1
    head_covar["CUNIT3"] = ""
    head_covar["CUNIT4"] = ""

    outname = os.path.join(outDir, prefixOut + "covariance.fits")
    pf.writeto(outname, covars, head_covar, overwrite=True)

    if verbose:
        print("Saving model I cube image. ")
    savefits_model_I(data=modelIcube, header=header, outDir=outDir, prefixOut=prefixOut)

    np.savetxt(os.path.join(outDir, prefixOut + "noise.dat"), rms_Arr)

    return modelIcube


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
