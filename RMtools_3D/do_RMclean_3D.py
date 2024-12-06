#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     do_RM-clean.py                                                    #
#                                                                             #
# PURPOSE:  Run RM-clean on a  cube of dirty Faraday dispersion functions.    #
#                                                                             #
# MODIFIED: 15-May-2016 by C. Purcell                                         #
# MODIFIED: 23-October-2019 by A. Thomson                                     #
#                                                                             #
# =============================================================================#
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
# =============================================================================#

import os
import sys
import time

import astropy.io.fits as pf
import numpy as np

try:
    import schwimmbad

    parallel_available = True
except:
    parallel_available = False

from RMtools_3D.do_RMsynth_3D import _setStokes
from RMutils.util_RM import do_rmclean_hogbom, fits_make_lin_axis


# -----------------------------------------------------------------------------#
def run_rmclean(
    fitsFDF,
    fitsRMSF,
    cutoff,
    maxIter=1000,
    gain=0.1,
    nBits=32,
    pool=None,
    chunksize=None,
    verbose=True,
    log=print,
    window=np.nan,
):
    """Run RM-CLEAN on a 2/3D FDF cube given an RMSF cube stored as FITS.

    If you want to run RM-CLEAN on arrays, just use util_RM.do_rmclean_hogbom.

    Args:
        fitsFDF (str): Name of FDF FITS file.
        fitsRMSF (str): Name of RMSF FITS file
        cutoff (float): CLEAN cutoff in flux units

    Kwargs:
        maxIter (int): Maximum number of CLEAN iterations per pixel.
        gain (float): CLEAN loop gain.
        nBits (int): Precision of floating point numbers.
        pool (multiprocessing Pool): Pool function from multiprocessing
            or schwimmbad
        chunksize (int): Number of chunks which it submits to the process pool
            as separate tasks. The (approximate) size of these chunks can be
            specified by setting chunksize to a positive integer.
        verbose (bool): Verbosity.
        log (function): Which logging function to use.

    Returns:
        cleanFDF (ndarray): Cube of RMCLEANed FDFs.
        ccArr (ndarray): Cube of RMCLEAN components (i.e. the model).
        iterCountArr (ndarray): Cube of number of RMCLEAN iterations.
        residFDF (ndarray): Cube of residual RMCLEANed FDFs.
        head (fits.header): Header of FDF FITS file for template.

    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    # Read the FDF
    dirtyFDF, head, FD_axis = read_FDF_cubes(fitsFDF)

    phiArr_radm2 = fits_make_lin_axis(head, axis=FD_axis - 1, dtype=dtFloat)

    # Read the RMSF

    RMSFArr, headRMSF, FD_axis = read_FDF_cubes(fitsRMSF)
    HDULst = pf.open(
        fitsRMSF.replace("_real", "_FWHM")
        .replace("_im", "_FWHM")
        .replace("_tot", "_FWHM"),
        "readonly",
        memmap=True,
    )
    fwhmRMSFArr = np.squeeze(HDULst[0].data)
    HDULst.close()
    phi2Arr_radm2 = fits_make_lin_axis(headRMSF, axis=FD_axis - 1, dtype=dtFloat)

    startTime = time.time()

    # Do the clean
    cleanFDF, ccArr, iterCountArr, residFDF = do_rmclean_hogbom(
        dirtyFDF=dirtyFDF,
        phiArr_radm2=phiArr_radm2,
        RMSFArr=RMSFArr,
        phi2Arr_radm2=phi2Arr_radm2,
        fwhmRMSFArr=fwhmRMSFArr,
        cutoff=cutoff,
        maxIter=maxIter,
        gain=gain,
        verbose=verbose,
        doPlots=False,
        pool=pool,
        chunksize=chunksize,
        window=window,
    )

    endTime = time.time()
    cputime = endTime - startTime
    if verbose:
        log("> RM-clean completed in %.2f seconds." % cputime)
    if verbose:
        log("Saving the clean FDF and ancillary FITS files")

    # Move FD axis back to original position, and restore dimensionality:
    old_Ndim, FD_axis = find_axes(head)
    new_Ndim = cleanFDF.ndim
    # The difference is the number of dimensions that need to be added:
    if old_Ndim - new_Ndim != 0:  # add missing (degenerate) dimensions back in
        cleanFDF = np.expand_dims(cleanFDF, axis=tuple(range(old_Ndim - new_Ndim)))
        ccArr = np.expand_dims(ccArr, axis=tuple(range(old_Ndim - new_Ndim)))
        residFDF = np.expand_dims(residFDF, axis=tuple(range(old_Ndim - new_Ndim)))
    # New dimensions are added to the beginning of the axis ordering
    # (revserse of FITS ordering)

    # Move the FDF axis to it's original spot. Hopefully this means that all
    # axes are in their original place after all of that.
    cleanFDF = np.moveaxis(cleanFDF, old_Ndim - new_Ndim, old_Ndim - FD_axis)
    ccArr = np.moveaxis(ccArr, old_Ndim - new_Ndim, old_Ndim - FD_axis)
    residFDF = np.moveaxis(residFDF, old_Ndim - new_Ndim, old_Ndim - FD_axis)

    return cleanFDF, ccArr, iterCountArr, residFDF, head


def writefits(
    cleanFDF,
    ccArr,
    iterCountArr,
    residFDF,
    headtemp,
    nBits=32,
    prefixOut="",
    outDir="",
    write_separate_FDF=False,
    verbose=True,
    log=print,
):
    """Write data to disk in FITS


    Output files:
        Default:
            FDF_clean.fits: RMCLEANed FDF, in 3 extensions: Q,U, and PI.
            FDF_CC.fits: RMCLEAN components, in 3 extensions: Q,U, and PI.
            CLEAN_nIter.fits: RMCLEAN iterations.

        write_seperate_FDF=True:
            FDF_clean_real.fits and FDF_CC.fits are split into
            three constituent components:
                FDF_clean_real.fits: Stokes Q
                FDF_clean_im.fits: Stokes U
                FDF_clean_tot.fits: Polarized Intensity (sqrt(Q^2+U^2))
                FDF_CC_real.fits: Stokes Q
                FDF_CC_im.fits: Stokes U
                FDF_CC_tot.fits: Polarized Intensity (sqrt(Q^2+U^2))
                CLEAN_nIter.fits: RMCLEAN iterations.
    Args:
        cleanFDF (ndarray): Cube of RMCLEANed FDFs.
        ccArr (ndarray): Cube of RMCLEAN components (i.e. the model).
        iterCountArr (ndarray): Cube of number of RMCLEAN iterations.
        residFDF (ndarray): Cube of residual RMCLEANed FDFs.

    Kwargs:
        prefixOut (str): Prefix for filenames.
        outDir (str): Directory to save files.
        write_seperate_FDF (bool): Write Q, U, and PI separately?
        verbose (bool): Verbosity.
        log (function): Which logging function to use.
    """
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)
    header = headtemp.copy()

    header["HISTORY"] = "RM-CLEAN3D output from RM-Tools"

    if outDir == "":  # To prevent code breaking if file is in current directory
        outDir = "."
    # Save the clean FDF
    if not write_separate_FDF:
        header = _setStokes(header, "Q")
        hdu0 = pf.PrimaryHDU(cleanFDF.real.astype(dtFloat), header)
        header = _setStokes(header, "U")
        hdu1 = pf.ImageHDU(cleanFDF.imag.astype(dtFloat), header)
        header = _setStokes(
            header, "PI"
        )  # Sets Stokes axis to zero, which is a non-standard value.
        del header["STOKES"]
        hdu2 = pf.ImageHDU(np.abs(cleanFDF).astype(dtFloat), header)

        fitsFileOut = outDir + "/" + prefixOut + "FDF_clean.fits"
        if verbose:
            log("> %s" % fitsFileOut)
        hduLst = pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        hduLst.close()
    else:
        header = _setStokes(header, "Q")
        hdu0 = pf.PrimaryHDU(cleanFDF.real.astype(dtFloat), header)
        header = _setStokes(header, "U")
        hdu1 = pf.PrimaryHDU(cleanFDF.imag.astype(dtFloat), header)
        header = _setStokes(
            header, "PI"
        )  # Sets Stokes axis to zero, which is a non-standard value.
        del header["STOKES"]
        hdu2 = pf.PrimaryHDU(np.abs(cleanFDF).astype(dtFloat), header)

        fitsFileOut = outDir + "/" + prefixOut + "FDF_clean_real.fits"
        hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if verbose:
            log("> %s" % fitsFileOut)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_clean_im.fits"
        hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if verbose:
            log("> %s" % fitsFileOut)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_clean_tot.fits"
        hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if verbose:
            log("> %s" % fitsFileOut)

    if not write_separate_FDF:
        # Save the complex clean components as another file.
        header = _setStokes(header, "Q")
        hdu0 = pf.PrimaryHDU(ccArr.real.astype(dtFloat), header)
        header = _setStokes(header, "U")
        hdu1 = pf.ImageHDU(ccArr.imag.astype(dtFloat), header)
        header = _setStokes(
            header, "PI"
        )  # Sets Stokes axis to zero, which is a non-standard value.
        del header["STOKES"]
        hdu2 = pf.ImageHDU(np.abs(ccArr).astype(dtFloat), header)

        fitsFileOut = outDir + "/" + prefixOut + "FDF_CC.fits"
        if verbose:
            log("> %s" % fitsFileOut)
        hduLst = pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        hduLst.close()
    else:
        header = _setStokes(header, "Q")
        hdu0 = pf.PrimaryHDU(ccArr.real.astype(dtFloat), header)
        header = _setStokes(header, "U")
        hdu1 = pf.PrimaryHDU(ccArr.imag.astype(dtFloat), header)
        header = _setStokes(
            header, "PI"
        )  # Sets Stokes axis to zero, which is a non-standard value.
        del header["STOKES"]
        hdu2 = pf.PrimaryHDU(np.abs(ccArr).astype(dtFloat), header)

        fitsFileOut = outDir + "/" + prefixOut + "FDF_CC_real.fits"
        hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if verbose:
            log("> %s" % fitsFileOut)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_CC_im.fits"
        hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if verbose:
            log("> %s" % fitsFileOut)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_CC_tot.fits"
        hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if verbose:
            log("> %s" % fitsFileOut)

    # Because there can be problems with different axes having different FITS keywords,
    # don't try to remove the FD axis, but just make it degenerate.

    if headtemp["NAXIS"] > 2:
        header["NAXIS3"] = 1
        header["CTYPE3"] = ("DEGENERATE", "Axis left in to avoid FITS errors")
        header["CUNIT3"] = ""
    if headtemp["NAXIS"] == 4:
        header["NAXIS4"] = 1
        header["CTYPE4"] = ("DEGENERATE", "Axis left in to avoid FITS errors")
        header["CUNIT4"] = ""

    # Save the iteration count mask
    fitsFileOut = outDir + "/" + prefixOut + "CLEAN_nIter.fits"
    if verbose:
        log("> %s" % fitsFileOut)
    header["BUNIT"] = "Iterations"
    if "STOKES" in header:
        del header["STOKES"]
    hdu0 = pf.PrimaryHDU(
        np.expand_dims(
            iterCountArr.astype(dtFloat),
            axis=tuple(range(headtemp["NAXIS"] - iterCountArr.ndim)),
        ),
        header,
    )
    hduLst = pf.HDUList([hdu0])
    hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
    hduLst.close()


# Old method (for multi-extension files)
def read_FDF_cube(filename):
    """Read in a FDF/RMSF cube. Figures out which axis is Faraday depth and
    puts it first (in numpy order) to accommodate the rest of the code.
    Returns: (complex_cube, header,FD_axis)
    """
    HDULst = pf.open(filename, "readonly", memmap=True)
    head = HDULst[0].header.copy()
    FDFreal = HDULst[0].data
    FDFimag = HDULst[1].data
    complex_cube = FDFreal + 1j * FDFimag

    # Identify Faraday depth axis (assumed to be last one if not explicitly found)
    Ndim = head["NAXIS"]
    FD_axis = Ndim
    # Check for FD axes:
    for i in range(1, Ndim + 1):
        try:
            if "FDEP" in head["CTYPE" + str(i)].upper():
                FD_axis = i
        except:
            pass  # The try statement is needed for if the FITS header does not
            # have CTYPE keywords.

    # Move FD axis to first place in numpy order.
    if FD_axis != Ndim:
        complex_cube = np.moveaxis(complex_cube, Ndim - FD_axis, 0)

    # Remove degenerate axes to prevent problems with later steps.
    complex_cube = complex_cube.squeeze()

    return complex_cube, head, FD_axis


def find_axes(header):
    """Idenfities how many axes are present in a FITS file, and which is the
    Faraday depth axis. Necessary for bookkeeping on cube dimensionality,
    given that RM-clean only supports 3D cubes, but data may be 4D files."""
    Ndim = header["NAXIS"]
    FD_axis = Ndim
    # Check for FD axes:
    for i in range(1, Ndim + 1):
        try:
            if "FDEP" in header["CTYPE" + str(i)].upper():
                FD_axis = i
        except:
            pass  # The try statement is needed for if the FITS header does not
            # have CTYPE keywords.
    return Ndim, FD_axis


def read_FDF_cubes(filename):
    """Read in a FDF/RMSF cube. Input filename can be any of real, imag, or tot components.
    Figures out which axis is Faraday depth and
    puts it first (in numpy order) to accommodate the rest of the code.
    Returns: (complex_cube, header,FD_axis)
    """
    HDUreal = pf.open(
        filename.replace("_tot", "_real").replace("_im", "_real"),
        "readonly",
        memmap=True,
    )
    head = HDUreal[0].header.copy()
    FDFreal = HDUreal[0].data

    HDUimag = pf.open(
        filename.replace("_tot", "_im").replace("_real", "_im"), "readonly", memmap=True
    )
    FDFimag = HDUimag[0].data
    complex_cube = FDFreal + 1j * FDFimag

    # Identify Faraday depth axis (assumed to be last one if not explicitly found)
    Ndim, FD_axis = find_axes(head)

    # Move FD axis to first place in numpy order.
    if FD_axis != Ndim:
        complex_cube = np.moveaxis(complex_cube, Ndim - FD_axis, 0)

    # Remove degenerate axes to prevent problems with later steps.
    complex_cube = complex_cube.squeeze()

    return complex_cube, head, FD_axis


# -----------------------------------------------------------------------------#
def main():
    import argparse

    """
    Start the function to perform RM-clean if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run RM-CLEAN on a cube of Faraday dispersion functions (FDFs), applying
    a cube of rotation measure spread functions created by the script
    'do_RMsynth_3D.py'. Saves cubes of deconvolved FDFs & clean-component
    spectra, and a pixel map showing the number of iterations performed.
    Set any of the multiprocessing options to enable parallelization
    (otherwise, pixels will be processed serially).

    Expects that the input is in the form of the Stokes-separated
    (single extension) FITS cubes produced by do_RMsynth_3D.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "fitsFDF",
        metavar="FDF_dirty.fits",
        nargs=1,
        help="FITS cube containing the dirty FDF.\n(Can be any of the FDF output cubes from do_RMsynth_3D.py)",
    )
    parser.add_argument(
        "fitsRMSF",
        metavar="RMSF.fits",
        nargs=1,
        help="FITS cube containing the RMSF and FWHM image.\n(Cans be any of the RMSF output cubes (so not _FWHM.fits!) from do_RMsynth_3D.py)",
    )
    parser.add_argument(
        "-c",
        dest="cutoff",
        type=float,
        default=1,
        help="Initial CLEAN cutoff in flux units [1].",
    )
    parser.add_argument(
        "-w",
        dest="window",
        type=float,
        default=np.nan,
        help="Threshold for (deeper) windowed clean [Not used if not set].",
    )
    parser.add_argument(
        "-n",
        dest="maxIter",
        type=int,
        default=1000,
        help="Maximum number of CLEAN iterations per pixel [1000].",
    )
    parser.add_argument(
        "-g", dest="gain", type=float, default=0.1, help="CLEAN loop gain [0.1]."
    )
    parser.add_argument(
        "-o",
        dest="prefixOut",
        default="",
        help="Prefix to prepend to output files [None].",
    )
    parser.add_argument(
        "-f",
        dest="write_separate_FDF",
        action="store_false",
        help="Store different Stokes as FITS extensions [False, store as separate files].",
    )

    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="Verbose [False]."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    parser.add_argument(
        "--chunk",
        dest="chunk",
        default=None,
        type=int,
        help="Chunk size (uses multiprocessing -- not available in MPI!)",
    )
    group.add_argument(
        "--mpi", dest="mpi", default=False, action="store_true", help="Run with MPI."
    )

    args = parser.parse_args()

    # Check if the user is trying to use parallelization without installing it:
    if parallel_available == False and (
        args.n_cores != 1 or args.chunk != None or args.mpi != False
    ):
        raise Exception(
            "Parallel processing not available. Please install the schwimmbad module to enable parallel processing."
        )

    # If parallelization requested use it, otherwise use the old-fashioned way.
    if parallel_available == True and (
        args.n_cores != 1 or args.chunk != None or args.mpi != False
    ):
        pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
        if args.mpi:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        if args.n_cores > 1:
            chunksize = args.chunk
        else:
            chunksize = None
    else:
        pool = None
        chunksize = None

    verbose = args.verbose
    # Sanity checks
    for f in args.fitsFDF + args.fitsRMSF:
        if not os.path.exists(f):
            print("File does not exist: '%s'." % f)
            sys.exit()
    dataDir, dummy = os.path.split(args.fitsFDF[0])

    # Run RM-CLEAN on the cubes
    cleanFDF, ccArr, iterCountArr, residFDF, headtemp = run_rmclean(
        fitsFDF=args.fitsFDF[0],
        fitsRMSF=args.fitsRMSF[0],
        cutoff=args.cutoff,
        maxIter=args.maxIter,
        gain=args.gain,
        chunksize=chunksize,
        pool=pool,
        nBits=32,
        window=args.window,
        verbose=verbose,
    )
    # Write results to disk
    writefits(
        cleanFDF,
        ccArr,
        iterCountArr,
        residFDF,
        headtemp,
        prefixOut=args.prefixOut,
        outDir=dataDir,
        write_separate_FDF=args.write_separate_FDF,
        verbose=verbose,
    )


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
