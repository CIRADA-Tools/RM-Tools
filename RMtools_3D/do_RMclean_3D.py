#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RM-clean.py                                                    #
#                                                                             #
# PURPOSE:  Run RM-clean on a  cube of dirty Faraday dispersion functions.    #
#                                                                             #
# MODIFIED: 15-May-2016 by C. Purcell                                         #
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
import os
import time
import argparse
import math as m
import numpy as np
import astropy.io.fits as pf

import schwimmbad

from RMutils.util_RM import do_rmclean_hogbom
from RMutils.util_RM import fits_make_lin_axis

C = 2.997924538e8  # Speed of light [m/s]

#-----------------------------------------------------------------------------#


def main():
    """
    Start the function to perform RM-clean if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run RM-CLEAN on a cube of Faraday dispersion functions (FDFs), applying
    a cube of rotation measure spread functions created by the script
    'do_RMsynth_3D.py'. Saves a cube of deconvolved FDFs & clean-component
    spectra, and a pixel map showing the number of iterations performed.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fitsFDF", metavar="FDF_dirty.fits", nargs=1,
                        help="FITS cube containing the dirty FDF.\n(Must be the single-file output from do_RMsynth_3D.py)")
    parser.add_argument("fitsRMSF", metavar="RMSF.fits", nargs=1,
                        help="FITS cube containing the RMSF and FWHM image.\n(Must be the single-file output from do_RMsynth_3D.py)")
    parser.add_argument("-c", dest="cutoff_mJy", type=float, nargs=1,
                        default=1.0, help="CLEAN cutoff in mJy")
    parser.add_argument("-n", dest="maxIter", type=int, default=1000,
                        help="Maximum number of CLEAN iterations per pixel [1000].")
    parser.add_argument("-g", dest="gain", type=float, default=0.1,
                        help="CLEAN loop gain [0.1].")
    parser.add_argument("-o", dest="prefixOut", default="",
                        help="Prefix to prepend to output files [None].")
    parser.add_argument("-f", dest="write_separate_FDF", action="store_true",
                        help="Separate complex (multi-extension) FITS files into individual files [False].")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="Verbose [False].")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    parser.add_argument("--chunk", dest="chunk", default=None,
                       type=int, help="Chunk size (uses multiprocessing -- not available in MPI!)")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    if args.mpi:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    if args.n_cores > 1:
        chunksize = args.chunk
    else:
        chunksize = None

    verbose = args.verbose
    # Sanity checks
    for f in args.fitsFDF + args.fitsRMSF:
        if not os.path.exists(f):
            print("File does not exist: '%s'." % f)
            sys.exit()
    dataDir, dummy = os.path.split(args.fitsFDF[0])

    # Run RM-CLEAN on the cubes
    run_rmclean(fitsFDF=args.fitsFDF[0],
                fitsRMSF=args.fitsRMSF[0],
                cutoff_mJy=args.cutoff_mJy,
                maxIter=args.maxIter,
                gain=args.gain,
                prefixOut=args.prefixOut,
                outDir=dataDir,
                nBits=32,
                write_separate_FDF=args.write_separate_FDF,
                pool=pool,
                chunksize=chunksize)

#-----------------------------------------------------------------------------#


def run_rmclean(fitsFDF, fitsRMSF, cutoff_mJy, maxIter=1000, gain=0.1,
                prefixOut="", outDir="", nBits=32, write_separate_FDF=False, verbose=True, log=print, pool=None, chunksize=None, verbose=verbose):
    """Run RM-CLEAN on a FDF cube given a RMSF cube."""

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Read the FDF
    dirtyFDF, head, FD_axis = read_FDF_cube(fitsFDF)

    phiArr_radm2 = fits_make_lin_axis(head, axis=FD_axis-1, dtype=dtFloat)

    # Read the RMSF

    RMSFArr, headRMSF, FD_axis = read_FDF_cube(fitsRMSF)
    HDULst = pf.open(fitsRMSF, "readonly", memmap=True)
    fwhmRMSFArr = HDULst[3].data
    HDULst.close()
    phi2Arr_radm2 = fits_make_lin_axis(headRMSF, axis=FD_axis-1, dtype=dtFloat)

    startTime = time.time()

    # Do the clean
    cleanFDF, ccArr, iterCountArr, residFDF = \
        do_rmclean_hogbom(dirtyFDF=dirtyFDF * 1e3,
                          phiArr_radm2=phiArr_radm2,
                          RMSFArr=RMSFArr,
                          phi2Arr_radm2=phi2Arr_radm2,
                          fwhmRMSFArr=fwhmRMSFArr,
                          cutoff=cutoff_mJy,
                          maxIter=maxIter,
                          gain=gain,
                          verbose=True,
                          doPlots=False,
                          pool=pool,
                          chunksize=chunksize,
                          verbose = verbose)
    cleanFDF /= 1e3
    ccArr /= 1e3

    endTime=time.time()
    cputime=(endTime - startTime)
    if (verbose): log("> RM-clean completed in %.2f seconds." % cputime)
    if (verbose): log("Saving the clean FDF and ancillary FITS files")


    if outDir == '':  # To prevent code breaking if file is in current directory
        outDir='.'

    # Move FD axis back to original position:
    Ndim=cleanFDF.ndim
    cleanFDF=np.moveaxis(cleanFDF, 0, Ndim-FD_axis)
    ccArr=np.moveaxis(ccArr, 0, Ndim-FD_axis)
    residFDF=np.moveaxis(residFDF, 0, Ndim-FD_axis)


    # Save the clean FDF
    if not write_separate_FDF:
        fitsFileOut=outDir + "/" + prefixOut + "FDF_clean.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hdu0=pf.PrimaryHDU(cleanFDF.real.astype(dtFloat), head)
        hdu1=pf.ImageHDU(cleanFDF.imag.astype(dtFloat), head)
        hdu2=pf.ImageHDU(np.abs(cleanFDF).astype(dtFloat), head)
        hduLst=pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        hduLst.close()
    else:
        hdu0=pf.PrimaryHDU(cleanFDF.real.astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_clean_real.fits"
        hdu0.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu1=pf.PrimaryHDU(cleanFDF.imag.astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_clean_im.fits"
        hdu1.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu2=pf.PrimaryHDU(np.abs(cleanFDF).astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_clean_tot.fits"
        hdu2.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)

    if not write_separate_FDF:
    # Save the complex clean components as another file.
        fitsFileOut=outDir + "/" + prefixOut + "FDF_CC.fits"
        if (verbose): log("> %s" % fitsFileOut)
        hdu0=pf.PrimaryHDU(ccArr.real.astype(dtFloat), head)
        hdu1=pf.ImageHDU(ccArr.imag.astype(dtFloat), head)
        hdu2=pf.ImageHDU(np.abs(ccArr).astype(dtFloat), head)
        hduLst=pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        hduLst.close()
    else:
        hdu0=pf.PrimaryHDU(ccArr.real.astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_CC_real.fits"
        hdu0.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu1=pf.PrimaryHDU(ccArr.imag.astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_CC_im.fits"
        hdu1.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu2=pf.PrimaryHDU(np.abs(ccArr).astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_CC_tot.fits"
        hdu2.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)

    if not write_separate_FDF:
    # Save the complex residuals as another file.
        fitsFileOut=outDir + "/" + prefixOut + "FDF_resid.fits"
        if (verbose): log("> %s" % fitsFileOut)
        hdu0=pf.PrimaryHDU(residFDF.real.astype(dtFloat), head)
        hdu1=pf.ImageHDU(residFDF.imag.astype(dtFloat), head)
        hdu2=pf.ImageHDU(np.abs(residFDF).astype(dtFloat), head)
        hduLst=pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        hduLst.close()
    else:
        hdu0=pf.PrimaryHDU(residFDF.real.astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_resid_real.fits"
        hdu0.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu1=pf.PrimaryHDU(residFDF.imag.astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_resid_im.fits"
        hdu1.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu2=pf.PrimaryHDU(np.abs(residFDF).astype(dtFloat), head)
        fitsFileOut=outDir + "/" + prefixOut + "FDF_resid_tot.fits"
        hdu2.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
        if (verbose): log("> %s" % fitsFileOut)


    # Save the iteration count mask
    fitsFileOut=outDir + "/" + prefixOut + "CLEAN_nIter.fits"
    if (verbose): log("> %s" % fitsFileOut)
    head["BUNIT"]="Iterations"
    hdu0=pf.PrimaryHDU(iterCountArr.astype(dtFloat), head)
    hduLst=pf.HDUList([hdu0])
    hduLst.writeto(fitsFileOut, output_verify = "fix", overwrite = True)
    hduLst.close()


def read_FDF_cube(filename):
    """Read in a FDF/RMSF cube. Figures out which axis is Faraday depth and
    puts it first (in numpy order) to accommodate the rest of the code.
    Returns: (complex_cube, header,FD_axis)
    """
    HDULst=pf.open(filename, "readonly", memmap = True)
    head=HDULst[0].header.copy()
    FDFreal=HDULst[0].data
    FDFimag=HDULst[1].data
    complex_cube=FDFreal + 1j * FDFimag

    # Identify Faraday depth axis (assumed to be last one if not explicitly found)
    Ndim=head['NAXIS']
    FD_axis=Ndim
    # Check for FD axes:
    for i in range(1, Ndim+1):
        try:
            if 'FARADAY' in head['CTYPE'+str(i)].upper():
                FD_axis=i
        except:
            pass  # The try statement is needed for if the FITS header does not
                 # have CTYPE keywords.

    # Move FD axis to first place in numpy order.
    if FD_axis != Ndim:
        complex_cube=np.moveaxis(complex_cube, Ndim-FD_axis, 0)

    # Remove degenerate axes to prevent problems with later steps.
    complex_cube=complex_cube.squeeze()

    return complex_cube, head, FD_axis

#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
