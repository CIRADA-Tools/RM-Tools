#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_fitIcube.py                                                    #
#                                                                             #
# PURPOSE:  Make a model Stokes I cube and a noise vector.                    #
#                                                                             #
# MODIFIED: 26-Feb-2017 by C. Purcell                                         #
#                                                                             #
#=============================================================================#
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
#=============================================================================#

import sys
import os
import time
import traceback
import argparse
import math as m
import numpy as np
import astropy.io.fits as pf

from RMutils.util_misc import MAD
from RMutils.util_misc import fit_spec_poly5
from RMutils.util_misc import poly5
from RMutils.util_misc import progress
from RMutils.util_FITS import strip_fits_dims

#-----------------------------------------------------------------------------#
def main():
    """
    Start the make_model_I function if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Create a model Stokes I dataset by fitting a polynomial to emitting regions
    above a cutoff in the Stokes I cube.

    NOTE: This script is very simple and designed to only produce a basic model
    Stokes I cube and noise spectrum. In production environments you should
    use a model cube from a full-featured source finder. Treat the output of
    this script as an example of the data format required by the later 
    RM-synthesis script.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fitsI", metavar="StokesI.fits", nargs=1,
                        help="FITS cube containing Stokes I data.")
    parser.add_argument("freqFile", metavar="freqs_Hz.dat", nargs=1,
                        help="ASCII file containing the frequency vector.")
    parser.add_argument("-p", dest="polyOrd", type=int, default=3,
                        help="polynomial order to fit to I spectrum [3].")
    parser.add_argument("-c", dest="cutoff", type=float, default=-3,
                        help="emission cutoff (+ve = abs, -ve = sigma) [-3].")
    parser.add_argument("-o", dest="prefixOut", default="",
                        help="Prefix to prepend to output files [None].")
    parser.add_argument("-b", dest="buffCols", type=int, default=10,
                        help="# pixel columns to buffer for disk IO [10].")
    parser.add_argument("-D", dest="debug", action="store_true",
                        help="turn on debugging messages [False].")
    args = parser.parse_args()
    
    # Sanity checks
    for f in [args.fitsI[0], args.freqFile[0]]:
        if not os.path.exists(f):
            print("File does not exist: '%s'." % f)
            sys.exit()
    dataDir, dummy = os.path.split(args.fitsI[0])

    # Run RM-synthesis on the spectra
    make_model_I(fitsI = args.fitsI[0],
                 freqFile = args.freqFile[0],
                 polyOrd      = args.polyOrd,
                 cutoff       = args.cutoff,
                 prefixOut    = args.prefixOut,
                 outDir       = dataDir,
                 debug        = args.debug,
                 nBits        = 32,
                 buffCols     = args.buffCols)


#-----------------------------------------------------------------------------#
def make_model_I(fitsI, freqFile, polyOrd=3, cutoff=-1, prefixOut="",
                 outDir="", debug=True, nBits=32, verbose=True, buffCols=10):
    """
    Detect emission in a cube and fit a polynomial model spectrum to the
    emitting pixels. Create a representative noise spectrum using the residual
    planes.
    """

    # Default data type
    dtFloat = "float" + str(nBits)
    
    # Sanity check on header dimensions
    print("Reading FITS cube header from '%s':" % fitsI)
    headI = pf.getheader(fitsI, 0)
    nDim = headI["NAXIS"]
    if nDim < 3 or nDim > 4:
        print("Err: only 3 or 4 dimensions supported: D = %d." % headQ["NAXIS"])
        sys.exit()
    nDim = headI["NAXIS"]
    nChan = headI["NAXIS3"]
    
    # Read the frequency vector
    print("Reading frequency vector from '%s':" % freqFile)
    freqArr_Hz = np.loadtxt(freqFile, dtype=dtFloat)
    freqArr_GHz = freqArr_Hz/1e9
    if nChan!=len(freqArr_Hz):
        print("Err: frequency vector and axis 3 of cube unequal length.")
        sys.exit()
        
    # Measure the RMS spectrum using 2 passes of MAD on each plane
    # Determine which pixels have emission above the cutoff
    print("Measuring the RMS noise and creating an emission mask")
    rmsArr_Jy = np.zeros_like(freqArr_Hz)
    mskSrc = np.zeros((headI["NAXIS2"], headI["NAXIS1"]), dtype=dtFloat)
    mskSky = np.zeros((headI["NAXIS2"], headI["NAXIS1"]), dtype=dtFloat)
    for i in range(nChan):
        HDULst = pf.open(fitsI, "readonly", memmap=True)
        if nDim==3:
            dataPlane = HDULst[0].data[i,:,:]
        elif nDim==4:
            dataPlane = HDULst[0].data[0,i,:,:]
        if cutoff>0:
            idxSky = np.where(dataPlane<cutoff)
        else:
            idxSky = np.where(dataPlane)
        
        # Pass 1
        rmsTmp = MAD(dataPlane[idxSky])
        medTmp = np.median(dataPlane[idxSky])
        
        # Pass 2: use a fixed 3-sigma cutoff to mask off emission
        idxSky = np.where(dataPlane < medTmp + rmsTmp * 3)
        medSky = np.median(dataPlane[idxSky])
        rmsArr_Jy[i] = MAD(dataPlane[idxSky])
        mskSky[idxSky] +=1
        
        # When building final emission mask treat +ve cutoffs as absolute
        # values and negative cutoffs as sigma values
        if cutoff>0:
            idxSrc = np.where(dataPlane > cutoff)
        else:
            idxSrc = np.where(dataPlane > medSky -1 * rmsArr_Jy[i] * cutoff)
        mskSrc[idxSrc] +=1

        # Clean up
        HDULst.close()
        del HDULst

    # Save the noise spectrum
    print("Saving the RMS noise spectrum in an ASCII file:")
    outFile = outDir + "/"  + prefixOut + "Inoise.dat"
    print("> %s" % outFile)
    np.savetxt(outFile, rmsArr_Jy)
        
    # Save FITS files containing sky and source masks
    print("Saving sky and source mask images:")
    mskArr = np.where(mskSky>0, 1.0, np.nan)
    headMsk = strip_fits_dims(header=headI, minDim=2)
    headMsk["DATAMAX"] = 1
    headMsk["DATAMIN"] = 0
    del headMsk["BUNIT"]
    fitsFileOut = outDir + "/"  + prefixOut + "IskyMask.fits"
    print("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, mskArr, headMsk, output_verify="fix",
               clobber=True)
    mskArr = np.where(mskSrc>0, 1.0, np.nan)
    fitsFileOut = outDir + "/"  + prefixOut + "IsrcMask.fits"
    print("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, mskArr, headMsk, output_verify="fix",
               clobber=True)
        
    # Create a blank FITS file on disk using the large file method
    # http://docs.astropy.org/en/stable/io/fits/appendix/faq.html
    #  #how-can-i-create-a-very-large-fits-file-from-scratch
    fitsModelFile = outDir + "/"  + prefixOut + "Imodel.fits"
    print("Creating an empty FITS file on disk")
    print("> %s" % fitsModelFile)
    stub = np.zeros((10, 10, 10), dtype=dtFloat)
    hdu = pf.PrimaryHDU(data=stub)
    headModel = strip_fits_dims(header=headI, minDim=nDim)
    headModel["NAXIS1"] = headI["NAXIS1"]
    headModel["NAXIS2"] = headI["NAXIS2"]
    headModel["NAXIS3"] = headI["NAXIS3"]
    nVoxels = headI["NAXIS1"] * headI["NAXIS2"] * headI["NAXIS3"]
    if nDim==4:
        headModel["NAXIS4"] = headI["NAXIS4"]
        nVoxels *= headI["NAXIS4"]
    while len(headModel) < (36 * 4 - 1):
        headModel.append()
    headModel.tofile(fitsModelFile, clobber=True)
    with open(fitsModelFile, "rb+") as f:
        f.seek(len(headModel.tostring()) + (nVoxels*int(nBits/8)) - 1)
        f.write("\0")

    # Feeback to user
    srcIdx = np.where(mskSrc>0)
    srcCoords = np.rot90(np.where(mskSrc>0))
    if verbose:
        nPix = mskSrc.shape[-1] * mskSrc.shape[-2]
        nDetectPix = len(srcCoords)
        print("Emission present in %d spectra (%.1f percent)." % \
              (nDetectPix, (nDetectPix*100.0/nPix)))

    # Inform user job magnitude
    startTime = time.time()
    print("Fitting %d/%d spectra." % (nDetectPix, nPix))
    j = 0
    nFailPix = 0
    if verbose:
        progress(40, 0)
        
    # Loop through columns of pixels (buffers disk IO)
    for i in range(0, headI["NAXIS1"], buffCols):

        # Select the relevant pixel columns from the mask and cube
        mskSub = mskSrc[:, i:i+buffCols]
        srcCoords = np.rot90(np.where(mskSub>0))
        
        # Select the relevant pixel columns from the mask
        HDULst = pf.open(fitsI, "readonly", memmap=True)
        if nDim==3:
            IArr = HDULst[0].data[:, :, i:i+buffCols]*1e3
        elif nDim==4:
            IArr = HDULst[0].data[0,:, :, i:i+buffCols]*1e3
        HDULst.close()
        IModArr = np.ones_like(IArr)*medSky

        # Fit the spectra in turn
        for yi, xi in srcCoords:
            j += 1
            if verbose:
                progress(40, ((j)*100.0/nDetectPix))
                
            # Fit a <=5th order polynomial model to the Stokes I spectrum
            # Frequency axis must be in GHz to avoid overflow errors
            fitDict = {"fitStatus": 0,
                       "chiSq": 0.0,
                       "dof": len(freqArr_GHz)-polyOrd-1,
                       "chiSqRed": 0.0,
                       "nIter": 0,
                       "p": None}
            try:
                mp = fit_spec_poly5(freqArr_GHz,
                                    IArr[:,yi,xi],
                                    rmsArr_Jy * 1e3,
                                    polyOrd)
                fitDict["p"]         = mp.params
                fitDict["fitStatus"] = mp.status
                fitDict["chiSq"]     = mp.fnorm
                fitDict["chiSqRed"]  = mp.fnorm/fitDict["dof"]
                fitDict["nIter"]     = mp.niter
                IModArr[:,yi,xi]       = poly5(fitDict["p"])(freqArr_GHz)/1e3
            
            except Exception:
                nFailPix += 1
                if debug:
                    print("\nTRACEBACK:")
                    print("-" * 80)
                    print(traceback.format_exc())
                    print("-" * 80)
                    print() 
                    print("> Setting Stokes I spectrum to NaN.\n")
                fitDict["p"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                IModArr[:,yi,xi] = np.ones_like(IArr[:,yi,xi]) * np.nan
        
        # Write the spectrum to the model file
        HDULst = pf.open(fitsModelFile, "update", memmap=True)
        if nDim==3:            
            HDULst[0].data[:, :, i:i+buffCols] = IModArr
        elif nDim==4:
            HDULst[0].data[0, :, :, i:i+buffCols] = IModArr
        HDULst.close()
        
    endTime = time.time()
    cputime = (endTime - startTime)
    print("Fitting completed in %.2f seconds." % cputime)
    if nFailPix>0:
        print("Warn: Fitting failed on %d/%d spectra  (%.1f percent)." % \
              (nFailPix, nDetectPix, (nFailPix*100.0/nDetectPix)))
    
    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
