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
from RMutils.util_misc import fit_StokesI_model,calculate_StokesI_model
from RMutils.util_misc import progress
from RMutils.util_FITS import strip_fits_dims
from RMtools_3D.do_RMsynth_3D import readFitsCube,find_freq_axis

#-----------------------------------------------------------------------------#
def main():
    """
    Start the make_model_I function if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Create a model Stokes I dataset by fitting a polynomial to emitting regions
    above a cutoff in the Stokes I cube. Also outputs a noise spectrum with the
    Stokes I noise per channel.

    NOTE: Each pixel is fit independently, so there are no protections in place
    to ensure smoothness across the image-plane. Noise levels are estimated
    per-channel using 2 passes of the MAD.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fitsI", metavar="StokesI.fits", nargs=1,
                        help="FITS cube containing Stokes I data.")
    parser.add_argument("freqFile", metavar="freqs_Hz.dat", nargs=1,
                        help="ASCII file containing the frequency vector.")
    parser.add_argument("-f", dest="fit_function", type=str, default="log",
                        help="Stokes I fitting function: 'linear' or ['log'] polynomials.")
    parser.add_argument("-p", dest="polyOrd", type=int, default=2,
                        help="polynomial order to fit to I spectrum: 0-5 supported, 2 is default.\nSet to negative number to enable dynamic order selection.")
    parser.add_argument("-c", dest="cutoff", type=float, default=-3,
                        help="emission cutoff (+ve = abs, -ve = sigma) [-3].")
    parser.add_argument("-o", dest="prefixOut", default="",
                        help="Prefix to prepend to output files [None].")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="turn on verbose messages [False].")
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
                 verbose        = args.verbose,
                 fit_function = args.fit_function)


#-----------------------------------------------------------------------------#
def make_model_I(fitsI, freqFile, polyOrd=2, cutoff=-1, prefixOut="",
                 outDir="", verbose=True,fit_function='log'):
    """
    Detect emission in a cube and fit a polynomial or power law model spectrum 
    to the emitting pixels. Create a representative noise spectrum using the 
    residual planes.
    """

    # Default data type
    
    # Sanity check on header dimensions
    print("Reading FITS cube header from '%s':" % fitsI)
    headI,datacube=readFitsCube(fitsI, verbose)
    
    
    nDim = headI["NAXIS"]
    if nDim < 3 or nDim > 4:
        print("Err: only 3 or 4 dimensions supported: D = %d." % headI["NAXIS"])
        sys.exit()


    freq_axis=find_freq_axis(headI) 
    #If the frequency axis isn't the last one, rotate the array until it is.
    #Recall that pyfits reverses the axis ordering, so we want frequency on
    #axis 0 of the numpy array.
    if freq_axis != 0 and freq_axis != nDim:
        datacube=np.moveaxis(datacube,nDim-freq_axis,0)


    nBits=np.abs(headI['BITPIX'])    
    dtFloat = "float" + str(nBits)

    
    nChan = headI["NAXIS"+str(freq_axis)]
    
    # Read the frequency vector
    print("Reading frequency vector from '%s'." % freqFile)
    freqArr_Hz = np.loadtxt(freqFile, dtype=dtFloat)
    if nChan!=len(freqArr_Hz):
        print("Err: frequency vector and frequency axis of cube unequal length.")
        sys.exit()
        
    # Measure the RMS spectrum using 2 passes of MAD on each plane
    # Determine which pixels have emission above the cutoff
    print("Measuring the RMS noise and creating an emission mask")
    rmsArr = np.zeros_like(freqArr_Hz)
    mskSrc = np.zeros((headI["NAXIS2"], headI["NAXIS1"]), dtype=dtFloat)
    mskSky = np.zeros((headI["NAXIS2"], headI["NAXIS1"]), dtype=dtFloat)
    for i in range(nChan):
        dataPlane = datacube[i]
        if cutoff>0:
            idxSky = np.where(dataPlane<cutoff)
        else:
            idxSky = np.where(dataPlane)
        
        # Pass 1
        rmsTmp = MAD(dataPlane[idxSky])
        medTmp = np.nanmedian(dataPlane[idxSky])
        
        # Pass 2: use a fixed 3-sigma cutoff to mask off emission
        idxSky = np.where(dataPlane < medTmp + rmsTmp * 3)
        medSky = np.nanmedian(dataPlane[idxSky])
        rmsArr[i] = MAD(dataPlane[idxSky])
        mskSky[idxSky] +=1
        
        # When building final emission mask treat +ve cutoffs as absolute
        # values and negative cutoffs as sigma values
        if cutoff>0:
            idxSrc = np.where(dataPlane > cutoff)
        else:
            idxSrc = np.where(dataPlane > medSky -1 * rmsArr[i] * cutoff)
        mskSrc[idxSrc] +=1

        # Clean up

    # Save the noise spectrum
    if outDir == '':
        outDir='.'
    print("Saving the RMS noise spectrum in an ASCII file:")
    outFile = outDir + "/"  + prefixOut + "Inoise.dat"
    print("> %s" % outFile)
    np.savetxt(outFile, rmsArr)
        
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
               overwrite=True)
    mskArr = np.where(mskSrc>0, 1.0, np.nan)
    fitsFileOut = outDir + "/"  + prefixOut + "IsrcMask.fits"
    print("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, mskArr, headMsk, output_verify="fix",
               overwrite=True)
        
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
    headModel.tofile(fitsModelFile, overwrite=True)
    with open(fitsModelFile, "rb+") as f:
        f.seek(len(headModel.tostring()) + (nVoxels*int(nBits/8)) - 1)
        f.write(b"\0")

    # Feeback to user
    srcCoords = np.rot90(np.where(mskSrc>0))
    if verbose:
        nPix = mskSrc.shape[-1] * mskSrc.shape[-2]
        nDetectPix = len(srcCoords)
        print("Emission present in %d spectra (%.1f percent)." % \
              (nDetectPix, (nDetectPix*100.0/nPix)))

    # Inform user job magnitude
    startTime = time.time()
    if verbose:
        print("Fitting %d/%d spectra." % (nDetectPix, nPix))
        progress(40, 0)

    datacube=np.squeeze(datacube) #Remove any degenerate axes if needed.'
    modelIcube=np.zeros_like(datacube)
    modelIcube[:]=np.nan

#    Loop through pixels individually
    i=0
    for pixCoords in srcCoords:
        x=pixCoords[0]
        y=pixCoords[1]
        
        Ispectrum=datacube[:,x,y]
        pixFitDict=fit_StokesI_model(freqArr_Hz,Ispectrum,rmsArr,
                          polyOrd=polyOrd,
                          fit_function=fit_function)

        pixImodel=calculate_StokesI_model(pixFitDict,freqArr_Hz)
        modelIcube[:,x,y]=pixImodel
        i+=1
        if verbose:
            progress(40, i/nDetectPix*100.)
    
    
    HDULst = pf.open(fitsModelFile, "update", memmap=True)
    HDULst[0].data = modelIcube
    HDULst.close()
    

        
    endTime = time.time()
    cputime = (endTime - startTime)
    print("Fitting completed in %.2f seconds." % cputime)
    
    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
