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
from RMutils.util_misc import fit_StokesI_model, calculate_StokesI_model
from RMutils.util_misc import progress
from RMutils.util_FITS import strip_fits_dims
from RMtools_3D.do_RMsynth_3D import readFitsCube 
from RMtools_3D.make_freq_file import  get_freq_array


from functools import partial
import multiprocessing as mp

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
    parser.add_argument("-f", dest="fit_function", type=str, default="log",
                        help="Stokes I fitting function: 'linear' or ['log'] polynomials.")
    parser.add_argument("-p", dest="polyOrd", type=int, default=2,
                        help="polynomial order to fit to I spectrum: 0-5 supported, 2 is default.\nSet to negative number to enable dynamic order selection.")
    parser.add_argument("-c", dest="cutoff", type=float, default=-5,
                        help="emission cutoff (+ve = abs, -ve = sigma) [-5].")
    parser.add_argument("-t", dest="threshold", type=float, default=3,
                        help="threshold in factors of sigma used to estimate rms noise. Default is 3.")
    parser.add_argument("-n", dest="num_cores", type=int, default=10,
                        help="Number of cores to use for multiprocessing. Default is 10.")
    parser.add_argument("-m", dest="apply_mask", action='store_true',
                        help="Apply masking before spectral fitting. Default is False.")
    parser.add_argument("-o", dest="prefixOut", default="",
                        help="Prefix to prepend to output files [None].")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="turn on verbose messages [False].")
    args = parser.parse_args()
    
    # Sanity checks
    for f in [args.fitsI[0]]:
        if not os.path.exists(f):
            print("File does not exist: '%s'." % f)
            sys.exit()
    dataDir, dummy = os.path.split(args.fitsI[0])
    
    I_filename = args.fitsI[0]
    
    datacube, headI = open_datacube(fitsI=I_filename, verbose=args.verbose)
    freqArr_Hz = get_freq_array (I_filename) 
       
    # Run polynomial fitting on the spectra
    make_model_I(datacube     = datacube, 
                 header       = headI, 
                 freqArr_Hz   = freqArr_Hz,
                 polyOrd      = args.polyOrd,
                 cutoff       = args.cutoff,
                 prefixOut    = args.prefixOut,
                 outDir       = dataDir,
                 nBits        = 32,
                 threshold    = args.threshold,
                 apply_mask   = args.apply_mask,
                 num_cores    = args.num_cores,
                 verbose        = args.verbose,
                 fit_function = args.fit_function)


#-----------------------------------------------------------------------------#


def open_datacube(fitsI, verbose=True):

    # Default data type
    
    # Sanity check on header dimensions
    print("Reading FITS cube header from '%s':" % fitsI)
    header, datacube = readFitsCube(fitsI, verbose)
    
    nDim = datacube.ndim
    if nDim < 3 or nDim > 4:
        print("Err: only 3 or 4 dimensions supported: D = %d." % headI["NAXIS"])
        sys.exit()

    # freq_axis=find_freq_axis(headI) 
    # #If the frequency axis isn't the last one, rotate the array until it is.
    # #Recall that pyfits reverses the axis ordering, so we want frequency on
    # #axis 0 of the numpy array.
    # if freq_axis != 0 and freq_axis != nDim:
    #     datacube=np.moveaxis(datacube,nDim-freq_axis,0)
    
    return datacube, header
    
    
def get_frequencies(datacube, headI, freqFile):

    nBits = np.abs(headI['BITPIX'])    
    dtFloat = "float" + str(nBits)

    nChan = datacube.shape[0] # for now, assumes frequency is the first axis
    
    # Read the frequency vector
    print("Reading frequency vector from '%s'." % freqFile)
    freqArr_Hz = np.loadtxt(freqFile, dtype=dtFloat)
    if nChan!=len(freqArr_Hz):
        print("Err: frequency vector and frequency axis of cube unequal length.")
        sys.exit()
        
    return freqArr_Hz
   
   
def cube_noise(datacube, header, freqArr_Hz, cutoff=-1, threshold=3):


    """
    Estimate channel noise of a cube data. Returns rms values and a mask 2D data.
     
    datacube: input cube data.
    header : header of a cube image
    frequency: frequency values of a cube image in Hz.
    cutoff : cut off to use for creating the final mask. +ve cutoff means absolute, -ve sigma. 
    threshold : Sigma cut off to use to remove emission pixels before calculating rms_noise.  
     
    """
    nBits=np.abs(header['BITPIX'])    
    dtFloat = "float" + str(nBits)    
    nChan = datacube.shape[0]
    

    if nChan!=len(freqArr_Hz):
        print("Err: frequency vector and frequency axis of cube unequal length.")
        sys.exit()
        
    # Measure the RMS spectrum using 2 passes of MAD on each plane
    # Determine which pixels have emission above the cutoff
    print("Measuring the RMS noise and creating an emission mask")
    rmsArr = np.zeros_like(freqArr_Hz)
    medSky = np.zeros_like(freqArr_Hz)
    mskSrc = np.zeros((header["NAXIS2"], header["NAXIS1"]), dtype=dtFloat)
    
    start = time.time()
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
        
        idxSky = np.where(dataPlane < medTmp + rmsTmp * threshold)
        medSky[i] = np.nanmedian(dataPlane[idxSky])
        rmsArr[i] = MAD(dataPlane[idxSky])

        
        # When building final emission mask treat +ve cutoffs as absolute
        # values and negative cutoffs as sigma values
        if cutoff > 0:
            idxSrc = np.where(dataPlane > cutoff)
        else:
            idxSrc = np.where(dataPlane > medSky[i] -1 * rmsArr[i] * cutoff)

        mskSrc[idxSrc] +=1

    end = time.time()
    print(' For loop masking takes %.3fs'%(end-start))
    return rmsArr, mskSrc

    
    
def channel_noise(chan, datacube, header, cutoff, threshold, nBits):
    
    dtFloat = "float" + str(nBits)    
 
    mask_channel = np.zeros((header["NAXIS2"], header["NAXIS1"]), dtype=dtFloat)
    
    dataPlane = datacube[chan]
    if cutoff > 0:
        idxSky = np.where(dataPlane < cutoff)
    else:
        idxSky = np.where(dataPlane)
        
    # Pass 1
    rmsTmp = MAD(dataPlane[idxSky])
    medTmp = np.nanmedian(dataPlane[idxSky])
    idxSky = np.where(dataPlane < medTmp + rmsTmp * threshold)
    
    rms_noise    = MAD(dataPlane[idxSky])
    median_noise =  np.nanmedian(dataPlane[idxSky])
    
    if cutoff > 0:
        idxSrc = np.where(dataPlane > cutoff)
    else:
        idxSrc = np.where(dataPlane > median_noise -1 * rms_noise * cutoff)
    mask_channel[idxSrc] = 1
    
    outs_noise = dict()
    outs_noise['rms_noise']    = rms_noise
    outs_noise['mask_data']    = mask_channel
    
    return outs_noise 
    
    
    
def savefits_mask(data, header, outDir, prefixOut):
    
       
    headMask = strip_fits_dims(header=header, minDim=2)
    headMask["DATAMAX"] = 1
    headMask["DATAMIN"] = 0
    del headMask["BUNIT"]
    
    mskArr = np.where(data > 0, 1.0, np.nan)
    MaskfitsFile = outDir + "/"  + prefixOut + "Mask.fits"
    print("> %s" % MaskfitsFile)
    pf.writeto(MaskfitsFile, mskArr, headMask, output_verify="fix",
               overwrite=True)


def savefits_Coeffs(data, dataerr, header, polyOrd, outDir, prefixOut):

                   
    headcoeff = strip_fits_dims(header=header, minDim=2)
    del headcoeff["BUNIT"]
    
    for i in range(np.abs(polyOrd)+1):
        outname = outDir + "/"  + prefixOut + 'Icoeff'+str(i) + '.fits'
        pf.writeto(outname, data[i], headcoeff, overwrite=True)
        
        outname = outDir + "/"  + prefixOut + 'Icoeff'+str(i) + '_err.fits'
        pf.writeto(outname, dataerr[i], headcoeff, overwrite=True)


    
def savefits_model_I(data, header, outDir, prefixOut):
    
    
    nDim = data.ndim
    nBits = np.abs(header['BITPIX'])
    
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
        
    fitsModelFile = outDir + "/"  + prefixOut + "Imodel.fits"
    headModelCube.tofile(fitsModelFile, overwrite=True)
    with open(fitsModelFile, "rb+") as f:
        f.seek(len(headModelCube.tostring()) + (nVoxels*int(nBits/8)) - 1)
        f.write(b"\0")
    HDULst = pf.open(fitsModelFile, "update", memmap=True)
    HDULst[0].data = data
    HDULst.close()


def fit_spectra_I(xy, datacube, freqArr_Hz, rms_Arr, polyOrd, 
                 fit_function, nDetectPix, verbose=True):
    
    i, x, y = xy 
    
    Ispectrum = datacube[:, x, y]
    
    pixFitDict = fit_StokesI_model(freqArr_Hz, Ispectrum, rms_Arr,
                 polyOrd=polyOrd, fit_function=fit_function)
        
    pixImodel = calculate_StokesI_model(pixFitDict, freqArr_Hz)
        
    outs = dict()
        
    outs['I'] = pixImodel
    outs['coeffs'] = pixFitDict['p']
    outs['coeffs_err'] = pixFitDict['perror']
    outs['chiSq']    = pixFitDict['chiSq']
    outs['chiSqRed'] = pixFitDict['chiSqRed']
    outs['nIter']    = pixFitDict['nIter']
    outs['AIC']      = pixFitDict['AIC']
    
    if verbose:
       progress(40, i/nDetectPix*100.)
    
    return outs         
       

def make_model_I(datacube, header, freqArr_Hz, polyOrd=2, cutoff=-1,  
                 nBits=32, threshold=3, num_cores = 10,verbose=True, fit_function='log', 
                 apply_mask=False, outDir=None, prefixOut=None):  
                
                 
    """
    Estimates Stokes I model data by fitting polynomial function and predicting I
    using the derived coeffiencients. 
    
    datacube:  Stokes I data cube.
    header: header of the data cube. 
    freqArr_Hz: frequency values of the cube in Hz. 
    polyOrd: the order of the polynomial to fit. 0-5 supported, 2 is default.
    fit_function: fit log or linear.
    
    apply_mask: if true a mask will be applied. 
    See channel_noise for definitions of cutoff, threshold.
    
    """
    nChan = datacube.shape[0]
    dtFloat = "float" + str(nBits) 
    
    rms_Arr, mskSrc = cube_noise(datacube, header, freqArr_Hz, cutoff=cutoff,
            threshold=threshold)
    
    mskArr = np.where(mskSrc > 0, 1.0, np.nan)
        
    if not apply_mask:
        mskSrc = np.ones((header['naxis2'], header['naxis1']), dtype=dtFloat)
        mskArr = np.where(mskSrc > 0, 1.0, np.nan)

    srcCoords = np.rot90(np.where(mskSrc > 0))
    
    nPix = mskSrc.shape[-1] * mskSrc.shape[-2]
    nDetectPix = len(srcCoords)
    
    if verbose:
        print("Emission present in %d spectra (%.1f percent)." % \
              (nDetectPix, (nDetectPix*100.0/nPix)))

    modelIcube = np.zeros_like(datacube)
    modelIcube[:] = np.nan
    results = []
    
    coeffs = np.array([mskArr] * 6)
    coeffs_error = np.array([mskArr] * 6)
    datacube = np.squeeze(datacube)
   
    # Inform user job magnitude
    startTime = time.time()
    
    xy = list(zip(np.arange(1, len(srcCoords)), srcCoords[:, 0], srcCoords[:, 1]))
    #print(xy)
    
    if verbose:
        print("Fitting %d/%d spectra." % (nDetectPix, nPix))
        progress(40, 0)
        
    with mp.Pool(num_cores) as pool_:
        results = pool_.map( partial(fit_spectra_I, datacube=datacube, freqArr_Hz=freqArr_Hz, 
                    rms_Arr=rms_Arr, polyOrd=polyOrd, fit_function=fit_function,
                    nDetectPix=nDetectPix),
           xy)
    
    results = list(results)
    
    #print(results)
    headcoeff = strip_fits_dims(header=header, minDim=2)
    del headcoeff["BUNIT"]
                   
    endTime = time.time()
    cputime = (endTime - startTime)
    print("Fitting completed in %.2f seconds." % cputime)
    
    print('Saving results ...')
    for _, an in enumerate(xy):
        i, x, y =  an
        
        modelIcube[:, x, y] =  results[_]['I']
        
        for k,j,l in zip(range(len(coeffs)), results[_]['coeffs'], 
                         results[_]['coeffs_err']):
            coeffs[5-k,x,y] = j     
            coeffs_error[5-k,x,y] = l         

    #if apply_mask:
    print('Saving mask image.')
    savefits_mask(data=mskSrc, header=header, outDir=outDir, prefixOut=prefixOut)
    print("Saving model I coefficients.")
    savefits_Coeffs(data=coeffs, dataerr=coeffs_error, header=header,
         polyOrd=polyOrd, outDir=outDir, prefixOut=prefixOut)
    print("Saving model I cube image. ")
    savefits_model_I(data=modelIcube, header=header, 
         outDir=outDir, prefixOut=prefixOut)
        
    return modelIcube
    


    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
