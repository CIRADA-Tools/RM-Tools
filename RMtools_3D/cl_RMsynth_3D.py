#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     cl_RMsynth_3D.py                                                  #
#                                                                             #
# PURPOSE:  Run RM-synthesis on a Stokes Q & U cubes.                         #
#                                                                             #
# MODIFIED: 7-March-2019 by J. West                                           #
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
import argparse
import math as m
import numpy as np
import astropy.io.fits as pf
import pdb

from RMutils.util_RM import do_rmsynth_planes
from RMutils.util_RM import get_rmsf_planes
from RMutils.util_misc import interp_images

C = 2.997924538e8 # Speed of light [m/s]

#-----------------------------------------------------------------------------#
def run_rmsynth(dataQ, dataU, freqArr_Hz, headtemplate, dataI=None, rmsArr_Jy=None,
                phiMax_radm2=None, dPhi_radm2=None, nSamples=10.0,
                weightType="uniform", prefixOut="", outDir="",
                fitRMSF=False, nBits=32, write_seperate_FDF=False, verbose=True,
                log = print):

    """Read the Q & U data and run RM-synthesis."""
    # Sanity check on header dimensions

    if not str(dataQ.shape) == str(dataU.shape):
        log("Err: unequal dimensions: Q = "+str(dataQ.shape)+", U = "+str(dataU.shape)+".")
        sys.exit()

    # Check dimensions of Stokes I cube, if present
    if not dataI is None:
        if not str(dataI.shape) == str(dataQ.shape):
            log("Err: unequal dimensions: Q = "+str(dataQ.shape)+", I = "+str(dataI.shape)+".")
            sys.exit()
    
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    lambdaSqArr_m2 = np.power(C/freqArr_Hz, 2.0)

    dFreq_Hz = np.nanmin(np.abs(np.diff(freqArr_Hz)))
    lambdaSqRange_m2 = ( np.nanmax(lambdaSqArr_m2) -
                         np.nanmin(lambdaSqArr_m2) )        
    dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
    dLambdaSqMax_m2 = np.nanmax(np.abs(np.diff(lambdaSqArr_m2)))
    
    # Set the Faraday depth range
    fwhmRMSF_radm2 = 2.0 * m.sqrt(3.0) / lambdaSqRange_m2
    if dPhi_radm2 is None:
        dPhi_radm2 = fwhmRMSF_radm2 / nSamples
    if phiMax_radm2 is None:
        phiMax_radm2 = m.sqrt(3.0) / dLambdaSqMax_m2
        phiMax_radm2 = max(phiMax_radm2, 600.0)    # Force the minimum phiMax

    # Faraday depth sampling. Zero always centred on middle channel
    nChanRM = round(abs((phiMax_radm2 - 0.0) / dPhi_radm2)) * 2.0 + 1.0
    startPhi_radm2 = - (nChanRM-1.0) * dPhi_radm2 / 2.0
    stopPhi_radm2 = + (nChanRM-1.0) * dPhi_radm2 / 2.0
    phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, nChanRM)
    phiArr_radm2 = phiArr_radm2.astype(dtFloat)
    if(verbose): log("PhiArr = %.2f to %.2f by %.2f (%d chans)." % (phiArr_radm2[0],
                                                        phiArr_radm2[-1],
                                                        float(dPhi_radm2),
                                                        nChanRM))
    
        
    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weightType=="variance" and rmsArr_Jy is not None:
        weightArr = 1.0 / np.power(rmsArr_Jy, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)    
    if(verbose): log("Weight type is '%s'." % weightType)

    startTime = time.time()

    # Read the Stokes I model and divide into the Q & U data
    if dataI is not None:    
        with np.errstate(divide='ignore', invalid='ignore'):
            qArr = np.true_divide(dataQ, dataI)
            uArr = np.true_divide(dataU, dataI)
    else:
        qArr = dataQ
        uArr = dataU
        
    # Perform RM-synthesis on the cube
    FDFcube, lam0Sq_m2 = do_rmsynth_planes(dataQ           = qArr,
                                           dataU           = uArr,
                                           lambdaSqArr_m2  = lambdaSqArr_m2,
                                           phiArr_radm2    = phiArr_radm2,
                                           weightArr       = weightArr,
                                           nBits           = 32,
                                           verbose         = True)
    # Calculate the Rotation Measure Spread Function cube
    RMSFcube, phi2Arr_radm2, fwhmRMSFCube, fitStatArr = \
        get_rmsf_planes(lambdaSqArr_m2   = lambdaSqArr_m2,
                        phiArr_radm2     = phiArr_radm2,
                        weightArr        = weightArr,
                        mskArr           = ~np.isfinite(dataQ),
                        lam0Sq_m2        = lam0Sq_m2,
                        double           = True,
                        fitRMSF          = fitRMSF,
                        fitRMSFreal      = False,
                        nBits            = 32,
                        verbose          = True,
                        log              = log)
    endTime = time.time()
    cputime = (endTime - startTime)
    if(verbose): log("> RM-synthesis completed in %.2f seconds." % cputime)
    if(verbose): log("Saving the dirty FDF, RMSF and ancillary FITS files.")

    # Determine the Stokes I value at lam0Sq_m2 from the Stokes I model
    # Note: the Stokes I model MUST be continuous throughout the cube,
    # i.e., no NaNs as the amplitude at freq0_Hz is interpolated from the
    # nearest two planes.
    freq0_Hz = C / m.sqrt(lam0Sq_m2)
    if dataI is not None:
        idx = np.abs(freqArr_Hz - freq0_Hz).argmin()
        if freqArr_Hz[idx]<freq0_Hz:
            Ifreq0Arr = interp_images(dataI[idx, :, :], dataI[idx+1, :, :], f=0.5)
        elif freqArr_Hz[idx]>freq0_Hz:
            Ifreq0Arr = interp_images(dataI[idx-1, :, :], dataI[idx, :, :], f=0.5)
        else:
            Ifreq0Arr = dataI[idx, :, :]

        # Multiply the dirty FDF by Ifreq0 to recover the PI in Jy
        FDFcube *= Ifreq0Arr
    
    
    # Make a copy of the Q header and alter frequency-axis as Faraday depth
    header = headtemplate.copy()
    Ndim=header['NAXIS']
    freq_axis=Ndim #If frequency axis not found, assume it's the last one.
    #Check for frequency axes. Because I don't know what different formatting
    #I might get ('FREQ' vs 'OBSFREQ' vs 'Freq' vs 'Frequency'), convert to 
    #all caps and check for 'FREQ' anywhere in the axis name.
    for i in range(1,Ndim+1):
        try:
            if 'FREQ' in header['CTYPE'+str(i)].upper():
                freq_axis=i
        except:
            pass #The try statement is needed for if the FITS header does not
                 # have CTYPE keywords.

    
    header["NAXIS"+str(freq_axis)] = phiArr_radm2.size
    header["CTYPE"+str(freq_axis)] = "FARADAY DEPTH"
    header["CDELT"+str(freq_axis)] = np.diff(phiArr_radm2)[0]
    header["CRPIX"+str(freq_axis)] = 1.0
    header["CRVAL"+str(freq_axis)] = phiArr_radm2[0]
    header["CUNIT"+str(freq_axis)] = "rad/m^2"
    if "DATAMAX" in header:
        del header["DATAMAX"]
    if "DATAMIN" in header:
        del header["DATAMIN"]

    if outDir=='':  #To prevent code breaking if file is in current directory
        outDir='.'
    
    #Re-add any initially removed degenerate axes (to match with FITS header)
    #NOTE THIS HAS NOT BEEN RIGOROUSLY TESTED!!!
    output_axes=[]
    for i in range(1,Ndim+1):
        output_axes.append(header['NAXIS'+str(i)]) #Get FITS dimensions
    del output_axes[freq_axis-1] #Remove frequency axis (since it's first in the array)
    output_axes.reverse()  #To get into numpy order.
    #Put frequency axis first, and reshape to add degenerate axes:
    FDFcube=np.reshape(FDFcube,[FDFcube.shape[0]]+output_axes) 
    RMSFcube=np.reshape(RMSFcube,[RMSFcube.shape[0]]+output_axes) 
        
    
    #Move Faraday depth axis to appropriate position to match header.
    FDFcube=np.moveaxis(FDFcube,0,Ndim-freq_axis)
    RMSFcube=np.moveaxis(RMSFcube,0,Ndim-freq_axis)

    
    if(write_seperate_FDF):
        hdu0 = pf.PrimaryHDU(FDFcube.real.astype(dtFloat), header)
        hdu1 = pf.PrimaryHDU(FDFcube.imag.astype(dtFloat), header)
        hdu2 = pf.PrimaryHDU(np.abs(FDFcube).astype(dtFloat), header)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_real_dirty.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)

        fitsFileOut = outDir + "/" + prefixOut + "FDF_im_dirty.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)

        fitsFileOut = outDir + "/" + prefixOut + "FDF_tot_dirty.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)

    else:
        # Save the dirty FDF
        hdu0 = pf.PrimaryHDU(FDFcube.real.astype(dtFloat), header)
        hdu1 = pf.ImageHDU(FDFcube.imag.astype(dtFloat), header)
        hdu2 = pf.ImageHDU(np.abs(FDFcube).astype(dtFloat), header)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_dirty.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hduLst = pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        hduLst.close()
    
    
    # Save the RMSF
    header["NAXIS"+str(freq_axis)] = phi2Arr_radm2.size
    header["CRVAL"+str(freq_axis)] = phi2Arr_radm2[0]
    header["DATAMAX"] = np.max(fwhmRMSFCube) + 1
    header["DATAMIN"] = np.max(fwhmRMSFCube) - 1
    if(write_seperate_FDF):
        hdu0 = pf.PrimaryHDU(RMSFcube.real.astype(dtFloat), header)
        hdu1 = pf.PrimaryHDU(RMSFcube.imag.astype(dtFloat), header)
        hdu2 = pf.PrimaryHDU(np.abs(RMSFcube).astype(dtFloat), header)
        hdu3 = pf.PrimaryHDU(fwhmRMSFCube.astype(dtFloat), header)
        fitsFileOut = outDir + "/" + prefixOut + "RMSF_real.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)

        fitsFileOut = outDir + "/" + prefixOut + "RMSF_im.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)

        fitsFileOut = outDir + "/" + prefixOut + "RMSF_tot.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)

        fitsFileOut = outDir + "/" + prefixOut + "RMSF_FWHM.fits"
        if(verbose): log("> %s" % fitsFileOut)
        hdu3.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        
    else:
        fitsFileOut = outDir + "/" + prefixOut + "RMSF.fits"
        hdu0 = pf.PrimaryHDU(RMSFcube.real.astype(dtFloat), header)
        hdu1 = pf.ImageHDU(RMSFcube.imag.astype(dtFloat), header)
        hdu2 = pf.ImageHDU(np.abs(RMSFcube).astype(dtFloat), header)
        hdu3 = pf.ImageHDU(fwhmRMSFCube.astype(dtFloat), header)
        hduLst = pf.HDUList([hdu0, hdu1, hdu2, hdu3])
        if(verbose): log("> %s" % fitsFileOut)
        hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        hduLst.close()
        

 
    #Because there can be problems with different axes having different FITS keywords,
    #don't try to remove the FD axis, but just make it degenerate.
    header["NAXIS"+str(freq_axis)] = 1
    if "DATAMAX" in header:
        del header["DATAMAX"]
    if "DATAMIN" in header:
        del header["DATAMIN"]

        

    # Save a maximum polarised intensity map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_maxPI.fits"
    if(verbose): log("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, np.max(np.abs(FDFcube), freq_axis-1).astype(dtFloat), header,
               overwrite=True, output_verify="fix")
    
    # Save a peak RM map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_peakRM.fits"
    header["BUNIT"] = "rad/m^2"
    peakFDFmap = np.argmax(np.abs(FDFcube), Ndim-freq_axis).astype(dtFloat)
    peakFDFmap = header["CRVAL"+str(freq_axis)] + (peakFDFmap + 1
                                     - header["CRPIX"+str(freq_axis)]) * header["CDELT"+str(freq_axis)]
    if(verbose): log("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, peakFDFmap, header, overwrite=True,
               output_verify="fix")
    
    # Save an RM moment-1 map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_mom1.fits"
    header["BUNIT"] = "rad/m^2"
    mom1FDFmap = (np.nansum(np.moveaxis(np.abs(FDFcube),FDFcube.ndim-freq_axis,FDFcube.ndim-1) * phiArr_radm2, FDFcube.ndim-1)
                  /np.nansum(np.abs(FDFcube), FDFcube.ndim-freq_axis))
#    mom1FDFmap = (np.nansum(np.abs(FDFcube).transpose(1,2,0) * phiArr_radm2, 2)
#                  /np.nansum(np.abs(FDFcube).transpose(1,2,0), 2))
    mom1FDFmap = mom1FDFmap.astype(dtFloat)
    if(verbose): log("> %s" % fitsFileOut)
    pf.writeto(fitsFileOut, mom1FDFmap, header, overwrite=True,
               output_verify="fix")


def readFitsCube_old(file, verbose, log = print):

    if not os.path.exists(file):
        log("Err: File not found")
    
    if(verbose): log("Reading " + file + " ...")    
    data = pf.getdata(file)
    head = pf.getheader(file)
    if(verbose): log("done.")
    
    if head['CTYPE3']=='FREQ': 
        freqAx=3
        data=data[:,:,:]
        # Feeback
        if(verbose): log("The first 3 dimensions of the cubes are [X=%d, Y=%d, Z=%d]." % \
          (head["NAXIS1"], head["NAXIS2"], head["NAXIS3"]))

    elif head["NAXIS"]==4:
        # Feeback
        if(verbose): log("The first 4 dimensions of the cubes are [X=%d, Y=%d, Z=%d, F=%d]." % \
          (head["NAXIS1"], head["NAXIS2"], head["NAXIS3"], head["NAXIS4"]))
        if(head['CTYPE4']=='FREQ'): 
            freqAx=4
            data=data[:,0,:,:]
        else: log("Err: No frequency axis found")

    return head, data
    

def readFitsCube(file, verbose, log = print):
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
    
    if(verbose): log("Reading " + file + " ...")    
    data = pf.getdata(file)
    head = pf.getheader(file)
    if(verbose): log("done.")
    
    N_dim=head['NAXIS'] #Get number of axes
    if verbose:
        print('Dimensions of the input cube are: ',end=' ')
        for i in range(1,N_dim+1):
            print('NAXIS{} = {}'.format(i,head['NAXIS'+str(i)]),end='  ')
        print()
    
    freq_axis=0 #Default for 'frequency axis not identified'
    #Check for frequency axes. Because I don't know what different formatting
    #I might get ('FREQ' vs 'OBSFREQ' vs 'Freq' vs 'Frequency'), convert to 
    #all caps and check for 'FREQ' anywhere in the axis name.
    for i in range(1,N_dim+1):
        try:
            if 'FREQ' in head['CTYPE'+str(i)].upper():
                freq_axis=i
        except:
            pass #The try statement is needed for if the FITS header does not
                 # have CTYPE keywords.
    
    #If the frequency axis isn't the last one, rotate the array until it is.
    #Recall that pyfits reverses the axis ordering, so we want frequency on
    #axis 0 of the numpy array.
    if freq_axis != 0 and freq_axis != N_dim:
        data=np.moveaxis(data,N_dim-freq_axis,0)
    
    if N_dim >= 4:
        data=np.squeeze(data) #Remove degenerate axes

    if verbose:
        print('Dimensions of the input array are: ',data.shape)
        
    if data.ndim > 3:
        raise Exception('Data cube has too many (non-degenerate) axes!')
        
    return head, data
    

def readFreqFile(file, verbose, log = print):
    # Read the frequency vector and wavelength sampling
    freqArr_Hz = np.loadtxt(file, dtype=float)
    return freqArr_Hz
