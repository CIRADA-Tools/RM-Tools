#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RMsynth_3D.py                                                  #
#                                                                             #
# PURPOSE:  Run RM-synthesis on a Stokes Q & U cubes.                         #
#                                                                             #
# MODIFIED: 15-May-2016 by C. Purcell                                         #
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

from RMutils.util_RM import do_rmsynth_planes
from RMutils.util_RM import get_rmsf_planes
from RMutils.util_misc import interp_images

C = 2.997924538e8 # Speed of light [m/s]


#-----------------------------------------------------------------------------#
def main():
    
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

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fitsQ", metavar="StokesQ.fits", nargs=1,
                        help="FITS cube containing Stokes Q data.")
    parser.add_argument("fitsU", metavar="StokesU.fits", nargs=1,
                        help="FITS cube containing Stokes U data.")
    parser.add_argument("freqFile", metavar="freqs_Hz.dat", nargs=1,
                        help="ASCII file containing the frequency vector.")
    parser.add_argument("-i", dest="fitsI", default=None,
                        help="FITS cube containing Stokes I model [None].")
    parser.add_argument("-n", dest="noiseFile", default=None,
                        help="ASCII file containing RMS noise values [None].")
    parser.add_argument("-w", dest="weightType", default="natural",
                        help="weighting [natural] (all 1s) or 'variance'.")
    parser.add_argument("-t", dest="fitRMSF", action="store_true",
                        help="Fit a Gaussian to the RMSF [False]")
    parser.add_argument("-l", dest="phiMax_radm2", type=float, default=None,
                        help="Absolute max Faraday depth sampled [Auto].")
    parser.add_argument("-d", dest="dPhi_radm2", type=float, default=None,
                        help="Width of Faraday depth channel [Auto].")
    parser.add_argument("-o", dest="prefixOut", default="",
                        help="Prefix to prepend to output files [None].")
    parser.add_argument("-s", dest="nSamples", type=float, default=5,
                        help="Number of samples across the FWHM RMSF.")
    args = parser.parse_args()

    # Sanity checks
    for f in args.fitsQ + args.fitsU:
        if not os.path.exists(f):
            print "File does not exist: '%s'." % f
            sys.exit()
    dataDir, dummy = os.path.split(args.fitsQ[0])
    
    # Run RM-synthesis on the cubes
    run_rmsynth(fitsQ        = args.fitsQ[0],
                fitsU        = args.fitsU[0],
                freqFile     = args.freqFile[0],
                fitsI        = args.fitsI,
                noiseFile    = args.noiseFile,
                phiMax_radm2 = args.phiMax_radm2,
                dPhi_radm2   = args.dPhi_radm2,
                nSamples     = args.nSamples,
                weightType   = args.weightType,
                prefixOut    = args.prefixOut,
                outDir       = dataDir,
                fitRMSF      = args.fitRMSF,
                nBits        = 32)


#-----------------------------------------------------------------------------#
def run_rmsynth(fitsQ, fitsU, freqFile, fitsI=None, noiseFile=None,
                phiMax_radm2=None, dPhi_radm2=None, nSamples=10.0,
                weightType="natural", prefixOut="", outDir="",
                fitRMSF=False, nBits=32):
    """Read the Q & U data from the given files and run RM-synthesis."""
    
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Sanity check on header dimensions
    headQ = pf.getheader(fitsQ, 0)
    headU = pf.getheader(fitsU, 0)
    if not headQ["NAXIS"] == headU["NAXIS"]:
        print "Err: unequal header dimensions: Q = %d, U = %d." % \
              (headQ["NAXIS"], headU["NAXIS"])
        sys.exit()
    if headQ["NAXIS"] < 3 or headQ["NAXIS"] > 4:
        print "Err: only  3 dimensions supported: D = %d." % headQ["NAXIS"]
        sys.exit()
    for i in [str(x + 1) for x in range(3)]:
        if not headQ["NAXIS" + i] == headU["NAXIS" + i]:
            print "Err: Axis %d of data are unequal: Q = %d, U = %d." % \
                  (headQ["NAXIS" + i], headU["NAXIS" + i])
            sys.exit()

    # Check dimensions of Stokes I cube, if present
    if not fitsI is None:
        headI = pf.getheader(fitsI, 0)
        if not headI["NAXIS"] == headQ["NAXIS"]:
            print "Err: unequal header dimensions: I = %d, Q = %d." % \
                  (headI["NAXIS"], headQ["NAXIS"])
            sys.exit()
        for i in [str(x + 1) for x in range(3)]:
            if not headI["NAXIS" + i] == headQ["NAXIS" + i]:
                print "Err: Axis %d of data are unequal: I = %d, Q = %d." % \
                      (headI["NAXIS" + i], headQ["NAXIS" + i])
                sys.exit()

    # Feeback
    print "The first 3 dimensions of the cubes are [X=%d, Y=%d, Z=%d]." % \
          (headQ["NAXIS1"], headQ["NAXIS2"], headQ["NAXIS3"])

    # Read the frequency vector and wavelength sampling
    freqArr_Hz = np.loadtxt(freqFile, dtype=dtFloat)
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
    print "PhiArr = %.2f to %.2f by %.2f (%d chans)." % (phiArr_radm2[0],
                                                        phiArr_radm2[-1],
                                                        float(dPhi_radm2),
                                                        nChanRM)
    
    # Read the noise vector, if provided
    rmsArr_Jy = None
    if noiseFile is not None and os.path.exists(noiseFile):
        rmsArr_Jy = np.loadtxt(noiseFile, dtype=dtFloat)
        
    # Calculate the weighting as 1/sigma^2 or all 1s (natural)
    if weightType=="variance" and rmsArr_Jy is not None:
        weightArr = 1.0 / np.power(rmsArr_Jy, 2.0)
    else:
        weightType = "natural"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)    
    print "Weight type is '%s'." % weightType

    # Read the Stokes data
    fQ = pf.open(fitsQ)
    fU = pf.open(fitsU)
    print "Reading Q data array ...",
    if headQ["NAXIS"]==4:
        dataQ = fQ[0].data[0]
    else:
        dataQ = fQ[0].data
    print "done."
    print "Reading U data array ...",
    if headU["NAXIS"]==4:
        dataU = fU[0].data[0]
    else:
        dataU = fU[0].data
    print "done."
    fQ.close()
    fU.close()
    
    startTime = time.time()

    # Read the Stokes I model and divide into the Q & U data
    if fitsI:
        fI = pf.open(fitsI)
        print "Reading I data array ...",
        if headI["NAXIS"]==4:
            dataI = fI[0].data[0]
        else:
            dataI = fI[0].data
            print "done."
    
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
                        mskArr           = np.isnan(dataQ),
                        lam0Sq_m2        = lam0Sq_m2,
                        double           = True,
                        fitRMSF          = fitRMSF,
                        fitRMSFreal      = False,
                        nBits            = 32,
                        verbose          = True)

    endTime = time.time()
    cputime = (endTime - startTime)
    print "> RM-synthesis completed in %.2f seconds." % cputime
    print "Saving the dirty FDF, RMSF and ancillary FITS files."

    # Determine the Stokes I value at lam0Sq_m2 from the Stokes I model
    # Note: the Stokes I model MUST be continuous throughout the cube,
    # i.e., no NaNs as the amplitude at freq0_Hz is interpolated from the
    # nearest two planes.
    freq0_Hz = C / m.sqrt(lam0Sq_m2)
    if fitsI:
        idx = np.abs(freqArr_Hz - freq0_Hz).argmin()
        if freqArr_Hz[idx]<freq0_Hz:
            Ifreq0Arr = interp_images(dataI[idx, :, :], dataI[idx+1, :, :], f=0.5)
        elif freqArr_Hz[idx]>freq0_Hz:
            Ifreq0Arr = interp_images(dataI[idx-1, :, :], dataI[idx, :, :], f=0.5)
        else:
            Ifreq0Arr = dataI[idx, :, :]

        # Multiply the dirty FDF by Ifreq0 to recover the PI in Jy
        FDFcube *= Ifreq0Arr
    
    # Make a copy of the Q header and alter Z-axis as Faraday depth
    header = headQ.copy()
    header["CTYPE3"] = "FARADAY DEPTH"
    header["CDELT3"] = np.diff(phiArr_radm2)[0]
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = phiArr_radm2[0]
    if "DATAMAX" in header:
        del header["DATAMAX"]
    if "DATAMIN" in header:
        del header["DATAMIN"]

    # Save the dirty FDF
    fitsFileOut = outDir + "/" + prefixOut + "FDF_dirty.fits"
    print "> %s" % fitsFileOut
    hdu0 = pf.PrimaryHDU(FDFcube.real.astype(dtFloat), header)
    hdu1 = pf.ImageHDU(FDFcube.imag.astype(dtFloat), header)
    hdu2 = pf.ImageHDU(np.abs(FDFcube).astype(dtFloat), header)
    hduLst = pf.HDUList([hdu0, hdu1, hdu2])
    hduLst.writeto(fitsFileOut, output_verify="fix", clobber=True)
    hduLst.close()
    
    # Save a maximum polarised intensity map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_maxPI.fits"
    print "> %s" % fitsFileOut
    pf.writeto(fitsFileOut, np.max(np.abs(FDFcube), 0).astype(dtFloat), header,
               clobber=True, output_verify="fix")
    
    # Save a peak RM map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_peakRM.fits"
    header["BUNIT"] = "rad/m^2"
    peakFDFmap = np.argmax(np.abs(FDFcube), 0).astype(dtFloat)
    peakFDFmap = header["CRVAL3"] + (peakFDFmap + 1
                                     - header["CRPIX3"]) * header["CDELT3"]
    print "> %s" % fitsFileOut
    pf.writeto(fitsFileOut, peakFDFmap, header, clobber=True,
               output_verify="fix")
    
    # Save an RM moment-1 map
    fitsFileOut = outDir + "/" + prefixOut + "FDF_mom1.fits"
    header["BUNIT"] = "rad/m^2"
    mom1FDFmap = (np.nansum(np.abs(FDFcube).transpose(1,2,0) * phiArr_radm2, 2)
                  /np.nansum(np.abs(FDFcube).transpose(1,2,0), 2))
    mom1FDFmap = mom1FDFmap.astype(dtFloat)
    print "> %s" % fitsFileOut
    pf.writeto(fitsFileOut, mom1FDFmap, header, clobber=True,
               output_verify="fix")

    # Save the RMSF
    header["CRVAL3"] = phi2Arr_radm2[0]
    fitsFileOut = outDir + "/" + prefixOut + "RMSF.fits"
    hdu0 = pf.PrimaryHDU(RMSFcube.real.astype(dtFloat), header)
    hdu1 = pf.ImageHDU(RMSFcube.imag.astype(dtFloat), header)
    hdu2 = pf.ImageHDU(np.abs(RMSFcube).astype(dtFloat), header)
    header["DATAMAX"] = np.max(fwhmRMSFCube) + 1
    header["DATAMIN"] = np.max(fwhmRMSFCube) - 1
    hdu3 = pf.ImageHDU(fwhmRMSFCube.astype(dtFloat), header)
    hduLst = pf.HDUList([hdu0, hdu1, hdu2, hdu3])
    print "> %s" % fitsFileOut
    hduLst.writeto(fitsFileOut, output_verify="fix", clobber=True)
    hduLst.close()


#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
