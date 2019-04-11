#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     mk_test_cube_data.py                                              #
#                                                                             #
# PURPOSE:  Create a small FITS cube dataset for the purposes of testing      #
#           the RM-code. The script outputs a set of IQU image cubes          #
#           containing sources with properties read from an input catalogue   #
#           file. Properties of the output data (e.g., frequency sampling,    #
#           beam size) are set in variables at the top of this script. A      #
#           template describing the shape of the rms noise curve may also be  #
#           provided in an external ASCII file [freq_Hz amp].                 #
#                                                                             #
#           This version for use with the stand-alone code, not the pipeline. #
#                                                                             #
# MODIFIED: 25-Feb-2017 by C. Purcell                                         #
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

# BEGIN USER EDITS -----------------------------------------------------------#

# Frequency parameters
startFreq_Hz = 0.7e9
endFreq_Hz = 1.8e9
nChans = 111

# Spatial sampling parameters
beamMinFWHM_deg = 1.5/3600.0
beamMajFWHM_deg = 2.5/3600.0
beamPA_deg = 20.0
pixScale_deg = 0.3/3600.0
nPixX = 120
nPixY = 100

# Coordinate system ["EQU" or "GAL"] and field centre coordinate.
coordSys = "GAL"
xCent_deg = 90.0
yCent_deg = 0.0

# Noise level
rmsNoise_mJy = 0.04

# NOTE: Variations in the noise vs frequency can be specified in an external
# file. Properties of injected sources are given by an external CSV catalogue
# file. Two types of model may be specified, assuming a common flux & spectral
# index. Execute "./0_mk_test_image_data.py -h" to print detailed information.

# Frequency at which the flux is specified in the catalogue
freq0_Hz = startFreq_Hz

# END USER EDITS -------------------------------------------------------------#

import os
import sys
import argparse
import shutil
import math as m
import numpy as np
import astropy.wcs.wcs as pw

from RMutils.util_misc import twodgaussian
from RMutils.util_misc import create_IQU_spectra_burn
from RMutils.util_misc import create_IQU_spectra_diff 
from RMutils.util_misc import csv_read_to_list
from RMutils.util_misc import split_repeat_lst
from RMutils.util_misc import calc_stats
from RMutils.util_misc import progress
from RMutils.util_misc import extrap
from RMutils.util_FITS import strip_fits_dims
from RMutils.util_FITS import create_simple_fits_hdu

C = 2.99792458e8

#-----------------------------------------------------------------------------#
def main():
    """
    Start the create_IQU_fits_data function if called from the command line.
    """
    
    # Help string to be shown using the -h option
    descStr = """
    Create a new dataset directory and populate it with data cubes in FITS 
    format. Three FITS files are produced, one for each of the Stokes I, Q 
    and U parameters. A vector of frequency channels is also saved as an
    ASCII file 'freqs_Hz.dat'.
    
    The data is populated with polarised sources whose properties are given
    in an external CSV-format catalogue file. Two types of model may be 
    specified, assuming a total flux & spectral index:

        # MODEL TYPE 1: One or more components affected by Burn depolarisation.
        #
        # Column  |  Description
        #---------------------------------------------------
        # [0]     |  Model type [1]
        # [1]     |  X coordinate (deg)
        # [2]     |  Y coordinate (deg)
        # [3]     |  Major axis (arcsec)
        # [4]     |  Minor axis (arcsec)
        # [5]     |  Position angle (deg)
        # [6]     |  Total flux (mJy)
        # [7]     |  Spectral index
        # Component 1:
        # [8]     |  Intrinsic polarisation angle (deg)
        # [9]     |  Fractional polarisation
        # [10]    |  Faraday depth (radians m^-2)
        # [11]    |  Farday dispersion (radians m^-2)
        # Component 2:
        # [12]    |  Intrinsic polarisation angle (deg)
        # [13]    |  Fractional polarisation
        # [14]    |  Faraday depth (radians m^-2)
        # [15]    |  Farday dispersion (radians m^-2)
        # Component 3:
        # [16]    |  ...
        #---------------------------------------------------

        # MODEL TYPE 2: One or more stacked layers with differential Faraday
        # rotation (Sokoloff et al. 1998, Equation 9).
        #
        # Column  |  Description
        #---------------------------------------------------
        # [0]     |  Model type (2)
        # [1]     |  X coordinate (deg)
        # [2]     |  Y coordinate (deg)
        # [3]     |  Major axis (arcsec)
        # [4]     |  Minor axis (arcsec)
        # [5]     |  Position angle (deg)
        # [6]     |  Total flux (mJy)
        # [7]     |  Spectral index
        # Component 1:
        # [8]     |  Intrinsic polarisation angle (deg)
        # [9]     |  Fractional polarisation
        # [10]    |  Faraday depth (radians m^-2)
        # Component 2:
        # [11]    |  Intrinsic polarisation angle (deg)
        # [12]    |  Fractional polarisation
        # [13]    |  Faraday depth (radians m^-2)
        # Component 3:
        # [14]    |  ...
        #---------------------------------------------------

    Properties of the data (frequency sampling, noise level) are given at the
    top of this script, including an optional template for the shape of the
    noise curve and an optional list of frequency ranges to flag out.

    The beam may be elliptical and is set to a constant angular size as a
    function of frequency (i.e., assumes all data has been convolved to the
    same resolution). Please edit the variables at the top of the script to
    change the properties of the output data.

    Examples:

    ./mk_test_cube_data.py catalogue.csv data/
    
    ./mk_test_cube_data.py catalogue.csv -f 1.10e9,1.20e9,1.60e9,1.65e9
    
    ./mk_test_cube_data.py catalogue.csv -n NOISE.TXT
    
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("inCatFile", metavar="catalogue.csv", nargs=1,
                        help="Input catalogue file in CSV format")
    parser.add_argument("dataPath", metavar="PATH/TO/DATA",
                        default="data", nargs="?",
                        help="Path to new data directory [data/]")
    parser.add_argument('-n', dest='noiseTmpFile', metavar="NOISE.TXT",
                        help="File providing a template noise curve [freq amp]")
    parser.add_argument('-f', dest='flagFreqStr', metavar='f1,f2,f1,f2,...',
                        default="", help="Frequency ranges to flag out")
    args = parser.parse_args()
    inCatFile = args.inCatFile[0]
    dataPath = args.dataPath
    noiseTmpFile = args.noiseTmpFile
    flagRanges_Hz = []
    if len(args.flagFreqStr)>0:
        try:
            flagFreqLst = args.flagFreqStr.split(",")
            flagFreqLst = [float(x) for x in flagFreqLst]
            flagRanges_Hz = list(zip(*[iter(flagFreqLst)]*2))
        except Exception:
            "Warn: Failed to parse frequency flagging string!"

    # Read the RMS noise template
    try:
        noiseTmpArr = np.loadtxt(noiseTmpFile, unpack=True)
    except Exception:
        noiseTmpArr = None
    
    # Call the function to create FITS data
    nSrc = create_IQU_cube_data(dataPath, inCatFile, startFreq_Hz, endFreq_Hz,
                                nChans, rmsNoise_mJy, beamMinFWHM_deg,
                                beamMajFWHM_deg, beamPA_deg, pixScale_deg,
                                xCent_deg, yCent_deg, nPixX, nPixY,
                                coordSys, noiseTmpArr, flagRanges_Hz)


#-----------------------------------------------------------------------------#
def create_IQU_cube_data(dataPath, inCatFile, startFreq_Hz, endFreq_Hz, nChans,
                         rmsNoise_mJy, beamMinFWHM_deg, beamMajFWHM_deg,
                         beamPA_deg, pixScale_deg, xCent_deg, yCent_deg,
                         nPixX, nPixY, coordSys="EQU", noiseTmpArr=None,
                         flagRanges_Hz=[]):
    """
    Create a set of Stokes I Q & U data-cubes containing polarised sources.
    """

    # Sample frequency space
    freqArr_Hz = np.linspace(startFreq_Hz, endFreq_Hz, nChans)
    freqNoFlgArr_Hz = freqArr_Hz.copy()
    dFreq_Hz = (endFreq_Hz - startFreq_Hz)/ (nChans-1)
    print("\nSampling frequencies %.2f - %.2f MHz by %.2f MHz." % \
          (freqArr_Hz[0]/1e6, freqArr_Hz[-1]/1e6, dFreq_Hz/1e6))
    if len(flagRanges_Hz)>0:
        print("Flagging frequency ranges:")
        print("> ", flagRanges_Hz)
    for i in range(len(freqArr_Hz)):
        for fRng in flagRanges_Hz:            
            if freqArr_Hz[i]>=fRng[0] and freqArr_Hz[i]<=fRng[1]:
                freqArr_Hz[i]=np.nan

    # Create normalised noise array from a template or assume all ones.
    if noiseTmpArr is None:
        print("Assuming flat noise versus frequency curve.")
        noiseArr = np.ones(freqArr_Hz.shape, dtype="f4")
    else:
        print("Scaling noise curve by external template.")
        xp = noiseTmpArr[0]
        yp = noiseTmpArr[1]
        mDict = calc_stats(yp)
        yp /= mDict["median"]
        noiseArr = extrap(freqArr_Hz, xp, yp)
        
    # Check the catalogue file exists
    if not os.path.exists(inCatFile):
        print("Err: File does not exist '%s'." % inCatFile)
        sys.exit()
    catInLst = csv_read_to_list(inCatFile, doFloat=True)
    print("Found %d entries in the catalogue." % len(catInLst))

    # Create the output directory path
    dirs = dataPath.rstrip("/").split("/")
    for i in range(1, len(dirs)):
        dirStr = "/".join(dirs[:i])
        if not os.path.exists(dirStr):
            os.mkdir(dirStr)
    if os.path.exists(dataPath):
        print("\n ", end=' ')
        print("*** WARNING ***" *5)
        print("  About to delete existing data directory!", end=' ')
        print("Previous results will be deleted.\n")
        print("Press <RETURN> to continue ...", end=' ')
        input()
        shutil.rmtree(dataPath, True)
    os.mkdir(dataPath)

    # Create simple HDUs in memmory
    print("Creating test data in memory.")
    hduI = create_simple_fits_hdu(shape=(1, nChans, nPixY, nPixX),
                                  freq_Hz=startFreq_Hz,
                                  dFreq_Hz=dFreq_Hz,
                                  xCent_deg=xCent_deg,
                                  yCent_deg=yCent_deg,
                                  beamMinFWHM_deg=beamMinFWHM_deg,
                                  beamMajFWHM_deg=beamMajFWHM_deg,
                                  beamPA_deg=beamPA_deg,
                                  pixScale_deg=pixScale_deg,
                                  stokes="I",
                                  system=coordSys)
    head2D = strip_fits_dims(header=hduI.header, minDim=2, forceCheckDims=4)
    wcs2D = pw.WCS(head2D)
    shape2D = (nPixY, nPixX)
    hduQ = hduI.copy()
    hduQ.header["CRVAL4"] = 2.0
    hduU = hduI.copy()
    hduU.header["CRVAL4"] = 3.0
    
    # Calculate some beam parameters
    gfactor = 2.0*m.sqrt(2.0*m.log(2.0))
    beamMinSigma_deg = beamMinFWHM_deg/gfactor
    beamMajSigma_deg = beamMajFWHM_deg/gfactor
    beamMinSigma_pix = beamMinSigma_deg/pixScale_deg
    beamMajSigma_pix = beamMajSigma_deg/pixScale_deg
    beamPA_rad = m.radians(beamPA_deg)
    
    # Loop through the sources, calculate the spectra and pixel position
    spectraILst = []
    spectraQLst = []
    spectraULst = []
    coordLst_deg = []
    coordLst_pix = []
    successCount = 0
    for i in range(len(catInLst)):
        e = catInLst[i]
        modelType = int(e[0])

        # Type 1 = multiple Burn depolarisation affected components
        if modelType==1:
            
            # Parse the parameters of multiple components
            preLst, parmArr = split_repeat_lst(e[1:],7,4)
            
            # Create the model spectra from multiple thin components
            # modified by external depolarisation
            IArr_Jy, QArr_Jy, UArr_Jy = \
                create_IQU_spectra_burn(freqArr_Hz = freqArr_Hz,
                                        fluxI = preLst[5]/1e3, # mJy->Jy
                                        SI = preLst[6],
                                        fracPolArr = parmArr[0],
                                        psi0Arr_deg = parmArr[1],
                                        RMArr_radm2 = parmArr[2],
                                        sigmaRMArr_radm2 = parmArr[3],
                                        freq0_Hz = freq0_Hz)
                
        # Type 2 = multiple internal depolarisation affected components
        elif modelType==2:
            
            # Parse the parameters of multiple components
            preLst, parmArr = split_repeat_lst(e[1:],7,3)
            
            # Create the model spectra from multiple components
            # modified by internal Faraday depolarisation
            IArr_Jy, QArr_Jy, UArr_Jy = \
                create_IQU_spectra_diff(freqArr_Hz = freqArr_Hz,
                                        fluxI = preLst[5]/1e3, # mJy->Jy
                                        SI = preLst[6],
                                        fracPolArr = parmArr[0],
                                        psi0Arr_deg = parmArr[1],
                                        RMArr_radm2 = parmArr[2],
                                        freq0_Hz = freq0_Hz)
        else:
            continue
        
        spectraILst.append(IArr_Jy)
        spectraQLst.append(QArr_Jy)
        spectraULst.append(UArr_Jy)
        coordLst_deg.append([preLst[0], preLst[1]])        
        [ (x_pix, y_pix) ] = wcs2D.wcs_world2pix([ (preLst[0], preLst[1]) ], 0)
        coordLst_pix.append([x_pix, y_pix])        
        successCount +=1

    # Loop through the frequency channels & insert the IQU planes
    print("Looping through %d frequency channels:" % nChans)
    progress(40, 0.0)    
    for iChan in range(len(freqArr_Hz)):
        progress(40, (100.0 * (iChan + 1) / nChans))
        for iSrc in range(len(spectraILst)):
            params = [spectraILst[iSrc][iChan],  # amplitude
                      coordLst_pix[iSrc][0],     # X centre (pix)
                      coordLst_pix[iSrc][1],     # Y centre
                      beamMinSigma_pix,          # width (sigma)
                      beamMajSigma_pix,          # height (sigma)
                      beamPA_rad]                # PA (rad) W of N (clockwise)
            planeI = twodgaussian(params, shape2D).reshape((nPixY, nPixX))
            params[0] = spectraQLst[iSrc][iChan]
            planeQ = twodgaussian(params, shape2D).reshape((nPixY, nPixX))
            params[0] = spectraULst[iSrc][iChan]
            planeU = twodgaussian(params, shape2D).reshape((nPixY, nPixX))
            hduI.data[0, iChan, :, :] += planeI
            hduQ.data[0, iChan, :, :] += planeQ
            hduU.data[0, iChan, :, :] += planeU
        
        # Add the noise
        rmsNoise_Jy = rmsNoise_mJy/1e3
        hduI.data[0, iChan, :, :] += (np.random.normal(scale=rmsNoise_Jy,
                                      size=(nPixY, nPixX))* noiseArr[iChan])
        hduQ.data[0, iChan, :, :] += (np.random.normal(scale=rmsNoise_Jy,
                                      size=(nPixY, nPixX))* noiseArr[iChan])
        hduU.data[0, iChan, :, :] += (np.random.normal(scale=rmsNoise_Jy,
                                      size=(nPixY, nPixX))* noiseArr[iChan])

    # DEBUG
    if False:
        # Mask a sub=cube
        hduI.data[0, 20:50, 20:40, 20:40] = np.nan
        hduQ.data[0, 20:50, 20:40, 20:40] = np.nan
        hduU.data[0, 20:50, 20:40, 20:40] = np.nan
        # Mask a sub=cube
        hduI.data[0, 60:90, 50:70, 50:70] = np.nan
        hduQ.data[0, 60:90, 50:70, 50:70] = np.nan
        hduU.data[0, 60:90, 50:70, 50:70] = np.nan
    if False:
        # Mask full planes
        hduI.data[0, 50:60, :, :] = np.nan
        hduQ.data[0, 50:60, :, :] = np.nan
        hduU.data[0, 50:60, :, :] = np.nan
    if False:
        # Mask full spectra
        hduI.data[0, :, 20:60, 20:70] = np.nan
        hduQ.data[0, :, 20:60, 20:70] = np.nan
        hduU.data[0, :, 20:60, 20:70] = np.nan

    # Write to the FITS files
    print("Saving the FITS files to disk in directory '%s'" % dataPath)
    sys.stdout.flush()
    fitsFileOut =  dataPath + "/StokesI.fits"
    print("> %s" % fitsFileOut)
    hduI.writeto(fitsFileOut, clobber=True)
    fitsFileOut =  dataPath + "/StokesQ.fits"
    print("> %s" % fitsFileOut)
    hduQ.writeto(fitsFileOut, clobber=True)
    fitsFileOut =  dataPath + "/StokesU.fits"
    print("> %s" % fitsFileOut)
    hduU.writeto(fitsFileOut, clobber=True)

    # Save a vector of frequency values
    freqFileOut =  dataPath + "/freqs_Hz.dat"
    print("> %s" % freqFileOut)
    np.savetxt(freqFileOut,freqNoFlgArr_Hz)
    
    return successCount


#-----------------------------------------------------------------------------#
if __name__=="__main__":
    main()
