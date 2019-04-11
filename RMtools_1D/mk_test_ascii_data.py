#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     mk_test_ascii_data.py                                             #
#                                                                             #
# PURPOSE:  Create a small ASCII dataset for the purposes of testing the      #
#           RM-code. The script outputs a set ASCII files containing          #
#           frequency and Stokes vectors based on source parameters read from #
#           an input catalogue file. Properties of the output data are set    #
#           in variables at the top of this script. A template describing the #
#           shape of the rms noise curve may also be provided in an external  #
#           ASCII file [freq_Hz, amp].                                        #
#                                                                             #
#           This version for use with the stand-alone code, not the pipeline. #
#                                                                             #
# MODIFIED: 14-mar-2018 by C. Purcell                                         #
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

# NOTE: Variations in the noise vs frequency can be specified in an external
# file. Properties of injected sources are given by an external CSV catalogue
# file. Two types of model may be specified, assuming a common flux & spectral
# index. Execute "./0_mk_test_ascii_data.py -h" to print detailed information.

import os
import sys
import argparse
import shutil
import numpy as np

from RMutils.util_misc import create_IQU_spectra_burn
from RMutils.util_misc import create_IQU_spectra_diff
from RMutils.util_misc import csv_read_to_list
from RMutils.util_misc import split_repeat_lst 
from RMutils.util_misc import calc_stats
from RMutils.util_misc import extrap

C = 2.99792458e8


#-----------------------------------------------------------------------------#
def main():
    """
    Run the create_IQU_ascii_data function if called from the command line.
    """
    
    # Help string to be shown using the -h option
    descStr = """
    Create a new dataset directory and populate it with ASCII files containing
    Stokes I, Q and U spectra. Each output file contains four columns
    corresponding to [freq_Hz, StokesI_Jy, StokesQ_Jy, StokesU_Jy] vectors for
    one source.

    The spectra are populated with polarised sources whose properties are given
    in an external CSV-format catalogue file. Two types of model may be 
    specified in the file, assuming a total flux & spectral index:

        # MODEL TYPE 1: One or more components affected by Burn depolarisation.
        # (see e.g., Equation 5 Tribble 1991, Burn 1966).
        # Column  |  Description
        #---------------------------------------------------
        # [0]     |  Model type [1]
        # [1]     |  X coordinate (deg)    ... dummy value
        # [2]     |  Y coordinate (deg)    ... dummy value
        # [3]     |  Major axis (arcsec)   ... dummy value
        # [4]     |  Minor axis (arcsec)   ... dummy value
        # [5]     |  Position angle (deg)  ... dummy value
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
        # [0]     |  Model type [2]      
        # [1]     |  X coordinate (deg)    ... dummy value
        # [2]     |  Y coordinate (deg)    ... dummy value
        # [3]     |  Major axis (arcsec)   ... dummy value
        # [4]     |  Minor axis (arcsec)   ... dummy value
        # [5]     |  Position angle (deg)  ... dummy value
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

    Global properties of the data (e.g., frequency sampling, noise level) are
    set by arguments to this script, including an optional template for the
    shape of the noise curve and an optional list of frequency ranges to flag.
    See below for a list and default values in square brackets.

    Examples:

    ./mk_test_ascii_data.py catalogue.csv data/

    ./mk_test_ascii_data.py catalogue.csv -f 1.10e9,1.20e9,1.60e9,1.65e9

    ./mk_test_ascii_data.py catalogue.csv -n NOISE.TXT 

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("inCatFile", metavar="catalogue.csv", nargs=1,
                        help="Input catalogue file in CSV format")
    parser.add_argument("dataPath", metavar="PATH/TO/DATA",
                        default="data/", nargs="?",
                        help="Path to new data directory [data/]")
    parser.add_argument('-f', nargs=2, metavar="X", type=float,
                        dest='freqRng_MHz', default=[700.0, 1800.0],
                        help='Frequency range [700 1800] (MHz)')
    parser.add_argument("-f0", dest="freq0_MHz", type=float, default=0.0,
                        help="Frequency of catalogue flux [1st channel] (MHz).")
    parser.add_argument("-c", dest="nChans", type=int, default=111,
                        help="Number of channels in output spectra [111].")
    parser.add_argument("-n", dest="rmsNoise_mJy", type=float, default=0.02,
                        help="RMS noise of the output spectra [0.02 mJy].")
    parser.add_argument('-t', dest='noiseTmpFile', metavar="NOISE.TXT",
                        help="ASCII file providing a template noise curve (freq amp)")
    parser.add_argument('-l', dest='flagFreqStr', metavar='f1,f2,f1,f2,...',
                        default="", help="Frequency ranges to flag out")
    args = parser.parse_args()
    inCatFile = args.inCatFile[0]
    dataPath = args.dataPath
    startFreq_Hz = args.freqRng_MHz[0] * 1e6
    endFreq_Hz = args.freqRng_MHz[1] * 1e6
    freq0_Hz = None
    if args.freq0_MHz>0.0:
        freq0_Hz = args.freq0_MHz * 1e6
    nChans = args.nChans
    rmsNoise_mJy = args.rmsNoise_mJy
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
    
    # Call the function to create the ASCII data files on disk
    nSrc = create_IQU_ascii_data(dataPath, inCatFile, startFreq_Hz,
                                 endFreq_Hz, nChans, rmsNoise_mJy,
                                 noiseTmpArr, flagRanges_Hz, freq0_Hz)

    
#-----------------------------------------------------------------------------#
def create_IQU_ascii_data(dataPath, inCatFile, startFreq_Hz, endFreq_Hz, 
                          nChans, rmsNoise_mJy, noiseTmpArr=None,
                          flagRanges_Hz=[], freq0_Hz=None):
    """
    Create a set of ASCII files containing Stokes I Q & U spectra.
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
    print("Input RMS noise is %.3g mJy" % rmsNoise_mJy)
    if noiseTmpArr is None:
        print("Assuming flat noise versus frequency curve.")
        noiseArr = np.ones(freqArr_Hz.shape, dtype="f8")
    else:
        print("Scaling noise by external template curve.")
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

    # Set the frequency at which the flux has been defined
    if freq0_Hz is None:
        freq0_Hz = startFreq_Hz
    print("Catalogue flux is defined at %.3f MHz" % (freq0_Hz/1e6))
    
    # Create the output directory path
    dataPath = dataPath.rstrip("/")
    print("Creating test dataset in '%s/'" % dataPath)
    dirs = dataPath.split("/")
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

    # Loop through the sources, calculate the spectra and save to disk
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
        
        # Add scatter to the data to simulate noise
        rmsNoise_Jy = rmsNoise_mJy/1e3
        IArr_Jy += (np.random.normal(scale=rmsNoise_Jy, size=IArr_Jy.shape)
                    * noiseArr)
        QArr_Jy += (np.random.normal(scale=rmsNoise_Jy, size=QArr_Jy.shape)
                    * noiseArr)
        UArr_Jy += (np.random.normal(scale=rmsNoise_Jy, size=UArr_Jy.shape)
                    * noiseArr)
        dIArr_Jy = noiseArr * rmsNoise_Jy
        dIArr_Jy *= np.random.normal(loc=1.0, scale=0.05, size=noiseArr.shape)
        dQArr_Jy = noiseArr * rmsNoise_Jy 
        dQArr_Jy *= np.random.normal(loc=1.0, scale=0.05, size=noiseArr.shape)
        dUArr_Jy = noiseArr * rmsNoise_Jy 
        dUArr_Jy *= np.random.normal(loc=1.0, scale=0.05, size=noiseArr.shape)
        
        # Save spectra to disk
        outFileName = "Source%d.dat" % (i+1)
        outFilePath = dataPath + "/" + outFileName
        print("> Writing ASCII file '%s' ..." % outFileName, end=' ')
        np.savetxt(outFilePath,
                   np.column_stack((freqArr_Hz,
                                    IArr_Jy,
                                    QArr_Jy,
                                    UArr_Jy,
                                    dIArr_Jy,
                                    dQArr_Jy,
                                    dUArr_Jy)))
        print("done.")
        successCount += 1

    return successCount


#-----------------------------------------------------------------------------#
if __name__=="__main__":
    main()
