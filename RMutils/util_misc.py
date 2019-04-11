#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     util_misc.py                                                      #
#                                                                             #
# PURPOSE:  Miscellaneous functions for working with Faraday spectra & cubes. #
#                                                                             #
# REQUIRED: Requires numpy and astropy.                                       #
#                                                                             #
# MODIFIED: 12-Mar-2018 by C. Purcell                                         #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#  config_read          ... read a key=value format text file                 #
#  csv_read_to_list     ... read rows from a CSV file into a list of lists    #
#  cleanup_str_input    ... condense multiple newlines and spaces in a string #
#  split_repeat_lst     ... split a list into preamble and repeating columns  #
#  deg2dms              ... convert decimal degrees to dms string             #
#  progress             ... print a progress bar                              #
#  calc_mom2_FDF        ... calculate the 2nd moment of the CC                #
#  calc_parabola_vertex ... interpolate the peak position using a parabola    #
#  create_frac_spectra  ... fit a Stokes I model and divide into other Stokes #
#  interp_images        ... interpolate between two image planes              #
#  fit_spec_poly5       ... fit a >=5th order polynomial to a spectrum        #
#  poly5                ... function to evaluate a 5th order polynomial       #
#  nanmedian            ... np.median ignoring NaNs                           #
#  nanmean              ... np.mean ignoring NaNs                             #
#  nanstd               ... np.std ignoring NaNs                              #
#  extrap               ... interpolate & extrapolate a Numpy array           #
#  toscalar             ... return a scalar version of a Numpy object         #
#  MAD                  ... calculate the madfm                               #
#  calc_stats           ... calculate the statistics of an array              #
#  sort_nicely          ... sort a list in the order a human would            #
#  twodgaussian         ... return an array containing a 2D Gaussian          #
#  create_pqu_spectra_burn ... return fractional spectra for N burn sources   #
#  create_IQU_spectra_burn ... return IQU spectra for N burn sources          #
#  create_pqu_spectra_diff ... return fractional spectra for N mixed sources  #
#  create_IQU_spectra_diff ... return IQU spectra for N mixed sources         #
#  create_pqu_spectra_RMthin ... return fractional spectra for a thin source  #
#  create_IQU_spectra_RMthin ... return IQU spectra for a thin source         #
#  create_pqu_resid_RMthin ... return fractional spectra - a thin component   #
#  xfloat               ... convert to float, default to None on fail         #
#  norm_cdf             ... calculate the CDF of a Normal distribution        #
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

import os
import sys
import copy
import re
import time
import traceback
import math as m
import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndi
from scipy.stats import norm
#import ConfigParser
import sqlite3
import csv
import json

from RMutils.mpfit import mpfit

C = 2.99792458e8


#-----------------------------------------------------------------------------#
def config_read(filename, delim='=', doValueSplit=True):
    """
    Read a configuration file and output a 'KEY=VALUE' dictionary.
    """

    configTable = {}
    CONFIGFILE = open(filename, "r")
    
    # Compile a few useful regular expressions
    spaces = re.compile('\s+')
    commaAndSpaces = re.compile(',\s+')
    commaOrSpace = re.compile('[\s|,]')
    brackets = re.compile('[\[|\]\(|\)|\{|\}]')
    comment = re.compile('#.*')
    quotes = re.compile('\'[^\']*\'')
    keyVal = re.compile('^.+' + delim + '.+')

    # Read in the input file, line by line
    for line in CONFIGFILE:

        valueLst=[]
        line = line.rstrip("\n\r")

        # Filter for comments and blank lines
        if not comment.match(line) and keyVal.match(line):

            # Weed out internal comments & split on 1st space
            line = comment.sub('',line)
            (keyword, value) = line.split(delim,1)

            # If the line contains a value
            keyword = keyword.strip()              # kill external whitespace
            keyword = spaces.sub('', keyword)      # kill internal whitespaces
            value = value.strip()                  # kill external whitespace
            value = spaces.sub(' ', value)         # shrink internal whitespace
            value = value.replace("'", '')         # kill quotes
            value = commaAndSpaces.sub(',', value) # kill ambiguous spaces

            # Split comma/space delimited value strings
            if doValueSplit:
                valueLst = commaOrSpace.split(value)
                if len(valueLst)<=1:
                    valueLst = valueLst[0]
                configTable[keyword] = valueLst
            else:
                configTable[keyword] = value

    return configTable


#-----------------------------------------------------------------------------#
def csv_read_to_list(fileName, delim=",", doFloat=False):
    """Read rows from an ASCII file into a list of lists."""

    outLst = []
    DATFILE = open(fileName, "r")
     
    # Compile a few useful regular expressions
    spaces = re.compile('\s+')
    comma_and_spaces = re.compile(',\s+')
    comma_or_space = re.compile('[\s|,]')
    brackets = re.compile('[\[|\]\(|\)|\{|\}]')
    comment = re.compile('#.*')
    quotes = re.compile('\'[^\']*\'')
    keyVal = re.compile('^.+=.+')
    words = re.compile('\S+')

    # Read in the input file, line by line
    for line in DATFILE:
        line = line.rstrip("\n\r")
        if comment.match(line):
            continue
        line = comment.sub('', line)     # remove internal comments
        line = line.strip()              # kill external whitespace
        line = spaces.sub(' ', line)     # shrink internal whitespace
        if line=='':
            continue
        line = line.split(delim)
        if len(line)<1:
            continue
        if doFloat:
            line = [float(x) for x in line]
        
        outLst.append(line)

    return outLst


#-----------------------------------------------------------------------------#
def cleanup_str_input(textBlock):
    
    # Compile a few useful regular expressions
    spaces = re.compile(r"[^\S\r\n]+")
    newlines = re.compile(r"\n+")
    rets = re.compile(r"\r+")

    # Strip multiple spaces etc
    textBlock = textBlock.strip()
    textBlock = rets.sub('\n', textBlock)
    textBlock = newlines.sub('\n', textBlock)
    textBlock = spaces.sub(' ', textBlock)

    return textBlock


#-----------------------------------------------------------------------------#
def split_repeat_lst(inLst, nPre, nRepeat):
    """Split entries in a list into a preamble and repeating columns. The 
    repeating entries are pushed into a 2D array of type float64."""


    preLst = inLst[:nPre]
    repeatLst = list(zip(*[iter(inLst[nPre:])]*nRepeat))
    parmArr = np.array(repeatLst, dtype="f8").transpose()

    return preLst, parmArr


#-----------------------------------------------------------------------------#
def deg2dms(deg, delim=':', doSign=False, nPlaces=2):
    """
    Convert a float in degrees to 'dd mm ss' format.
    """

    try:
        angle = abs(deg)
        sign=1
        if angle!=0: sign = angle/deg
        
        # Calcuate the degrees, min and sec
        dd = int(angle)
        rmndr = 60.0*(angle - dd)
        mm = int(rmndr)
        ss = 60.0*(rmndr-mm)

        # If rounding up to 60, carry to the next term
        if float("%05.2f" % ss) >=60.0:
            mm+=1.0
            ss = ss - 60.0
        if float("%02d" % mm) >=60.0:
            dd+=1.0
            mm = mm -60.0
        if nPlaces> 0:
            formatCode = "%0" + "%s.%sf" % (str(2 + nPlaces + 1), str(nPlaces))
        else:
            formatCode = "%02.0f"
        if sign>0:
            if doSign:
                formatCode = "+%02d%s%02d%s" + formatCode
            else:
                formatCode = "%02d%s%02d%s" + formatCode
        else:
            formatCode = "-%02d%s%02d%s" + formatCode
        return formatCode % (dd, delim, mm, delim, ss)
        
    except Exception:
        return None


#-----------------------------------------------------------------------------#
def progress(width, percent):
    """
    Print a progress bar to the terminal.
    Stolen from Mike Bell.
    """
    
    marks = m.floor(width * (percent / 100.0))
    spaces = m.floor(width - marks)
    loader = '  [' + ('=' * int(marks)) + (' ' * int(spaces)) + ']'
    #sys.stdout.write("%s %d%%\r" % (loader, percent))
    #if percent >= 100:
    #    sys.stdout.write("\n")
    #sys.stdout.flush()


#-----------------------------------------------------------------------------#
def calc_mom2_FDF(FDF, phiArr):
    """
    Calculate the 2nd moment of the polarised intensity FDF. Can be applied to
    a clean component spectrum or a standard FDF
    """
    
    K = np.sum( np.abs(FDF) )
    phiMean = np.sum( phiArr * np.abs(FDF) ) / K
    phiMom2 = np.sqrt( np.sum( np.power((phiArr - phiMean), 2.0) *
                                np.abs(FDF) ) / K )
    
    return phiMom2


#-----------------------------------------------------------------------------#
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    """
    Calculate the vertex of a parabola given three adjacent points.
    """
    
    D = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / D
    B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / D
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 *
         (x1 - x2) * y3) / D

    xv = -B / (2.0 * A)
    yv = C - B * B / (4.0 * A)

    return xv, yv

#-----------------------------------------------------------------------------#
def create_frac_spectra(freqArr, IArr, QArr, UArr, dIArr, dQArr, dUArr,
                        polyOrd=5, verbose=False, debug=False):
    """Fit the Stokes I spectrum with a polynomial and divide into the Q & U
    spectra to create fractional spectra."""

    ### TODO: loop to decrease order if chiSq<1 to guard against over-fitting

    # Fit a <=5th order polynomial model to the Stokes I spectrum
    # Frequency axis must be in GHz to avoid overflow errors
    fitDict = {"fitStatus": 0,
               "chiSq": 0.0,
               "dof": len(freqArr)-polyOrd-1,
               "chiSqRed": 0.0,
               "nIter": 0,
               "p": None}
    try:
        mp = fit_spec_poly5(freqArr, IArr, dIArr, polyOrd)
        fitDict["p"] = mp.params
        fitDict["fitStatus"] = mp.status
        fitDict["chiSq"] = mp.fnorm
        fitDict["chiSqRed"] = mp.fnorm/fitDict["dof"]
        fitDict["nIter"] = mp.niter
        IModArr = poly5(fitDict["p"])(freqArr)

        #if verbose:
        #    print("\n")
        #    print("-"*80)
        #    print("Details of the polynomial fit to the spectrum:")
        #    for key, val in fitDict.iteritems():
        #        print(" %s = %s" % (key, val))
        #    print("-"*80)
        #    print("\n")
    except Exception:
        print("Err: Failed to fit polynomial to Stokes I spectrum.")
        if debug:
            print("\nTRACEBACK:")
            print(("-" * 80))
            print((traceback.format_exc()))
            print(("-" * 80))
            print("\n")
        print("> Setting Stokes I spectrum to unity.\n")
        fitDict["p"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        IModArr = np.ones_like(IArr)
    
    # Calculate the fractional spectra and errors
    with np.errstate(divide='ignore', invalid='ignore'):
        qArr = np.true_divide(QArr, IModArr)
        uArr = np.true_divide(UArr, IModArr)
        dqArr = qArr * np.sqrt( np.true_divide(dQArr, QArr)**2.0 +
                                np.true_divide(dIArr, IArr)**2.0 )
        duArr = uArr * np.sqrt( np.true_divide(dUArr, UArr)**2.0 +
                                np.true_divide(dIArr, IArr)**2.0 )

    return IModArr, qArr, uArr, dqArr, duArr, fitDict


#-----------------------------------------------------------------------------#
def interp_images(arr1, arr2, f=0.5):
    """Create an interpolated image between two other images."""
    
    nY, nX =  arr1.shape
    
    # Concatenate arrays into a single array of shape (2, nY, nX)
    arr = np.r_["0,3", arr1, arr2]
    
    # Define the grid coordinates where you want to interpolate
    X, Y = np.meshgrid(np.arange(nX), np.arange(nY))
    
    # Create coordinates for interpolated frame
    coords = np.ones(arr1.shape) * f, Y, X
    
    # Interpolate using the map_coordinates function
    interpArr = ndi.map_coordinates(arr, coords, order=2)

    return interpArr

    
#-----------------------------------------------------------------------------#
def fit_spec_poly5(xData, yData, dyData=None, order=5):
    """Fit a 5th order polynomial to a spectrum. To avoid overflow errors the
    X-axis data should not be large numbers (e.g.: x10^9 Hz; use GHz
    instead)."""

    # Impose limits on polynomial order
    if order<1:
        order = 1
    if order>5:
        order = 5
    if dyData is None:
        dyData = np.ones_like(yData)
    if np.all(dyData==0):
        dyData = np.ones_like(yData)
        
    # Estimate starting coefficients
    C1 = nanmean(np.diff(yData)) / nanmedian(np.diff(xData))
    ind = int(np.median(np.where(~np.isnan(yData))))
    C0 = yData[ind] - (C1 * xData[ind])
    C5 = 0.0
    C4 = 0.0
    C3 = 0.0
    C2 = 0.0
    inParms=[ {'value': C5, 'parname': 'C5', 'fixed': False},
              {'value': C4, 'parname': 'C4', 'fixed': False},
              {'value': C3, 'parname': 'C3', 'fixed': False},
              {'value': C2, 'parname': 'C2', 'fixed': False},
              {'value': C1, 'parname': 'C1', 'fixed': False},
              {'value': C0, 'parname': 'C0', 'fixed': False} ]
    
    # Set the parameters as fixed of > order
    for i in range(len(inParms)):
        if len(inParms)-i-1>order:
            inParms[i]['fixed'] = True

    # Function to evaluate the difference between the model and data.
    # This is minimised in the least-squared sense by the fitter
    def errFn(p, fjac=None):
        status = 0
        return status, (poly5(p)(xData) - yData)/dyData

    # Use MPFIT to perform the LM-minimisation
    mp = mpfit(errFn, parinfo=inParms, quiet=True)
    
    return mp


#-----------------------------------------------------------------------------
def poly5(p):
    """Returns a function to evaluate a polynomial. The subfunction can be
    accessed via 'argument unpacking' like so: 'y = poly5(p)(*x)', 
    where x is a vector of X values and p is a vector of coefficients."""

    # Fill out the vector to length 6 if necessary
    p = np.append(np.zeros((6-len(p))), p)
    
    def rfunc(x):
        y = (p[0]*x**5.0 + p[1]*x**4.0 + p[2]*x**3.0 + p[3]*x**2.0 + p[4]*x
             + p[5])
        return y
             
    return rfunc

#-----------------------------------------------------------------------------#
def nanmedian(arr, **kwargs):
    """
    Returns median ignoring NaNs.
    """
    
    return ma.median( ma.masked_where(arr!=arr, arr), **kwargs )


#-----------------------------------------------------------------------------#
def nanmean(arr, **kwargs):
    """
    Returns mean ignoring NaNs.
    """
    
    return ma.mean( ma.masked_where(arr!=arr, arr), **kwargs )

#-----------------------------------------------------------------------------#
def nanstd(arr, **kwargs):
    """
    Returns standard deviation ignoring NaNs.
    """
    
    return ma.std( ma.masked_where(arr!=arr, arr), **kwargs )

#-----------------------------------------------------------------------------#
def extrap(x, xp, yp):
    """
    Wrapper to allow np.interp to linearly extrapolate at function ends.
    
    np.interp function with linear extrapolation
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate
    -give-a-an-extrapolated-result-beyond-the-input-ran
    """
    
    y = np.interp(x, xp, yp)
    y = np.where(x < xp[0], yp[0]+(x-xp[0])*(yp[0]-yp[1])/(xp[0]-xp[1]), y)
    y = np.where(x > xp[-1], yp[-1]+(x-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2]),
                 y)
    return y


#-----------------------------------------------------------------------------#
def toscalar(a):
    """
    Returns a scalar version of a Numpy object.
    """
    try:
        return np.asscalar(a)
    except Exception:
        return a


#-----------------------------------------------------------------------------#
def MAD(a, c=0.6745, axis=None):
    """
    Median Absolute Deviation along given axis of an array:
    median(abs(a - median(a))) / c
    c = 0.6745 is the constant to convert from MAD to std
    """
    
    a = ma.masked_where(a!=a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = ma.median(ma.fabs(a - d) / c)
    else:
        d = ma.median(a, axis=axis)
        if axis > 0:
            aswp = ma.swapaxes(a,0,axis)
        else:
            aswp = a
        m = ma.median(ma.fabs(aswp - d) / c, axis=0)

    return m


#-----------------------------------------------------------------------------#
def calc_stats(a, maskzero=False):
    """
    Calculate the statistics of an array.
    """
    
    statsDict = {}
    a = np.array(a)

    # Mask off bad values and count valid pixels
    if maskzero:
        a = np.where( np.equal(a, 0.0), np.nan, a)
    am = ma.masked_invalid(a)
    statsDict['npix'] = np.sum(~am.mask)
    
    if statsDict['npix']>=2:
        statsDict['stdev'] = float(np.std(am))
        statsDict['mean'] = float(np.mean(am))
        statsDict['median'] = float(nanmedian(am))
        statsDict['max'] = float(np.max(am))
        statsDict['min'] = float(np.min(am))
        statsDict['centmax'] = list(np.unravel_index(np.argmax(am),
                                                     a.shape))
        statsDict['madfm'] = float(MAD(am.flatten()))
        statsDict['success'] = True
        
    else:
        statsDict['npix'] == 0
        statsDict['stdev']   = 0.0
        statsDict['mean']    = 0.0
        statsDict['median']  = 0.0
        statsDict['max']     = 0.0
        statsDict['min']     = 0.0
        statsDict['centmax'] = (0.0, 0.0)
        statsDict['madfm']   = 0.0
        statsDict['success'] = False
        
    return statsDict


#-----------------------------------------------------------------------------#
def sort_nicely(l):
    """
    Sort a list in the order a human would.
    """
    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort( key=alphanum_key ) 


#-----------------------------------------------------------------------------#
def twodgaussian(params, shape):
    """
    Build a 2D Gaussian ellipse as parameterised by 'params' for a region with
    'shape'
        params - [amp, xo, yo, cx, cy, pa] where:
                amp - amplitude
                xo  - centre of Gaussian in X
                yo  - centre of Gaussian in Y
                cx  - width of Gaussian in X (sigma or c, not FWHM)
                cy  - width of Gaussian in Y (sigma or c, not FWHM)
                pa  - position angle of Gaussian, aka theta (radians)
        shape - (y, x) dimensions of region
    Returns a 2D numpy array with shape="shape" 
    """
    
    assert(len(shape) == 2)
    amp, xo, yo, cx, cy, pa = params
    y, x = np.indices(shape)
    st = m.sin(pa)**2
    ct = m.cos(pa)**2
    s2t = m.sin(2*pa)
    a = (ct/cx**2 + st/cy**2)/2
    b = s2t/4 *(1/cy**2-1/cx**2)
    c = (st/cx**2 + ct/cy**2)/2
    v = amp*np.exp(-1*(a*(x-xo)**2 + 2*b*(x-xo)*(y-yo) + c*(y-yo)**2))
    
    return v


#-----------------------------------------------------------------------------#
def create_pqu_spectra_burn(freqArr_Hz, fracPolArr, psi0Arr_deg,
                              RMArr_radm2, sigmaRMArr_radm2=None):
    """Return fractional P/I, Q/I & U/I spectra for a sum of Faraday thin
    components (multiple values may be given as a list for each argument).
    Burn-law external depolarisation may be applied to each
    component via the optional 'sigmaRMArr_radm2' argument. If
    sigmaRMArr_radm2=None, all values are set to zero, i.e., no
    depolarisation."""

    # Convert lists to arrays
    freqArr_Hz = np.array(freqArr_Hz, dtype="f8")
    fracPolArr = np.array(fracPolArr, dtype="f8")
    psi0Arr_deg = np.array(psi0Arr_deg, dtype="f8")
    RMArr_radm2 = np.array(RMArr_radm2, dtype="f8")
    if sigmaRMArr_radm2 is None:
        sigmaRMArr_radm2 = np.zeros_like(fracPolArr)
    else:
        sigmaRMArr_radm2 = np.array(sigmaRMArr_radm2, dtype="f8")

    # Calculate some prerequsites
    nChans = len(freqArr_Hz)
    nComps = len(fracPolArr)
    lamArr_m = C/freqArr_Hz
    lamSqArr_m2 = np.power(lamArr_m, 2.0)
            
    # Convert the inputs to column vectors
    fracPolArr = fracPolArr.reshape((nComps, 1))
    psi0Arr_deg = psi0Arr_deg.reshape((nComps, 1))
    RMArr_radm2 = RMArr_radm2.reshape((nComps, 1))
    sigmaRMArr_radm2 = sigmaRMArr_radm2.reshape((nComps, 1))
    
    # Calculate the p, q and u Spectra for all components
    pArr = fracPolArr *  np.ones((nComps, nChans), dtype="f8")
    quArr = pArr * (
        np.exp( 2j * (np.radians(psi0Arr_deg) + RMArr_radm2*lamSqArr_m2) )
        * np.exp(-2.0 * sigmaRMArr_radm2 * np.power(lamArr_m, 4.0))
        )
    
    # Sum along the component axis to create the final spectra
    quArr = quArr.sum(0)
    qArr = quArr.real
    uArr = quArr.imag
    pArr = np.abs(quArr)
    
    return pArr, qArr, uArr


#-----------------------------------------------------------------------------#
def create_IQU_spectra_burn(freqArr_Hz, fluxI, SI, fracPolArr, psi0Arr_deg,
                              RMArr_radm2, sigmaRMArr_radm2=None,
                              freq0_Hz=None):
    """Create Stokes I, Q & U spectra for a source with 1 or more polarised
    Faraday components affected by external (burn) depolarisation."""
    
    # Create the polarised fraction spectra
    pArr, qArr, uArr = create_pqu_spectra_burn(freqArr_Hz,
                                               fracPolArr,
                                               psi0Arr_deg,
                                               RMArr_radm2,
                                               sigmaRMArr_radm2)
    
    # Default reference frequency is first channel
    if freq0_Hz is None:
        freq0_Hz = freqArr_Hz[0]
        
    # Create the absolute value spectra
    IArr = fluxI * np.power(freqArr_Hz/freq0_Hz, SI)
    PArr = IArr * pArr
    QArr = IArr * qArr
    UArr = IArr * uArr
    
    return IArr, QArr, UArr


#-----------------------------------------------------------------------------#
def create_pqu_spectra_diff(freqArr_Hz, fracPolArr, psi0Arr_deg, RMArr_radm2):
    """Return fractional P/I, Q/I & U/I spectra for a sum of Faraday 
    components which are affected by internal (differential) Faraday
    depolariation."""

    # Convert lists to arrays
    freqArr_Hz = np.array(freqArr_Hz, dtype="f8")
    fracPolArr = np.array(fracPolArr, dtype="f8")
    psi0Arr_rad = np.radians(psi0Arr_deg, dtype="f8")
    RMArr_radm2 = np.array(RMArr_radm2, dtype="f8")

    # Calculate some prerequsites
    nChans = len(freqArr_Hz)
    nComps = len(fracPolArr)
    lamArr_m = C/freqArr_Hz
    lamSqArr_m2 = np.power(lamArr_m, 2.0)

    # Convert the inputs to column vectors
    fracPolArr = fracPolArr.reshape((nComps, 1))
    psi0Arr_deg = psi0Arr_deg.reshape((nComps, 1))
    RMArr_radm2 = RMArr_radm2.reshape((nComps, 1))

    # Calculate the p, q and u Spectra for all components
    RMLamSqArr = RMArr_radm2*lamSqArr_m2
    pArr = fracPolArr * np.sinc(RMLamSqArr/np.pi)
    pArr = pArr.astype("complex")
    for i in range(nComps):
        RMLamSqArr[i] *= 0.5
        pArr[i] *= np.exp(2j * (psi0Arr_rad[i] + RMLamSqArr[i:].sum(0)))
    
    # Sum along the component axis to create the final spectra
    pArr =pArr.sum(0)
    qArr = pArr.real
    uArr = pArr.imag
    pArr = np.abs(pArr)
    
    return pArr, qArr, uArr


#-----------------------------------------------------------------------------#
def create_IQU_spectra_diff(freqArr_Hz, fluxI, SI, fracPolArr, psi0Arr_deg,
                              RMArr_radm2, freq0_Hz=None):
    """Create Stokes I, Q & U spectra for a source with 1 or more polarised
    Faraday components affected by internal Faraday depolarisation"""
    
    # Create the polarised fraction spectra
    pArr, qArr, uArr = create_pqu_spectra_diff(freqArr_Hz,
                                               fracPolArr,
                                               psi0Arr_deg,
                                               RMArr_radm2)
    
    # Default reference frequency is first channel
    if freq0_Hz is None:
        freq0_Hz = freqArr_Hz[0]
        
    # Create the absolute value spectra
    IArr = fluxI * np.power(freqArr_Hz/freq0_Hz, SI)
    PArr = IArr * pArr
    QArr = IArr * qArr
    UArr = IArr * uArr
    
    return IArr, QArr, UArr

    
#-----------------------------------------------------------------------------#
def create_pqu_spectra_RMthin(freqArr_Hz, fracPol, psi0_deg, RM_radm2):
    """Return fractional P/I, Q/I & U/I spectra for a Faraday thin source"""
    
    # Calculate the p, q and u Spectra
    lamSqArr_m2 = np.power(C/freqArr_Hz, 2.0)
    pArr = fracPol * np.ones_like(lamSqArr_m2)
    quArr = pArr * np.exp( 2j * (np.radians(psi0_deg) +
                                 RM_radm2 * lamSqArr_m2 ) )
    qArr = quArr.real
    uArr = quArr.imag
    
    return pArr, qArr, uArr


#-----------------------------------------------------------------------------#
def create_IQU_spectra_RMthin(freqArr_Hz, fluxI, SI, fracPol, psi0_deg, 
                              RM_radm2, freq0_Hz=None):
    """Return Stokes I, Q & U spectra for a Faraday thin source"""

    pArr, qArr, uArr = create_pqu_spectra_RMthin(freqArr_Hz,
                                                 fracPol,
                                                 psi0_deg, 
                                                 RM_radm2)
    if freq0_Hz is None:
        freq0_Hz = freqArr_Hz[0]
    IArr = fluxI * np.power(freqArr_Hz/freq0_Hz, SI)
    PArr = IArr * pArr
    QArr = IArr * qArr
    UArr = IArr * uArr

    return IArr, QArr, UArr


#-----------------------------------------------------------------------------#
def create_pqu_resid_RMthin(qArr, uArr, freqArr_Hz, fracPol, psi0_deg,
                            RM_radm2):
    """Subtract a RM-thin component from the fractional q and u data."""

    pModArr, qModArr, uModArr = create_pqu_spectra_RMthin(freqArr_Hz,
                                                          fracPol,
                                                          psi0_deg,
                                                          RM_radm2)
    qResidArr = qArr - qModArr
    uResidArr = uArr - uModArr
    pResidArr = np.sqrt(qResidArr**2.0 + uResidArr**2.0)

    return pResidArr, qResidArr, uResidArr


#-----------------------------------------------------------------------------#
def xfloat(x, default=None):

    if x is None or x is "":
        return default
    try:
        return float(x)
    except Exception:
        return default

    
#-----------------------------------------------------------------------------#
def norm_cdf(mean=0.0, std=1.0, N=50, xArr=None):
    """Return the CDF of a normal distribution between -6 and 6 sigma, or at
    the values of an input array."""
    
    if xArr is None:
        x = np.linspace(-6.0*std, 6.0*std, N)
    else:
        x = xArr
    y = norm.cdf(x, loc=mean, scale=std)
    
    return x, y
