#!/usr/bin/env python
# =============================================================================#
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

import math as m
import re
import sys
import traceback
import warnings
from typing import NamedTuple, Optional, Tuple

import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndi
from astropy.constants import c as speed_of_light
from deprecation import deprecated
from scipy.stats import norm

from RMutils.mpfit import mpfit

from . import __version__

# import ConfigParser


def update_position_wcsaxes(header):
    # Store and delete the WCSAXES keyword
    wcsaxes = header["WCSAXES"]
    del header["WCSAXES"]

    # Determine the correct insertion point before the first WCS-related keyword
    wcs_keywords = [
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
        "CTYPE1",
        "CTYPE2",
        "CUNIT1",
        "CUNIT2",
        "PC1_1",
        "PC2_2",
        "CD1_1",
        "CD2_2",
    ]

    # Convert the header keys to a list
    header_keys = list(header.keys())

    # Find the first occurrence of any WCS-related keyword
    insert_pos = min(
        header_keys.index(key) for key in wcs_keywords if key in header_keys
    )

    # Insert WCSAXES at the correct position
    header.insert(insert_pos, ("WCSAXES", wcsaxes))

    return header


def remove_header_third_fourth_axis(header):
    """Removes extra axes from header to compress down to 2 axes"""
    # List of keys related to the 3rd and 4th axes to remove (essentially everything with a '3' or '4')
    keys_to_remove = [
        "NAXIS3",
        "NAXIS4",
        "CRPIX3",
        "CRPIX4",
        "CDELT3",
        "CDELT4",
        "CUNIT3",
        "CUNIT4",
        "CTYPE3",
        "CTYPE4",
        "CRVAL3",
        "CRVAL4",
        "PC1_3",
        "PC2_3",
        "PC3_3",
        "PC4_3",
        "PC1_4",
        "PC2_4",
        "PC3_4",
        "PC4_4",
        "PC3_1",
        "PC3_2",
        "PC3_3",
        "PC3_4",
        "PC4_1",
        "PC4_2",
        "PC4_3",
        "PC4_4",
    ]

    for key in keys_to_remove:
        # Header can dynamically change when keys are removed so use pop
        header.pop(key, None)

    # Set correct NAXIS
    header.set("NAXIS", 2)

    # Remove STOKES axis for 2D maps
    header.pop("STOKES", None)

    # Finally set correct WCSAXES param
    header.set("WCSAXES", 2)

    # To obey fitsverify, the WCSAXES param must come before the other WCS params
    header = update_position_wcsaxes(header)

    return header


# -----------------------------------------------------------------------------#
@deprecated(
    deprecated_in="1.3.1",
    removed_in="1.4",
    current_version=__version__,
    details="This function is not used anywhere in current RM-Tools.",
)
def config_read(filename, delim="=", doValueSplit=True):
    """
    Read a configuration file and output a 'KEY=VALUE' dictionary.
    """

    configTable = {}
    CONFIGFILE = open(filename, "r")

    # Compile a few useful regular expressions
    spaces = re.compile(r"\s+")
    commaAndSpaces = re.compile(r",\s+")
    commaOrSpace = re.compile(r"[\s|,]")
    brackets = re.compile(r"[\[|\]\(|\)|\{|\}]")
    comment = re.compile(r"#.*")
    quotes = re.compile(r"'[^']*'")
    keyVal = re.compile(r"^.+" + delim + ".+")

    # Read in the input file, line by line
    for line in CONFIGFILE:
        valueLst = []
        line = line.rstrip("\n\r")

        # Filter for comments and blank lines
        if not comment.match(line) and keyVal.match(line):
            # Weed out internal comments & split on 1st space
            line = comment.sub("", line)
            (keyword, value) = line.split(delim, 1)

            # If the line contains a value
            keyword = keyword.strip()  # kill external whitespace
            keyword = spaces.sub("", keyword)  # kill internal whitespaces
            value = value.strip()  # kill external whitespace
            value = spaces.sub(" ", value)  # shrink internal whitespace
            value = value.replace("'", "")  # kill quotes
            value = commaAndSpaces.sub(",", value)  # kill ambiguous spaces

            # Split comma/space delimited value strings
            if doValueSplit:
                valueLst = commaOrSpace.split(value)
                if len(valueLst) <= 1:
                    valueLst = valueLst[0]
                configTable[keyword] = valueLst
            else:
                configTable[keyword] = value

    return configTable


# -----------------------------------------------------------------------------#
def csv_read_to_list(fileName, delim=",", doFloat=False):
    """Read rows from an ASCII file into a list of lists."""

    outLst = []
    DATFILE = open(fileName, "r")

    # Compile a few useful regular expressions
    spaces = re.compile(r"\s+")
    comma_and_spaces = re.compile(r",\s+")
    comma_or_space = re.compile(r"[\s|,]")
    brackets = re.compile(r"[\[|\]\(|\)|\{|\}]")
    comment = re.compile(r"#.*")
    quotes = re.compile(r"'[^']*'")
    keyVal = re.compile(r"^.+=.+")
    words = re.compile(r"\S+")

    # Read in the input file, line by line
    for line in DATFILE:
        line = line.rstrip("\n\r")
        if comment.match(line):
            continue
        line = comment.sub("", line)  # remove internal comments
        line = line.strip()  # kill external whitespace
        line = spaces.sub(" ", line)  # shrink internal whitespace
        if line == "":
            continue
        line = line.split(delim)
        if len(line) < 1:
            continue
        if doFloat:
            line = [float(x) for x in line]

        outLst.append(line)

    return outLst


# -----------------------------------------------------------------------------#
@deprecated(
    deprecated_in="1.3.1",
    removed_in="1.4",
    current_version=__version__,
    details="This function is not used anywhere in RM-Tools.",
)
def cleanup_str_input(textBlock):
    # Compile a few useful regular expressions
    spaces = re.compile(r"[^\S\r\n]+")
    newlines = re.compile(r"\n+")
    rets = re.compile(r"\r+")

    # Strip multiple spaces etc
    textBlock = textBlock.strip()
    textBlock = rets.sub("\n", textBlock)
    textBlock = newlines.sub("\n", textBlock)
    textBlock = spaces.sub(" ", textBlock)

    return textBlock


# -----------------------------------------------------------------------------#
def split_repeat_lst(inLst, nPre, nRepeat):
    """Split entries in a list into a preamble and repeating columns. The
    repeating entries are pushed into a 2D array of type float64."""

    preLst = inLst[:nPre]
    repeatLst = list(zip(*[iter(inLst[nPre:])] * nRepeat))
    parmArr = np.array(repeatLst, dtype="f8").transpose()

    return preLst, parmArr


# -----------------------------------------------------------------------------#
@deprecated(
    deprecated_in="1.3.1",
    removed_in="1.4",
    current_version=__version__,
    details="This function is not used anywhere in RM-Tools.",
)
def deg2dms(deg, delim=":", doSign=False, nPlaces=2):
    """
    Convert a float in degrees to 'dd mm ss' format.
    """

    try:
        angle = abs(deg)
        sign = 1
        if angle != 0:
            sign = angle / deg

        # Calcuate the degrees, min and sec
        dd = int(angle)
        rmndr = 60.0 * (angle - dd)
        mm = int(rmndr)
        ss = 60.0 * (rmndr - mm)

        # If rounding up to 60, carry to the next term
        if float("%05.2f" % ss) >= 60.0:
            mm += 1.0
            ss = ss - 60.0
        if float("%02d" % mm) >= 60.0:
            dd += 1.0
            mm = mm - 60.0
        if nPlaces > 0:
            formatCode = "%0" + "%s.%sf" % (str(2 + nPlaces + 1), str(nPlaces))
        else:
            formatCode = "%02.0f"
        if sign > 0:
            if doSign:
                formatCode = "+%02d%s%02d%s" + formatCode
            else:
                formatCode = "%02d%s%02d%s" + formatCode
        else:
            formatCode = "-%02d%s%02d%s" + formatCode
        return formatCode % (dd, delim, mm, delim, ss)

    except Exception:
        return None


# -----------------------------------------------------------------------------#
def progress(width, percent):
    """
    Print a progress bar to the terminal.
    Stolen from Mike Bell.
    """

    marks = m.floor(width * (percent / 100.0))
    spaces = m.floor(width - marks)
    loader = "  [" + ("=" * int(marks)) + (" " * int(spaces)) + "]"
    sys.stdout.write("%s %d%%\r" % (loader, percent))
    if percent >= 100:
        sys.stdout.write("\n")
    sys.stdout.flush()


# -----------------------------------------------------------------------------#
def calc_mom2_FDF(FDF, phiArr):
    """
    Calculate the 2nd moment of the polarised intensity FDF. Can be applied to
    a clean component spectrum or a standard FDF
    """

    K = np.sum(np.abs(FDF))
    phiMean = np.sum(phiArr * np.abs(FDF)) / K
    phiMom2 = np.sqrt(np.sum(np.power((phiArr - phiMean), 2.0) * np.abs(FDF)) / K)

    return phiMom2


# -----------------------------------------------------------------------------#
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    """
    Calculate the vertex of a parabola given three adjacent points.
    Normalization of coordinates must be performed first to reduce risk of
    floating point errors.
    """
    midpoint = x2
    deltax = x2 - x3
    yscale = y2
    (x1, x2, x3) = [(x - x2) / deltax for x in (x1, x2, x3)]  # slide spectrum to zero
    (y1, y2, y3) = [y / yscale for y in (y1, y2, y3)]

    D = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / D
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / D
    C = (
        x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3
    ) / D

    xv = -B / (2.0 * A)
    yv = C - B * B / (4.0 * A)

    return xv * deltax + midpoint, yv * yscale


# -----------------------------------------------------------------------------#
class FitResult(NamedTuple):
    """
    Results of a polynomial fit.
    """

    params: np.ndarray
    """array of polynomial parameters (highest to lowest order)"""
    fitStatus: int
    """exit status of fitter."""
    chiSq: float
    """chi-squared of fit"""
    chiSqRed: float
    """Reduced chi-squared of fit"""
    AIC: float
    """Aikaike information criterion value for fit"""
    polyOrd: int
    """order of fit"""
    nIter: int
    """Number of iterations used by fitter."""
    reference_frequency_Hz: float
    """reference frequency for polynomial."""
    dof: int
    """degrees of freedom in the fit."""
    pcov: np.ndarray
    """covariance matrix of fit parameters"""
    perror: np.ndarray
    """parameter errors"""
    fit_function: str
    """fit function used"""

    def with_options(self, **kwargs):
        prop = self._asdict()
        prop.update(**kwargs)

        return FitResult(**prop)


def fit_StokesI_model(freqArr, IArr, dIArr, polyOrd, fit_function="log") -> FitResult:
    """Fit a model to a Stokes I spectrum with specified errors.
    Supports linear or log polynomials, and fixed or dynamic order selection.

    Args:
        freqArr (array): Numpy array-like containing channel frequencies (in Hz)
        IArr (array): array of Stokes I values (any units)
        dIArr (array): array of 1-sigma noise/uncertainties in Stokes I
        polyOrd (int): if positive (0 to 5), will fit that order of polynomial.
            If negative (-1 to -5), will dynamically find the best order up
            to the maximum of |polyOrd|.
        fit_function (str): 'linear' or 'log' to fit the corresponding function.

    Returns:
        FitResult: Model information.
    """
    # Frequency axis must be in GHz to avoid overflow errors
    goodchan = np.logical_and(
        np.isfinite(IArr), np.isfinite(dIArr)
    )  # Ignore NaN channels!
    # The fitting code is susceptible to numeric overflows because the frequencies are large.
    # To prevent this, normalize by a reasonable characteristic frequency in order
    # to make all the numbers close to 1:
    # The reference frequency is the frequency corresponding to the mean of lambda^2
    # since this should be close to the polarization reference frequency.
    reference_frequency_Hz = 1 / np.sqrt(nanmean(1 / freqArr[goodchan] ** 2))

    # negative orders indicate that the code should dynamically increase
    # order of polynomial so long as it improves the fit
    if polyOrd < 0:
        highest_order = np.abs(polyOrd)
        # Try zero-th order (constant) fit first:
        mp = fit_spec_poly5(
            freqArr[goodchan] / reference_frequency_Hz,
            IArr[goodchan],
            dIArr[goodchan],
            0,
            fit_function,
        )
        old_aic = 2 + mp.fnorm
        current_order = 1
        while current_order <= highest_order:
            new_mp = fit_spec_poly5(
                freqArr[goodchan] / reference_frequency_Hz,
                IArr[goodchan],
                dIArr[goodchan],
                current_order,
                fit_function,
            )
            new_aic = 2 * (current_order + 1) + new_mp.fnorm
            if new_aic < old_aic:  # if there's an improvement, keep going
                old_aic = new_aic
                mp = new_mp
                current_order += 1
            else:  # if not an improvement, stop here.
                break
        polyOrd = (
            current_order - 1
        )  # Best order is always one less than the last one checked.
        aic = old_aic
    else:
        mp = fit_spec_poly5(
            freqArr[goodchan] / reference_frequency_Hz,
            IArr[goodchan],
            dIArr[goodchan],
            polyOrd,
            fit_function,
        )
        aic = 2 * (polyOrd + 1) + mp.fnorm

    dof = len(freqArr) - polyOrd - 1
    return FitResult(
        params=mp.params,
        fitStatus=int(np.abs(mp.status)),
        chiSq=mp.fnorm,
        chiSqRed=mp.fnorm / dof,
        AIC=aic,
        polyOrd=polyOrd,
        nIter=mp.niter,
        reference_frequency_Hz=reference_frequency_Hz,
        dof=dof,
        pcov=mp.covar,
        perror=mp.perror if mp.perror is not None else np.zeros_like(mp.params),
        fit_function=fit_function,
    )


def calculate_StokesI_model(
    fit_result: FitResult, freqArr_Hz: np.ndarray
) -> np.ndarray:
    """Calculates channel values for a Stokes I model.

    Inputs:
        fitDict (FitResult): a dictionary returned from the Stokes I model fitting.
        freqArr_Hz (array): an array of frequency values (assumed to be in Hz).

    Returns: array containing Stokes I model values corresponding to each frequency."""
    if fit_result.fit_function == "linear":
        IModArr = poly5(fit_result.params)(
            freqArr_Hz / fit_result.reference_frequency_Hz
        )
    elif fit_result.fit_function == "log":
        IModArr = powerlaw_poly5(fit_result.params)(
            freqArr_Hz / fit_result.reference_frequency_Hz
        )
    return IModArr


def renormalize_StokesI_model(
    fit_result: FitResult, new_reference_frequency: float
) -> FitResult:
    """Adjust the reference frequency for the Stokes I model and fix the fit
    parameters such that the the model is the same.

    This is important because the initial Stokes I fitted model uses an arbitrary
    reference frequency, and it may be desirable for users to know the exact
    reference frequency of the model.

    This function now includes the ability to transform the model parameter
    errors to the new reference frequency. This feature uses a first order
    approximation, that scales with the ratio of new to old reference frequencies.
    Large changes in reference frequency may be outside the linear valid regime
    of the first order approximation, and thus should be avoided.

    Args:
        fit_result (FitResult): the result of a Stokes I model fit.
        new_reference_frequency (float): the new reference frequency for the model.

    Returns:
        FitResult: the fit results with the reference frequency adjusted to the new value.


    """
    # Renormalization ratio:
    x = new_reference_frequency / fit_result.reference_frequency_Hz

    # Check if ratio is within zone of probable accuracy (approx. 10%, from empirical tests)
    if (x < 0.9) or (x > 1.1):
        warnings.warn(
            "New Stokes I reference frequency more than 10% different than original, uncertainties may be unreliable",
            UserWarning,
        )

    (a, b, c, d, f, g) = fit_result.params

    # Modify fit parameters to new reference frequency.
    # I have derived all these conversion equations analytically for the
    # linear- and log-polynomial models.
    if fit_result.fit_function == "linear":
        new_parms = [a * x**5, b * x**4, c * x**3, d * x**2, f * x, g]
    elif fit_result.fit_function == "log":
        lnx = np.log10(x)
        new_parms = [
            a,
            5 * a * lnx + b,
            10 * a * lnx**2 + 4 * b * lnx + c,
            10 * a * lnx**3 + 6 * b * lnx**2 + 3 * c * lnx + d,
            5 * a * lnx**4 + 4 * b * lnx**3 + 3 * c * lnx**2 + 2 * d * lnx + f,
            g
            * np.power(10, a * lnx**5 + b * lnx**4 + c * lnx**3 + d * lnx**2 + f * lnx),
        ]

    # Modify fit parameter errors to new reference frequency.
    # Note this implicitly makes a first-order approximation in the correletion
    # structure between uncertainties
    # The general equation for the transformation of uncertainties is:
    #   var(p) = sum_i,j((\partial p / \partial a_i) * (\partial p / \partial a_j) * cov(a_i,a_j))
    # where a_i are the initial parameters, p is a final parameter,
    # and the partial derivatives are evaluated at the fit parameter values (and frequency ratio).
    # The partial derivatives all come from the parameter conversion equations above.

    cov = fit_result.pcov
    if fit_result.fit_function == "linear":
        new_errors = [
            np.sqrt(x**10 * cov[0, 0]),
            np.sqrt(x**8 * cov[1, 1]),
            np.sqrt(x**6 * cov[2, 2]),
            np.sqrt(x**4 * cov[3, 3]),
            np.sqrt(x**2 * cov[4, 4]),
            np.sqrt(cov[5, 5]),
        ]
    elif fit_result.fit_function == "log":
        g2 = new_parms[5]  # Convenient shorthand for new value of g variable.
        new_errors = [
            np.sqrt(cov[0, 0]),
            np.sqrt(25 * lnx**2 * cov[0, 0] + 10 * lnx * cov[0, 1] + cov[1, 1]),
            np.sqrt(
                100 * lnx**4 * cov[0, 0]
                + 80 * lnx**3 * cov[0, 1]
                + 20 * lnx**2 * cov[0, 2]
                + 16 * lnx**2 * cov[1, 1]
                + 8 * lnx * cov[1, 2]
                + cov[2, 2]
            ),
            np.sqrt(
                100 * lnx**6 * cov[0, 0]
                + 120 * lnx**5 * cov[0, 1]
                + 60 * lnx**4 * cov[0, 2]
                + 20 * lnx**3 * cov[0, 3]
                + 36 * lnx**4 * cov[1, 1]
                + 36 * lnx**3 * cov[1, 2]
                + 12 * lnx**2 * cov[1, 3]
                + 9 * lnx**2 * cov[2, 2]
                + 6 * lnx * cov[2, 3]
                + cov[3, 3]
            ),
            np.sqrt(
                25 * lnx**8 * cov[0, 0]
                + 40 * lnx**7 * cov[0, 1]
                + 30 * lnx**6 * cov[0, 2]
                + 20 * lnx**5 * cov[0, 3]
                + 10 * lnx**4 * cov[0, 4]
                + 16 * lnx**6 * cov[0, 5]
                + 24 * lnx**5 * cov[1, 2]
                + 16 * lnx**4 * cov[1, 3]
                + 8 * lnx**3 * cov[1, 4]
                + 9 * lnx**4 * cov[2, 2]
                + 12 * lnx**3 * cov[2, 3]
                + 6 * lnx**2 * cov[2, 4]
                + 4 * lnx**2 * cov[3, 3]
                + 4 * lnx * cov[3, 4]
                + cov[4, 4]
            ),
            np.abs(g2)
            * np.sqrt(
                lnx**10 * cov[0, 0]
                + 2 * lnx**9 * cov[0, 1]
                + 2 * lnx**8 * cov[0, 2]
                + 2 * lnx**7 * cov[0, 3]
                + 2 * lnx**6 * cov[0, 4]
                + 2 * lnx**5 / g * np.log(10) * cov[0, 5]
                + lnx**8 * cov[1, 1]
                + 2 * lnx**7 * cov[1, 2]
                + 2 * lnx**6 * cov[1, 3]
                + 2 * lnx**5 * cov[1, 4]
                + 2 * lnx**4 / g * np.log(10) * cov[1, 5]
                + lnx**6 * cov[2, 2]
                + 2 * lnx**5 * cov[2, 3]
                + 2 * lnx**4 * cov[2, 4]
                + 2 * lnx**3 / g * np.log(10) * cov[2, 5]
                + lnx**4 * cov[3, 3]
                + 2 * lnx**3 * cov[3, 4]
                + 2 * lnx**2 / g * np.log(10) * cov[3, 5]
                + lnx**2 * cov[4, 4]
                + 2 * lnx / g * np.log(10) * cov[4, 5]
                + 1 / g**2 * cov[5, 5]
            ),
        ]

    return fit_result.with_options(
        params=new_parms,
        reference_frequency_Hz=new_reference_frequency,
        perror=new_errors,
    )


# -----------------------------------------------------------------------------#
def create_frac_spectra(
    freqArr: np.ndarray,
    IArr: np.ndarray,
    QArr: np.ndarray,
    UArr: np.ndarray,
    dIArr: np.ndarray,
    dQArr: np.ndarray,
    dUArr: np.ndarray,
    polyOrd: int = 2,
    verbose: bool = False,
    debug: bool = False,
    fit_function: str = "log",
    modStokesI: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, FitResult]:
    """
    Fit the Stokes I spectrum with a polynomial and divide into the Q & U
    spectra to create fractional spectra.

    Args:

        freqArr (array): Numpy array-like containing channel frequencies (in Hz)
        IArr (array): array of Stokes I values (any units)
        QArr (array): array of Stokes Q values (any units)
        UArr (array): array of Stokes U values (any units)
        dIArr (array): array of 1-sigma noise/uncertainties in Stokes I
        dQArr (array): array of 1-sigma noise/uncertainties in Stokes Q
        dUArr (array): array of 1-sigma noise/uncertainties in Stokes U
        polyOrd (int): order of polynomial to fit to Stokes I spectrum
        verbose (bool): print extra information
        debug (bool): print debugging information
        fit_function (str): 'linear' or 'log' to fit the corresponding function.
        modStokesI (array): optional Stokes I model to use instead of fitting

    Returns:

        IModArr (array): Stokes I model spectrum
        qArr (array): fractional Stokes Q spectrum
        uArr (array): fractional Stokes U spectrum
        dqArr (array): fractional Stokes Q spectrum errors
        duArr (array): fractional Stokes U spectrum errors
        fit_result (FitResult): fit information

    """
    if modStokesI is None:
        # Fit a <=5th order polynomial model to the Stokes I spectrum
        try:
            # The input values are forced to 64-bit in order to maximize the
            # stability and quality of the fits. It was found that the fitter
            # is more susceptible to numerical issues in 32-bit.
            fit_result = fit_StokesI_model(
                freqArr.astype("float64"),
                IArr.astype("float64"),
                dIArr.astype("float64"),
                polyOrd,
                fit_function,
            )
            IModArr = calculate_StokesI_model(fit_result, freqArr)

            if np.min(IModArr) < 0:  # Flag sources with negative models.
                fit_result = fit_result.with_options(
                    fitStatus=fit_result.fitStatus + 128
                )
            if (IModArr < dIArr).sum() > 0:  # Flag sources with models with S/N < 1.
                # TODO: this can be made better: estimating the error on the model to see
                # if the model is within 1 sigma, rather than the data error bars.
                fit_result = fit_result.with_options(
                    fitStatus=fit_result.fitStatus + 64
                )
        except Exception:  # If fit fails, fallback:
            print("Err: Failed to fit polynomial to Stokes I spectrum.")
            if debug:
                print("\nTRACEBACK:")
                print(("-" * 80))
                print((traceback.format_exc()))
                print(("-" * 80))
                print("\n")
            print("> Setting Stokes I spectrum to unity.\n")
            fit_result = FitResult(
                params=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                fitStatus=9,
                chiSq=0.0,
                chiSqRed=0.0,
                AIC=0,
                polyOrd=polyOrd,
                nIter=0,
                reference_frequency_Hz=1,
                dof=len(freqArr) - polyOrd - 1,
                pcov=np.zeros((6, 6)),
                perror=np.zeros(6),
                fit_function=fit_function,
            )
            IModArr = np.ones_like(IArr)
    else:
        # Use provided model
        IModArr = modStokesI
        fit_result = FitResult(
            params=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            fitStatus=0,
            chiSq=0.0,
            chiSqRed=0.0,
            AIC=0,
            polyOrd=polyOrd,
            nIter=0,
            reference_frequency_Hz=1,
            dof=len(freqArr) - polyOrd - 1,
            pcov=np.zeros((6, 6)),
            perror=np.zeros(6),
            fit_function=fit_function,
        )
        if verbose:
            print("Using provided model Stokes I spectrum")

    # Calculate the fractional spectra and errors
    with np.errstate(divide="ignore", invalid="ignore"):
        qArr = np.true_divide(QArr, IModArr)
        uArr = np.true_divide(UArr, IModArr)
        qArr = np.where(np.isfinite(qArr), qArr, np.nan)
        uArr = np.where(np.isfinite(uArr), uArr, np.nan)

        ## These errors only apply when dividing by channel Stokes I values, but
        ## not when dividing by a Stokes I model (the errors on which are more difficult
        ## to determine). Also assumes errors in Q,U are uncorrelated with errors in I,
        ## which I'm skeptical about. For now, replacing with what I think is a better
        ## approximation. I'm leaving this here in case we ever decide to implement
        ## channel-wise Stokes I normalization.
        # dqArr = np.abs(qArr) * np.sqrt( np.true_divide(dQArr, QArr)**2.0 +
        #                         np.true_divide(dIArr, IArr)**2.0 )
        # duArr = np.abs(uArr) * np.sqrt( np.true_divide(dUArr, UArr)**2.0 +
        #                         np.true_divide(dIArr, IArr)**2.0 )

        # Alternative scheme: assume errors in Stokes I don't propagate through
        # (i.e., that the model has no errors.)
        # TODO: if I do figure out model errors at some point, fold them in here.
        dqArr = dQArr / IModArr
        duArr = dUArr / IModArr
        dqArr = np.where(np.isfinite(dqArr), dqArr, np.nan)
        duArr = np.where(np.isfinite(duArr), duArr, np.nan)

    return IModArr, qArr, uArr, dqArr, duArr, fit_result


# Documenting fitStatus return values:
# 0  Improper input parameters.
# 1  Both actual and predicted relative reductions in the sum of squares
# 		   are at most ftol.
# 2  Relative error between two consecutive iterates is at most xtol
# 3  Conditions for status = 1 and status = 2 both hold.
# 4  The cosine of the angle between fvec and any column of the jacobian
# 		   is at most gtol in absolute value.
# 5  The maximum number of iterations has been reached.
# 6  ftol is too small. No further reduction in the sum of squares is
# 		   possible.
# 7  xtol is too small. No further improvement in the approximate solution
# 		   x is possible.
# 8  gtol is too small. fvec is orthogonal to the columns of the jacobian
# 		   to machine precision.
# 9 Fit failed; reason unknown (check log/terminal)
# 16 		   A parameter or function value has become infinite or an undefined
# 		   number.  This is usually a consequence of numerical overflow in the
# 		   user's model function, which must be avoided.
# The following can be added to the previous flags:
# 64 Model contains one or more channels with S:N < 1
# 128 Model contains negative Stokes I values.

# All flags greater the 80 indicate questionable/low-signal Stokes I models.
# All flags greater than 128 indicate bad Stokes I model (negative values)


# -----------------------------------------------------------------------------#
def interp_images(arr1, arr2, f=0.5):
    """Create an interpolated image between two other images."""

    nY, nX = arr1.shape

    # Concatenate arrays into a single array of shape (2, nY, nX)
    arr = np.r_["0,3", arr1, arr2]

    # Define the grid coordinates where you want to interpolate
    X, Y = np.meshgrid(np.arange(nX), np.arange(nY))

    # Create coordinates for interpolated frame
    coords = np.ones(arr1.shape) * f, Y, X

    # Interpolate using the map_coordinates function
    interpArr = ndi.map_coordinates(arr, coords, order=1)

    return interpArr


# -----------------------------------------------------------------------------#
def fit_spec_poly5(xData, yData, dyData=None, order=5, fit_function="log"):
    """Fit a 5th order polynomial to a spectrum. To avoid overflow errors the
    X-axis data should not be large numbers (e.g.: x10^9 Hz; use GHz
    instead)."""

    # Impose limits on polynomial order
    if order < 0:
        order = np.abs(order)
    if order > 5:
        order = 5
    if dyData is None:
        dyData = np.ones_like(yData)
    if np.all(dyData == 0):
        dyData = np.ones_like(yData)

    # Estimate starting coefficients
    C1 = 0.0
    C0 = np.nanmean(yData)
    C5 = 0.0
    C4 = 0.0
    C3 = 0.0
    C2 = 0.0
    inParms = [
        {"value": C5, "parname": "C5", "fixed": False},
        {"value": C4, "parname": "C4", "fixed": False},
        {"value": C3, "parname": "C3", "fixed": False},
        {"value": C2, "parname": "C2", "fixed": False},
        {"value": C1, "parname": "C1", "fixed": False},
        {"value": C0, "parname": "C0", "fixed": False},
    ]

    # Set the parameters as fixed of > order
    for i in range(len(inParms)):
        if len(inParms) - i - 1 > order:
            inParms[i]["fixed"] = True

    # Function to evaluate the difference between the model and data.
    # This is minimised in the least-squared sense by the fitter
    if fit_function == "linear":

        def errFn(p, fjac=None):
            status = 0
            return status, (poly5(p)(xData) - yData) / dyData

    elif fit_function == "log":

        def errFn(p, fjac=None):
            status = 0
            return status, (powerlaw_poly5(p)(xData) - yData) / dyData

    # Use MPFIT to perform the LM-minimisation
    mp = mpfit(errFn, parinfo=inParms, quiet=True)

    return mp


# -----------------------------------------------------------------------------#


def poly5(p):
    """Returns a function to evaluate a polynomial. The subfunction can be
    accessed via 'argument unpacking' like so: 'y = poly5(p)(*x)',
    where x is a vector of X values and p is a vector of coefficients."""

    # Fill out the vector to length 6 if necessary
    p = np.append(np.zeros((6 - len(p))), p)

    def rfunc(x):
        y = (
            p[0] * x**5.0
            + p[1] * x**4.0
            + p[2] * x**3.0
            + p[3] * x**2.0
            + p[4] * x
            + p[5]
        )
        return y

    return rfunc


def powerlaw_poly5(p):
    """Returns a function to evaluate a power law polynomial. The subfunction can be
    accessed via 'argument unpacking' like so: 'y = powerlaw_poly5(p)(*x)',
    where x is a vector of X values and p is a vector of coefficients."""

    # Fill out the vector to length 6 if necessary
    p = np.append(np.zeros((6 - len(p))), p)

    def rfunc(x):
        y = (
            p[0] * np.log10(x) ** 4.0
            + p[1] * np.log10(x) ** 3.0
            + p[2] * np.log10(x) ** 2.0
            + p[3] * np.log10(x)
            + p[4]
        )
        return p[5] * np.power(x, y)

    return rfunc


# -----------------------------------------------------------------------------#
def nanmedian(arr, **kwargs):
    """
    Returns median ignoring NaNs.
    """

    return ma.median(ma.masked_where(arr != arr, arr), **kwargs)


# -----------------------------------------------------------------------------#
def nanmean(arr, **kwargs):
    """
    Returns mean ignoring NaNs.
    """

    return ma.mean(ma.masked_where(arr != arr, arr), **kwargs)


# -----------------------------------------------------------------------------#
def nanstd(arr, **kwargs):
    """
    Returns standard deviation ignoring NaNs.
    """

    return ma.std(ma.masked_where(arr != arr, arr), **kwargs)


# -----------------------------------------------------------------------------#
def extrap(x, xp, yp):
    """
    Wrapper to allow np.interp to linearly extrapolate at function ends.

    np.interp function with linear extrapolation
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate
    -give-a-an-extrapolated-result-beyond-the-input-ran
    """

    y = np.interp(x, xp, yp)
    y = np.where(x < xp[0], yp[0] + (x - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1]), y)
    y = np.where(
        x > xp[-1], yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]), y
    )
    return y


# -----------------------------------------------------------------------------#
def toscalar(a):
    """
    Returns a scalar version of a Numpy object.
    """
    try:
        return a.item()
    except Exception:
        return a


# -----------------------------------------------------------------------------#
def MAD(a, c=0.6745, axis=None):
    """
    Median Absolute Deviation along given axis of an array:
    median(abs(a - median(a))) / c
    c = 0.6745 is the constant to convert from MAD to std
    """

    a = ma.masked_where(a != a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = ma.median(ma.fabs(a - d) / c)
    else:
        d = ma.median(a, axis=axis)
        if axis > 0:
            aswp = ma.swapaxes(a, 0, axis)
        else:
            aswp = a
        m = ma.median(ma.fabs(aswp - d) / c, axis=0)

    return m


# -----------------------------------------------------------------------------#
def calc_stats(a, maskzero=False):
    """
    Calculate the statistics of an array.
    """

    statsDict = {}
    a = np.array(a)

    # Mask off bad values and count valid pixels
    if maskzero:
        a = np.where(np.equal(a, 0.0), np.nan, a)
    am = ma.masked_invalid(a)
    statsDict["npix"] = np.sum(~am.mask)

    if statsDict["npix"] >= 2:
        statsDict["stdev"] = float(np.std(am))
        statsDict["mean"] = float(np.mean(am))
        statsDict["median"] = float(nanmedian(am))
        statsDict["max"] = float(np.max(am))
        statsDict["min"] = float(np.min(am))
        statsDict["centmax"] = list(np.unravel_index(np.argmax(am), a.shape))
        statsDict["madfm"] = float(MAD(am.flatten()))
        statsDict["success"] = True

    else:
        statsDict["npix"] == 0
        statsDict["stdev"] = 0.0
        statsDict["mean"] = 0.0
        statsDict["median"] = 0.0
        statsDict["max"] = 0.0
        statsDict["min"] = 0.0
        statsDict["centmax"] = (0.0, 0.0)
        statsDict["madfm"] = 0.0
        statsDict["success"] = False

    return statsDict


# -----------------------------------------------------------------------------#
@deprecated(
    deprecated_in="1.3.1",
    removed_in="1.4",
    current_version=__version__,
    details="This function is not used anywhere in current RM-Tools.",
)
def sort_nicely(l):
    """
    Sort a list in the order a human would.
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    l.sort(key=alphanum_key)


# -----------------------------------------------------------------------------#
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

    assert len(shape) == 2
    amp, xo, yo, cx, cy, pa = params
    y, x = np.indices(shape)
    st = m.sin(pa) ** 2
    ct = m.cos(pa) ** 2
    s2t = m.sin(2 * pa)
    a = (ct / cx**2 + st / cy**2) / 2
    b = s2t / 4 * (1 / cy**2 - 1 / cx**2)
    c = (st / cx**2 + ct / cy**2) / 2
    v = amp * np.exp(
        -1 * (a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)
    )

    return v


# -----------------------------------------------------------------------------#
def create_pqu_spectra_burn(
    freqArr_Hz, fracPolArr, psi0Arr_deg, RMArr_radm2, sigmaRMArr_radm2=None
):
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
    lamArr_m = speed_of_light.value / freqArr_Hz
    lamSqArr_m2 = np.power(lamArr_m, 2.0)

    # Convert the inputs to column vectors
    fracPolArr = fracPolArr.reshape((nComps, 1))
    psi0Arr_deg = psi0Arr_deg.reshape((nComps, 1))
    RMArr_radm2 = RMArr_radm2.reshape((nComps, 1))
    sigmaRMArr_radm2 = sigmaRMArr_radm2.reshape((nComps, 1))

    # Calculate the p, q and u Spectra for all components
    pArr = fracPolArr * np.ones((nComps, nChans), dtype="f8")
    quArr = pArr * (
        np.exp(2j * (np.radians(psi0Arr_deg) + RMArr_radm2 * lamSqArr_m2))
        * np.exp(-2.0 * sigmaRMArr_radm2 * np.power(lamArr_m, 4.0))
    )

    # Sum along the component axis to create the final spectra
    quArr = quArr.sum(0)
    qArr = quArr.real
    uArr = quArr.imag
    pArr = np.abs(quArr)

    return pArr, qArr, uArr


# -----------------------------------------------------------------------------#
def create_IQU_spectra_burn(
    freqArr_Hz,
    fluxI,
    SI,
    fracPolArr,
    psi0Arr_deg,
    RMArr_radm2,
    sigmaRMArr_radm2=None,
    freq0_Hz=None,
):
    """Create Stokes I, Q & U spectra for a source with 1 or more polarised
    Faraday components affected by external (burn) depolarisation."""

    # Create the polarised fraction spectra
    pArr, qArr, uArr = create_pqu_spectra_burn(
        freqArr_Hz, fracPolArr, psi0Arr_deg, RMArr_radm2, sigmaRMArr_radm2
    )

    # Default reference frequency is first channel
    if freq0_Hz is None:
        freq0_Hz = freqArr_Hz[0]

    # Create the absolute value spectra
    IArr = fluxI * np.power(freqArr_Hz / freq0_Hz, SI)
    PArr = IArr * pArr
    QArr = IArr * qArr
    UArr = IArr * uArr

    return IArr, QArr, UArr


# -----------------------------------------------------------------------------#
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
    lamArr_m = speed_of_light.value / freqArr_Hz
    lamSqArr_m2 = np.power(lamArr_m, 2.0)

    # Convert the inputs to column vectors
    fracPolArr = fracPolArr.reshape((nComps, 1))
    psi0Arr_deg = psi0Arr_deg.reshape((nComps, 1))
    RMArr_radm2 = RMArr_radm2.reshape((nComps, 1))

    # Calculate the p, q and u Spectra for all components
    RMLamSqArr = RMArr_radm2 * lamSqArr_m2
    pArr = fracPolArr * np.sinc(RMLamSqArr / np.pi)
    pArr = pArr.astype("complex")
    for i in range(nComps):
        RMLamSqArr[i] *= 0.5
        pArr[i] *= np.exp(2j * (psi0Arr_rad[i] + RMLamSqArr[i:].sum(0)))

    # Sum along the component axis to create the final spectra
    pArr = pArr.sum(0)
    qArr = pArr.real
    uArr = pArr.imag
    pArr = np.abs(pArr)

    return pArr, qArr, uArr


# -----------------------------------------------------------------------------#
def create_IQU_spectra_diff(
    freqArr_Hz, fluxI, SI, fracPolArr, psi0Arr_deg, RMArr_radm2, freq0_Hz=None
):
    """Create Stokes I, Q & U spectra for a source with 1 or more polarised
    Faraday components affected by internal Faraday depolarisation"""

    # Create the polarised fraction spectra
    pArr, qArr, uArr = create_pqu_spectra_diff(
        freqArr_Hz, fracPolArr, psi0Arr_deg, RMArr_radm2
    )

    # Default reference frequency is first channel
    if freq0_Hz is None:
        freq0_Hz = freqArr_Hz[0]

    # Create the absolute value spectra
    IArr = fluxI * np.power(freqArr_Hz / freq0_Hz, SI)
    PArr = IArr * pArr
    QArr = IArr * qArr
    UArr = IArr * uArr

    return IArr, QArr, UArr


# -----------------------------------------------------------------------------#
def create_pqu_spectra_RMthin(freqArr_Hz, fracPol, psi0_deg, RM_radm2):
    """Return fractional P/I, Q/I & U/I spectra for a Faraday thin source"""

    # Calculate the p, q and u Spectra
    lamSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)
    pArr = fracPol * np.ones_like(lamSqArr_m2)
    quArr = pArr * np.exp(2j * (np.radians(psi0_deg) + RM_radm2 * lamSqArr_m2))
    qArr = quArr.real
    uArr = quArr.imag

    return pArr, qArr, uArr


# -----------------------------------------------------------------------------#
def create_IQU_spectra_RMthin(
    freqArr_Hz, fluxI, SI, fracPol, psi0_deg, RM_radm2, freq0_Hz=None
):
    """Return Stokes I, Q & U spectra for a Faraday thin source"""

    pArr, qArr, uArr = create_pqu_spectra_RMthin(
        freqArr_Hz, fracPol, psi0_deg, RM_radm2
    )
    if freq0_Hz is None:
        freq0_Hz = freqArr_Hz[0]
    IArr = fluxI * np.power(freqArr_Hz / freq0_Hz, SI)
    PArr = IArr * pArr
    QArr = IArr * qArr
    UArr = IArr * uArr

    return IArr, QArr, UArr


# -----------------------------------------------------------------------------#
def create_pqu_resid_RMthin(qArr, uArr, freqArr_Hz, fracPol, psi0_deg, RM_radm2):
    """Subtract a RM-thin component from the fractional q and u data."""

    pModArr, qModArr, uModArr = create_pqu_spectra_RMthin(
        freqArr_Hz, fracPol, psi0_deg, RM_radm2
    )
    qResidArr = qArr - qModArr
    uResidArr = uArr - uModArr
    pResidArr = np.sqrt(qResidArr**2.0 + uResidArr**2.0)

    return pResidArr, qResidArr, uResidArr


# -----------------------------------------------------------------------------#
def xfloat(x, default=None):
    if x is None or x == "":
        return default
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------------------------------------------------------#
def norm_cdf(mean=0.0, std=1.0, N=50, xArr=None):
    """Return the CDF of a normal distribution between -6 and 6 sigma, or at
    the values of an input array."""

    if xArr is None:
        x = np.linspace(-6.0 * std, 6.0 * std, N)
    else:
        x = xArr
    y = norm.cdf(x, loc=mean, scale=std)

    return x, y
