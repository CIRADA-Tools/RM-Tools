#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an experimental tool to generate Stokes Q and U models from
clean components produced by RMclean1D.


Author: cvaneck, Aug 2021
"""

import json

import numpy as np
from astropy.constants import c as speed_of_light

from RMtools_1D.do_RMsynth_1D import readFile as read_freqFile
from RMutils.util_misc import FitResult, calculate_StokesI_model


def calculate_QU_model(freqArr, phiArr, CCArr, lambdaSq_0, Iparms=None):
    """Compute the predicted Stokes Q and U values for each channel from a
    set of clean components (CCs), with optional accounting for Stokes I model.
    Inputs: freqArr: array of channel frequencies, in Hz
            phiArr: array of Faraday depth values for the clean component array
            CCarr: array of (complex) clean components.
            lambdaSq_0: scalar value of the reference wavelength squared to
                        which all the polarization angles are referenced.
            Iparms: list of Stokes I polynomial values. If None, all Stokes I
                    values will be set to 1.
    Returns:
        model: array of complex values, one per channel, of Stokes Q and U
                predictions based on the clean component model.

    CURRENTLY ASSUMES THAT STOKES I MODEL IS LOG MODEL. SHOULD BE FIXED!
    """

    lambdaSqArr_m2 = np.power(speed_of_light.value / freqArr, 2.0)

    a = lambdaSqArr_m2 - lambdaSq_0
    quarr = np.sum(CCArr[:, np.newaxis] * np.exp(2.0j * np.outer(phiArr, a)), axis=0)

    # TODO: Pass in fit function, which is currently not output by rmsynth1d
    fit_result = FitResult(
        params=Iparms if Iparms is not None else [0, 0, 0, 0, 0, 1],
        fitStatus=0,
        chiSq=0.0,
        chiSqRed=0.0,
        AIC=0,
        polyOrd=0,
        nIter=0,
        reference_frequency_Hz=speed_of_light.value / np.sqrt(lambdaSq_0),
        dof=0,
        pcov=None,
        perror=None,
        fit_function="log",
    )
    StokesI_model = calculate_StokesI_model(fit_result, freqArr)

    QUarr = StokesI_model * quarr

    return QUarr, StokesI_model


def save_model(filename, freqArr, Imodel, QUarr):
    np.savetxt(filename, list(zip(freqArr, Imodel, QUarr.real, QUarr.imag)))


def read_files(freqfile, rmSynthfile, CCfile):
    """Get necessary data from the RMsynth and RMclean files. These data are:
    * The array of channel frequencies, from the RMsynth input file.
    * The phi array and clean components, from the RMclean1D _FDFmodel.dat file
    * The Stokes I model and lambda^2_0 value, from the RMsynth1D  _RMsynth.json file.

    Inputs: freqfile (str): filename containing frequencies
            rmSynthfile (str): filename of RMsynth JSON output.
            CCfile (str): filename of clean component model file (_FDFmodel.dat)/

    Returns: phiArr: array of Faraday depth values for CC spectrum
            CCarr: array of (complex) clean components
            Iparms: list of Stokes I model parameters.
            lambdaSq_0: scalar value of lambda^2_0, in m^2.
    """

    phiArr, CCreal, CCimag = np.loadtxt(CCfile, unpack=True, dtype="float")
    CCArr = CCreal + 1j * CCimag

    # TODO: change filename to JSON if needed?
    synth_mDict = json.load(open(rmSynthfile, "r"))
    Iparms = [float(x) for x in synth_mDict["polyCoeffs"].split(",")]
    lambdaSq_0 = synth_mDict["lam0Sq_m2"]

    data = read_freqFile(freqfile, 64, verbose=False, debug=False)
    freqArr = data[0]

    return phiArr, CCArr, Iparms, lambdaSq_0, freqArr


def main():
    """Generate Stokes QU model based on clean components and (optional)
    Stokes I model. Requires inputs to rmsynth1D and outputs of rmsynth1d and
    rmclean1d.
    """
    import argparse

    descStr = """
    Generate Stokes QU model based on clean components and (optional)
    Stokes I model. Requires inputs to rmsynth1D and outputs of rmsynth1d and
    rmclean1d. Saves ASCII file containing arrays of IQU for each channel.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "freqfile",
        metavar="input.dat",
        nargs=1,
        help="ASCII file containing original frequency spectra.",
    )
    parser.add_argument(
        "rmSynthfile",
        metavar="_RMsynth.json",
        nargs=1,
        help="RMsynth1d output JSON file.",
    )
    parser.add_argument(
        "CCfile",
        metavar="_FDFmodel.dat",
        nargs=1,
        help="Clean component model file (_FDFmodel.dat)",
    )
    parser.add_argument(
        "outfile",
        metavar="QUmodel.dat",
        nargs=1,
        help="Filename to save output model to.",
    )
    args = parser.parse_args()

    phiArr, CCArr, Iparms, lambdaSq_0, freqArr = read_files(
        args.freqfile, args.rmSynthfile, args.CCfile
    )
    QUarr, Imodel = calculate_QU_model(freqArr, phiArr, CCArr, lambdaSq_0, Iparms)

    save_model(args.outfile, freqArr, Imodel, QUarr)


if __name__ == "__main__":
    main()
