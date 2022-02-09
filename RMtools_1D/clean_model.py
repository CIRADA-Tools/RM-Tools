#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an experimental tool to generate Stokes Q and U models from
clean components produced by RMclean1D.


Author: cvaneck, Aug 2021
"""

import numpy as np
from RMtools_1D.do_RMsynth_1D import readFile as read_freqFile
from RMutils.util_misc import create_frac_spectra
from RMutils.util_misc import calculate_StokesI_model
from RMutils.util_misc import toscalar
from RMutils.util_plotTk import plot_Ipqu_spectra_fig
from RMutils.util_misc import calculate_StokesI_model
import json
import matplotlib.pyplot as plt

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

    C = 2.997924538e8 # Speed of light [m/s]
    lambdaSqArr_m2 = np.power(C/freqArr, 2.0)


    a = lambdaSqArr_m2 - lambdaSq_0
    quarr = np.sum(CCArr[:,np.newaxis]*np.exp(2.0j * np.outer(phiArr,a)),axis=0)

    fitDict={}
    #TODO: Pass in fit function, which is currently not output by rmsynth1d
    fitDict['fit_function']= 'log'
    if Iparms is not None:
        fitDict["p"]=Iparms
    else:
        fitDict["p"]=[0,0,0,0,0,1]
    fitDict['reference_frequency_Hz']=C/np.sqrt(lambdaSq_0)
    StokesI_model=calculate_StokesI_model(fitDict,freqArr)

    QUarr=StokesI_model*quarr

    return QUarr,StokesI_model


def save_model(filename,freqArr,Imodel,QUarr):
    np.savetxt(filename, list(zip(freqArr, Imodel,QUarr.real, QUarr.imag)))



def read_files(freqfile,rmSynthfile, CCfile):
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
    phiArr, CCreal, CCimag = np.loadtxt(CCfile, unpack=True, dtype='float')
    CCArr=CCreal + 1j * CCimag
    
    #TODO: change filename to JSON if needed?
    synth_mDict = json.load(open(rmSynthfile, "r"))
    Iparms=[float(x) for x in synth_mDict['polyCoeffs'].split(',')]
    lambdaSq_0=synth_mDict['lam0Sq_m2']
    
    # Parse the data array
    # freq_Hz, I, Q, U, dI, dQ, dU
    data=read_freqFile(freqfile, 64, verbose=False, debug=False)
    freqArr_Hz=data[0]

    return phiArr, CCArr, Iparms, lambdaSq_0,freqArr_Hz


def plot_model(freqfile, QUarr,Imodel):
    # Parse the data array
    # freq_Hz, I, Q, U, dI, dQ, dU
    data=read_freqFile(freqfile, 64, verbose=False, debug=False)
    try:
        (freqArr_Hz, IArr, QArr, UArr,
         dIArr, dQArr, dUArr) = data
        print("\nFormat [freq_Hz, I, Q, U, dI, dQ, dU]")
    except Exception:
        # freq_Hz, Q, U, dQ, dU
        try:
            (freqArr_Hz, QArr, UArr, dQArr, dUArr) = data

            print("\nFormat [freq_Hz, Q, U,  dQ, dU]")
            noStokesI = True
        except Exception:
            print("\nError: Failed to parse data file!")
    dataArr = create_frac_spectra(freqArr=freqArr_Hz,
                                    IArr=IArr,
                                    QArr=QArr,
                                    UArr=UArr,
                                    dIArr=dIArr,
                                    dQArr=dQArr,
                                    dUArr=dUArr,
                                    polyOrd=2,
                                    verbose=True,
                                    fit_function='log')
    (IModArr, qArr, uArr, dqArr, duArr, IfitDict) = dataArr

    freqHirArr_Hz =  np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)

    IModHirArr = calculate_StokesI_model(IfitDict, freqHirArr_Hz)

    specFig = plt.figure(facecolor='w',figsize=(10, 6))
    plot_Ipqu_spectra_fig(freqArr_Hz     = freqArr_Hz,
                            IArr           = IArr,
                            qArr           = qArr,
                            uArr           = uArr,
                            dIArr          = dIArr,
                            dqArr          = dqArr,
                            duArr          = duArr,
                            freqHirArr_Hz  = freqArr_Hz,
                            IModArr        = Imodel,
                            qModArr        = QUarr.real,
                            uModArr        = QUarr.imag,
                            fig            = specFig)
    plt.show()

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
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("freqfile", metavar="input.dat",
                        help="ASCII file containing original frequency spectra.")
    parser.add_argument("rmSynthfile",metavar="_RMsynth.json",
                        help="RMsynth1d output JSON file.")
    parser.add_argument("CCfile",metavar="_FDFmodel.dat",
                        help="Clean component model file (_FDFmodel.dat)")
    parser.add_argument("outfile",metavar="QUmodel.dat",
                        help="Filename to save output model to.")
    args = parser.parse_args()


    phiArr, CCArr, Iparms, lambdaSq_0,freqArr=read_files(args.freqfile,
                                                         args.rmSynthfile,
                                                         args.CCfile)
    QUarr,Imodel=calculate_QU_model(freqArr, phiArr, CCArr, lambdaSq_0, Iparms)

    plot_model(args.freqfile, QUarr,Imodel)
    
    save_model(args.outfile,freqArr,Imodel,QUarr)



if __name__ == "__main__":
    main()



