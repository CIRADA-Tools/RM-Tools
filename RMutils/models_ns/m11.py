#=============================================================================#
#                          MODEL DEFINITION FILE                              #
#=============================================================================#
import numpy as np
C = 2.997924538e8 # Speed of light [m/s]

#-----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms, below. #
#  lamSqArr_m2 = Array of lambda-squared values                               #
#  quArr       = Complex array containing the Re and Im spectra and stokes V.              #
#-----------------------------------------------------------------------------#
def model(pDict, lamSqArr_m2):
    """Simple Faraday thin source + cable delay + I->V leakage"""

    freqArr=C/np.sqrt(lamSqArr_m2)

    # Calculate the complex fractional q and u spectra
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)
    # model Faraday rotation
    quArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                 pDict["RM_radm2"] * lamSqArr_m2) )
    
    # create model v spectrum
    vModArr = (pDict["fracPol_V"] * np.ones_like(lamSqArr_m2)) * (freqArr/freqArr.min())**(pDict["gamma"])

    # model cable delay leakage
    u_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*quArr.imag - np.sin(2*np.pi*freqArr*pDict["lag_s"])*vModArr
    v_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*vModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*quArr.imag
    #quArr.imag=uvleak.real
    quArr.imag=u_leak
    #vArr=uvleak.imag
    vArr=-v_leak
   
    return quArr, vArr


#-----------------------------------------------------------------------------#
# Parameters for the above model.                                             #
#                                                                             #
# Each parameter is defined by a dictionary with the following keywords:      #
#   parname    ...   parameter name used in the model function above          #
#   label      ...   latex style label used by plotting functions             #
#   value      ...   value of the parameter if priortype = "fixed"            #
#   bounds     ...   [low, high] limits of the prior                          #
#   priortype  ...   "uniform", "normal", "log" or "fixed"                    #
#   wrap       ...   set > 0 for periodic parameters (e.g., for an angle)     #
#-----------------------------------------------------------------------------#
inParms = [
    {"parname":   "fracPol",
     "label":     "$p$",
     "value":     0.1,
     "bounds":    [0.001, 1.0],
     "priortype": "uniform",
     "wrap":      0},
    
    {"parname":   "psi0_deg",
     "label":     "$\psi_0$ (deg)",
     "value":     0.0,
     "bounds":    [0.0, 180.0],
     "priortype": "uniform",
     "wrap":      1},
    
    {"parname":   "RM_radm2",
     "label":     "RM (rad m$^{-2}$)",
     "value":     0.0,
     "bounds":    [-1100.0, 1100.0],
     "priortype": "uniform",
     "wrap":      0},

    {"parname":   "lag_s",
     "label":     "lag (sec)",
     "value":     0.0,
     "bounds":    [-1e-7, 1e-7],
     "priortype": "uniform",
     "wrap":      0},
   
    {"parname":   "fracPol_V",
     "label":     "$p_v$",
     "value":     0.0,
     "bounds":    [0.0, 1.0],
     "priortype": "uniform",
     "wrap":      0},
    
    {"parname":   "gamma",
     "label":     "$\gamma$",
     "value":     0.0,
     "bounds":    [-3.0, 3.0],
     "priortype": "uniform",
     "wrap":      0}
]


#-----------------------------------------------------------------------------#
# Arguments controlling the Nested Sampling algorithm                         #
#-----------------------------------------------------------------------------#
nestArgsDict = {"n_live_points": 1000,
                "verbose": False}
