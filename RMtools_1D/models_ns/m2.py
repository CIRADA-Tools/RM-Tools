#=============================================================================#
#                          MODEL DEFINITION FILE                              #
#=============================================================================#
import numpy as np


#-----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms, below. #
#  lamSqArr_m2 = Array of lambda-squared values                               #
#  quArr       = Complex array containing the Re and Im spectra.              #
#-----------------------------------------------------------------------------#
def model(pDict, lamSqArr_m2):
    """Single Faraday component with Burn depolarisation"""
    
    # Calculate the complex fractional q and u spectra
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)
    quArr = (pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                  pDict["RM_radm2"] * lamSqArr_m2))
             * np.exp(-2.0 * pDict["sigmaRM_radm2"]**2.0 
                      * lamSqArr_m2**2.0))
    
    return quArr


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
     "bounds":    [0.0000001, 1.0],
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
    
    {"parname":   "sigmaRM_radm2",
     "label":     "$\sigma_{RM}$ (rad m$^{-2}$)",
     "value":     0.0,
     "bounds":    [0.0, 1000.0],
     "priortype": "uniform",
     "wrap":      0}
]


#-----------------------------------------------------------------------------#
# Switches controlling the Nested Sampling algorithm                          #
#-----------------------------------------------------------------------------#
nestArgsDict = {"n_live_points": 1000,
                "verbose": False}
