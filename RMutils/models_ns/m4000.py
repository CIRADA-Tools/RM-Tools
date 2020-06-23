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

C = 2.997924538e8 # Speed of light [m/s]

def model(pDict, lamSqArr_m2):
    """Simple Faraday thin source + cable delay + I->V leakage"""

    freqArr=C/np.sqrt(lamSqArr_m2)
    
    IArr = (pDict["I_amp"] * np.ones_like(freqArr)) * (freqArr/freqArr.min())**(pDict["gamma_I"])

    # Calculate the complex fractional q and u spectra
    pArr = pDict["fracPol"] * IArr
    # model Faraday rotation
    QUArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                 pDict["RM_radm2"] * lamSqArr_m2) )
    QArr = QUArr.real
    UArr = QUArr.imag
    
    # create model v spectrum
    VArr = np.zeros_like(lamSqArr_m2)

    # model cable delay leakage
    U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s"])*VArr
    V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*VArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*UArr
    UArr=U_leak
    VArr=-V_leak
    
    IArr_leak = 0.5*IArr*(pDict['gain_X']**2+pDict['gain_Y']**2)+0.5 \
    *QArr*(pDict['gain_X']**2-pDict['gain_Y']**2)
    
    QArr_leak = 0.5*IArr*(pDict['gain_X']**2-pDict['gain_Y']**2)+0.5 \
    *QArr*(pDict['gain_X']**2+pDict['gain_Y']**2)
    
    IArr = IArr_leak
    QArr = QArr_leak
    
    UArr = UArr*pDict['gain_X']*pDict['gain_Y']
    VArr = VArr*pDict['gain_X']*pDict['gain_Y']
    
    QUArr = QArr + 1j*UArr
   
    return QUArr, VArr, IArr


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

    {"parname":   "I_amp",
     "label":     "I amp.",
     "value":     0.0,
     "bounds":    [0, 1e8],
     "priortype": "uniform",
     "wrap":      0},

    {"parname":   "gamma_I",
     "label":     "$\gamma_I$",
     "value":     0.0,
     "bounds":    [-3.0, 3.0],
     "priortype": "uniform",
     "wrap":      0},

    {"parname":   "gain_X",
     "label":     "gain X",
     "value":     0.0,
     "bounds":    [0.01, 1.2],
     "priortype": "uniform",
     "wrap":      0},

    {"parname":   "gain_Y",
     "label":     "gain Y",
     "value":     0.0,
     "bounds":    [0.01, 1.2],
     "priortype": "uniform",
     "wrap":      0}
    
]



#-----------------------------------------------------------------------------#
# Arguments controlling the Nested Sampling algorithm                         #
#-----------------------------------------------------------------------------#
nestArgsDict = {"n_live_points": 1000,
                "verbose": False}
