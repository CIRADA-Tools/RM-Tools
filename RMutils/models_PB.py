#=============================================================================#
#                          MODEL DEFINITION FILE                              #
#=============================================================================#
import numpy as np
C = 2.997924538e8 # Speed of light [m/s]

#-----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms, below. #
#  lamSqArr_m2 = Array of lambda-squared values
#  IModArr     = Model of Stokes I spectrum                                   #
#  quArr, vArr = Complex array containing the Re (Q) and Im (U) spectra and stokes V. #
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Parameters for model.                                             #
#                                                                             #
# Each parameter is defined by a dictionary with the following keywords:      #
#   parname    ...   parameter name used in the model function above          #
#   label      ...   latex style label used by plotting functions             #
#   value      ...   value of the parameter if priortype = "fixed"            #
#   bounds     ...   [low, high] limits of the prior                          #
#   priortype  ...   "uniform", "normal", "log" or "fixed"                    #
#   wrap       ...   set > 0 for periodic parameters (e.g., for an angle)     #
#-----------------------------------------------------------------------------#

def get_model(name):

    if (name == None) or (name == 'auto'):

        def model(pDict, lamSqArr_m2, gains):
            """Simple Faraday thin source + cable delay + differential response of pb"""

            freqArr=C/np.sqrt(lamSqArr_m2)

            IArr = pDict["c0"]+pDict["c1"]*freqArr+pDict["c2"]*freqArr**2+pDict["c3"]*freqArr**3
            # differential X,Y response from PB
            gain_X = gains[0]
            gain_Y = gains[1]

            # Calculate the complex fractional Q and U spectra
            pArr = pDict["fracPol"] * IArr

            # model Faraday rotation
            QUArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                 pDict["RM_radm2"] * lamSqArr_m2) )

            QArr = QUArr.real
            UArr = QUArr.imag

            # model v spectrum (change this to non-zero array to model instrinsic stokes V)
            VArr = np.zeros_like(lamSqArr_m2)

            # model cable delay leakage
            U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s"])*VArr
            V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*VArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*UArr
            UArr=U_leak
            VArr=-V_leak

            # model differential X,Y response (see Johnston 2002 for details)
            IArr_leak = 0.5*IArr*(gain_X**2+gain_Y**2)+0.5*QArr*(gain_X**2-gain_Y**2)
            QArr_leak = 0.5*IArr*(gain_X**2-gain_Y**2)+0.5*QArr*(gain_X**2+gain_Y**2)
            IArr = IArr_leak
            QArr = QArr_leak
            UArr = UArr*gain_X*gain_Y
            VArr = VArr*gain_X*gain_Y

            QUArr = QArr + 1j*UArr

            return QUArr, VArr, IArr

        return model

    if (name == 'new'):

        def model(pDict, lamSqArr_m2, gains):
            """Simple Faraday thin source + cable delay + differential response of pb"""

            freqArr=C/np.sqrt(lamSqArr_m2)

            IArr = pDict["I_amp"]*(freqArr/freqArr.min())**(pDict["I_alpha"]+pDict["I_beta"]*np.log(freqArr/freqArr.min()))
            # differential X,Y response from PB
            gain_X = gains[0]
            gain_Y = gains[1]

            # Calculate the complex fractional Q and U spectra
            pArr = pDict["fracPol"] * IArr

            # model Faraday rotation
            QUArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                 pDict["RM_radm2"] * lamSqArr_m2) )

            QArr = QUArr.real
            UArr = QUArr.imag

            # model v spectrum (change this to non-zero array to model instrinsic stokes V)
            VArr = np.zeros_like(lamSqArr_m2)

            # model cable delay leakage
            U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s"])*VArr
            V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*VArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*UArr
            UArr=U_leak
            VArr=-V_leak

            # model differential X,Y response (see Johnston 2002 for details)
            IArr_leak = 0.5*IArr*(gain_X**2+gain_Y**2)+0.5*QArr*(gain_X**2-gain_Y**2)
            QArr_leak = 0.5*IArr*(gain_X**2-gain_Y**2)+0.5*QArr*(gain_X**2+gain_Y**2)
            IArr = IArr_leak
            QArr = QArr_leak
            UArr = UArr*gain_X*gain_Y
            VArr = VArr*gain_X*gain_Y

            QUArr = QArr + 1j*UArr

            return QUArr, VArr, IArr

        return model

def get_params(name):

    if (name == None) or (name == 'auto'):

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
             
            {"parname":   "c0",
             "label":     "$c_0$",
             "value":     1e3,
             "bounds":    [0.01, 1e6],
             "priortype": "uniform",
             "wrap":      0},                          

            {"parname":   "c1",
             "label":     "$c_1$",
             "value":     0,
             "bounds":    [0.01, 1e6],
             "priortype": "uniform",
             "wrap":      0},
             
            {"parname":   "c2",
             "label":     "$c_2$",
             "value":     0,
             "bounds":    [0.01, 1e6],
             "priortype": "uniform",
             "wrap":      0},
                          
            {"parname":   "c3",
             "label":     "$c_3$",
             "value":     0,
             "bounds":    [0.01, 1e6],
             "priortype": "uniform",
             "wrap":      0},
             
        ]
    
    if (name == 'new'):

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
             "label":     "$c_0$",
             "value":     1e5,
             "bounds":    [1e3, 1e10],
             "priortype": "uniform",
             "wrap":      0},

            {"parname":   "I_alpha",
             "label":     "$alpha$",
             "value":     0,
             "bounds":    [-1000, 1000],
             "priortype": "uniform",
             "wrap":      0},
             
            {"parname":   "I_beta",
             "label":     "$beta$",
             "value":     0,
             "bounds":    [-1000, 1000],
             "priortype": "uniform",
             "wrap":      0}
             
        ]

    return inParms

#-----------------------------------------------------------------------------#
# Arguments controlling the Nested Sampling algorithm                         #
#-----------------------------------------------------------------------------#
nestArgsDict = {"n_live_points": 1000,
                "verbose": False}


