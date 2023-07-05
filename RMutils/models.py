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
#  IModArr     = Model of Stokes I ("total intensity") spectrum                                   #
#  QUArr, VArr = Complex array containing the Re (Q) and Im (U) spectra and Stokes V. #
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

    if (name == 'cable_delay'):

        def model(pDict, lamSqArr_m2, IModArr):
            """Simple Faraday thin source + differential phase between X,Y polarizations (i.e. U->V leakage)"""

            IModArr[IModArr<0] =  np.nan
            freqArr=C/np.sqrt(lamSqArr_m2)

            # Calculate linear polarization w/ Stokes I model
            pArr = pDict["fracPol"] * IModArr
            # Model Faraday rotation
            QUArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                         pDict["RM_radm2"] * lamSqArr_m2) )
            
            # Create model V spectrum (change this to non-zero array to model instrinsic stokes V)
            VModArr = np.zeros_like(lamSqArr_m2)

            QArr = QUArr.real
            UArr = QUArr.imag

            # Model differential X,Y phase leakage
            U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*VModArr
            V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*VModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*UArr
            UArr=U_leak
            VArr=-V_leak
            
            QUArr = QArr + 1j*UArr

            return QUArr, VArr

        return model
        
    if (name == 'cable_delay+response'):

        def model(pDict, lamSqArr_m2, IModArr):
            """Same as 'cable_delay' model + differential response between X,Y polarizations (i.e. I->Q leakage)"""

            IModArr[IModArr<0] =  np.nan
            IArr = IModArr.copy()

            freqArr=C/np.sqrt(lamSqArr_m2)

            # Calculate linear polarization w/ Stokes I model
            pArr = pDict["fracPol"] * IModArr

            # Model differential X,Y response
            gain_X = 1
            gain_Y = gain_X * pDict['gain_diff']
        
            # Model Faraday rotation
            QUArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                 pDict["RM_radm2"] * lamSqArr_m2) )

            QArr = QUArr.real
            UArr = QUArr.imag

            # model V spectrum (change this to non-zero array to model instrinsic stokes V)
            VArr = np.zeros_like(lamSqArr_m2)

            # Model differential X,Y phase leakage
            U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*VModArr
            V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*VModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*UArr
            UArr=U_leak
            VArr=-V_leak

            # Model differential X,Y response (see Johnston 2006 for details)
            IArr_leak = 0.5*IArr*(gain_X**2+gain_Y**2)+0.5*QArr*(gain_X**2-gain_Y**2)
            QArr_leak = 0.5*IArr*(gain_X**2-gain_Y**2)+0.5*QArr*(gain_X**2+gain_Y**2)
            IArr = IArr_leak
            QArr = QArr_leak
            UArr = UArr*gain_X*gain_Y
            VArr = VArr*gain_X*gain_Y

            QUArr = QArr + 1j*UArr

            return QUArr, VArr

        return model

    if (name == 'full_fit'):
        
        def model(pDict, lamSqArr_m2,  IModArr):
            """Model incoporating params for systematics (e.g., differential phase & response bewteen X,Y) and non-zero/frequency dependent linear & circular polarization fraction"""
            
            IModArr[IModArr<0] =  np.nan
            IArr = IModArr.copy()

            freqArr=C/np.sqrt(lamSqArr_m2)

            # Model fractional linear polarization (power-law)
            pfracArr = (pDict["fracPol"] * np.ones_like(freqArr)) * (freqArr/400e6)**(pDict["gamma"])
            # Calculate linear polarization w/ Stokes I model
            pArr = pfracArr * IModArr
            # Create model V spectrum (power-law)
            vfracArr = (pDict["fracPol_V"] * np.ones_like(lamSqArr_m2)) * (freqArr/400e6)**(pDict["gamma_V"])
            VModArr = vfracArr * IModArr
            # Model differential X,Y response
            gain_X = 1
            gain_Y = gain_X * pDict['gain_diff']
            # Model Faraday rotation
            QUArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) + pDict["RM_radm2"] * lamSqArr_m2) )

            QArr = QUArr.real
            UArr = QUArr.imag

            # Model differential X,Y phase leakage
            U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*VModArr
            V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*VModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"] + np.radians(pDict["lag_phi"]))*UArr
            UArr=U_leak
            VArr=-V_leak
            
            # Model differential X,Y response (see Johnston 2006 for details)
            IArr_leak = 0.5*IArr*(gain_X**2+gain_Y**2)+0.5*QArr*(gain_X**2-gain_Y**2)
            QArr_leak = 0.5*IArr*(gain_X**2-gain_Y**2)+0.5*QArr*(gain_X**2+gain_Y**2)
            IArr = IArr_leak
            QArr = QArr_leak
            UArr = UArr*gain_X*gain_Y
            VArr = VArr*gain_X*gain_Y
    
            QUArr = QArr + 1j*UArr
   
            return QUArr, VArr

        return model

def get_params(name):
        
    if (name == 'cable_delay'):

        inParms = [
            {"parname":   "fracPol",
             "label":     "$p$",
             "value":     0.1,
             "bounds":    [0.001, 1.1],
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
             "bounds":    [-5000.0, 5000.0],
             "priortype": "uniform",
             "wrap":      0},

            {"parname":   "lag_s",
             "label":     "lag (sec)",
             "value":     0.0,
             "bounds":    [-1e-8, 1e-8],
             "priortype": "uniform",
             "wrap":      0},
             
            {"parname":   "lag_phi",
             "label":     "lag_phi (deg.)",
             "value":     0.0,
             "bounds":    [0.0, 360.0],
             "priortype": "uniform",
             "wrap":      1}
        ]

    if (name == 'cable_delay+response'):

        inParms = [
            {"parname":   "fracPol",
             "label":     "$p$",
             "value":     0.1,
             "bounds":    [0.001, 1.1],
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
             "bounds":    [-5000.0, 5000.0],
             "priortype": "uniform",
             "wrap":      0},

            {"parname":   "lag_s",
             "label":     "lag (sec)",
             "value":     0.0,
             "bounds":    [-1e-8, 1e-8],
             "priortype": "uniform",
             "wrap":      0},
             
            {"parname":   "lag_phi",
             "label":     "lag_phi (deg.)",
             "value":     0.0,
             "bounds":    [0.0, 360.0],
             "priortype": "uniform",
             "wrap":      1},

            {"parname":   "gain_diff",
             "label":     "gain diff",
             "value":     1.0,
             "bounds":    [0.1, 10.0],
             "priortype": "uniform",
             "wrap":      0},

        ]

    if (name == 'full_fit'):

        inParms = [
            {"parname":   "fracPol",
             "label":     "$p$",
             "value":     0.1,
             "bounds":    [0.001, 1.1],
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
             "bounds":    [-5000.0, 5000.0],
             "priortype": "uniform",
             "wrap":      0},

            {"parname":   "lag_s",
             "label":     "lag (sec)",
             "value":     0.0,
             "bounds":    [-1e-8, 1e-8],
             "priortype": "uniform",
             "wrap":      0},
             
            {"parname":   "lag_phi",
             "label":     "lag_phi (deg.)",
             "value":     0.0,
             "bounds":    [0.0, 360.0],
             "priortype": "uniform",
             "wrap":      1},

            {"parname":   "gamma",
             "label":     "$\gamma_L$",
             "value":     0.0,
             "bounds":    [-10.0, 10.0],
             "priortype": "uniform",
             "wrap":      0},

            {"parname":   "fracPol_V",
             "label":     "$p_V$",
             "value":     0.1,
             "bounds":    [-1.0, 1.0],
             "priortype": "uniform",
             "wrap":      0},

            {"parname":   "gamma_V",
             "label":     "$\gamma_V$",
             "value":     0.0,
             "bounds":    [-10.0, 10.0],
             "priortype": "uniform",
             "wrap":      0},

            {"parname":   "gain_diff",
             "label":     "gain diff",
             "value":     1.0,
             "bounds":    [0.1, 10.0],
             "priortype": "uniform",
             "wrap":      0}

        ]

    return inParms


#-----------------------------------------------------------------------------#
# Arguments controlling the Nested Sampling algorithm                         #
#-----------------------------------------------------------------------------#
nestArgsDict = {"n_live_points": 1000,
                "verbose": False}
