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

    if (name == None) or (name == 'cable_delay'):

        def model(pDict, lamSqArr_m2, IModArr):
            """Simple Faraday thin source + cable delay + I->V leakage"""

            freqArr=C/np.sqrt(lamSqArr_m2)

            # Calculate the complex fractional q and u spectra
            pArr = pDict["fracPol"] * IModArr
            # model Faraday rotation
            quArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                         pDict["RM_radm2"] * lamSqArr_m2) )
            
            # create model v spectrum
            vModArr = np.zeros_like(lamSqArr_m2)

            # model cable delay leakage
            u_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*quArr.imag - np.sin(2*np.pi*freqArr*pDict["lag_s"])*vModArr
            v_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*vModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*quArr.imag
            quArr.imag=u_leak
           #vArr=+v_leak
            vArr=-v_leak
           
            return quArr, vArr

        return model

    if (name == 'cable_delay+response'):

        def model(pDict, lamSqArr_m2, IModArr):
            """Simple Faraday thin source + cable delay + I->V leakage"""
 
            IArr = IModArr.copy()
    
            freqArr=C/np.sqrt(lamSqArr_m2)

            # Calculate the complex fractional q and u spectra
            pArr = pDict["fracPol"] * IModArr
    
            # model differential X,Y response
            gain_X = 1
            gain_Y = gain_X * pDict['gain_diff']
        
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


            # model differential X,Y response (see Johnston 2006 for details)
            IArr_leak = 0.5*IArr*(gain_X**2+gain_Y**2)+0.5*QArr*(gain_X**2-gain_Y**2)
            QArr_leak = 0.5*IArr*(gain_X**2-gain_Y**2)+0.5*QArr*(gain_X**2+gain_Y**2)
            IArr = IArr_leak
            QArr = QArr_leak
            UArr = UArr*gain_X*gain_Y
            VArr = VArr*gain_X*gain_Y
    
            QUArr = QArr + 1j*UArr
   
            return QUArr, VArr

        return model

    if (name == 'cable_delay+linear'):
        
        def model(pDict, lamSqArr_m2,  IModArr):
            """Simple Faraday thin source + cable delay + power law fractional polarization"""

            freqArr=C/np.sqrt(lamSqArr_m2)

            # Model power law fractional polarization
            pfracArr = (pDict["fracPol"] * np.ones_like(freqArr)) * (freqArr/freqArr.min())**(pDict["gamma"])
            # Calculate the complex fractional q and u spectra
            pArr = pfracArr * IModArr
            # model Faraday rotation
            quArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                 pDict["RM_radm2"] * lamSqArr_m2) )
    
            # create model v spectrum
            vModArr = np.zeros_like(lamSqArr_m2)

            # model cable delay leakage
            u_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*quArr.imag - np.sin(2*np.pi*freqArr*pDict["lag_s"])*vModArr
            v_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*vModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*quArr.imag
            quArr.imag=u_leak
            #vArr=+v_leak
            vArr=-v_leak
   
            return quArr, vArr
        
        return model

    if (name == 'cable_delay+linear+circular'):
        def model(pDict, lamSqArr_m2,  IModArr):
            """Simple Faraday thin source + cable delay + power law fractional polarization"""

            freqArr=C/np.sqrt(lamSqArr_m2)

            # Model power law fractional polarization
            pfracArr = (pDict["fracPol"] * np.ones_like(freqArr)) * (freqArr/freqArr.min())**(pDict["gamma"])
            # Calculate the complex fractional q and u spectra
            pArr = pfracArr * IModArr
            # model Faraday rotation
            quArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) + pDict["RM_radm2"] * lamSqArr_m2) )
    
            # create model v spectrum
            vfracArr = (pDict["fracPol_V"] * np.ones_like(lamSqArr_m2)) * (freqArr/freqArr.min())**(pDict["gamma_V"])
            vModArr = vfracArr * IModArr
            
            # model cable delay leakage
            u_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*quArr.imag - np.sin(2*np.pi*freqArr*pDict["lag_s"])*vModArr
            v_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*vModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*quArr.imag
            quArr.imag=u_leak
            #vArr=+v_leak
            vArr=-v_leak
   
            return quArr, vArr

        return model

    if (name == 'full_fit'):
        def model(pDict, lamSqArr_m2,  IModArr):
            """Simple Faraday thin source + cable delay + power law fractional polarization"""

            IArr = IModArr.copy()

            freqArr=C/np.sqrt(lamSqArr_m2)

            # Model power law fractional polarization
            pfracArr = (pDict["fracPol"] * np.ones_like(freqArr)) * (freqArr/freqArr.min())**(pDict["gamma"])
            # Calculate the complex fractional q and u spectra
            pArr = pfracArr * IModArr
            # create model v spectrum                                                                                                
            vfracArr = (pDict["fracPol_V"] * np.ones_like(lamSqArr_m2)) * (freqArr/freqArr.min())**(pDict["gamma_V"])
            VModArr = vfracArr * IModArr
            # model differential X,Y response
            gain_X = 1
            gain_Y = gain_X * pDict['gain_diff']
               # model Faraday rotation
            QUArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) + pDict["RM_radm2"] * lamSqArr_m2) )

            QArr = QUArr.real
            UArr = QUArr.imag

            # model cable delay leakage
            U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s"])*VModArr
            V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*VModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*UArr
            UArr=U_leak
            VArr=-V_leak
            
            # model differential X,Y response (see Johnston 2006 for details)
            IArr_leak = 0.5*IArr*(gain_X**2+gain_Y**2)+0.5*QArr*(gain_X**2-gain_Y**2)
            QArr_leak = 0.5*IArr*(gain_X**2-gain_Y**2)+0.5*QArr*(gain_X**2+gain_Y**2)
            IArr = IArr_leak
            QArr = QArr_leak
            UArr = UArr*gain_X*gain_Y
            VArr = VArr*gain_X*gain_Y
    
            QUArr = QArr + 1j*UArr
   
            return QUArr, VArr

        return model
        
    if (name == 'test_fit'):
        def model(pDict, lamSqArr_m2,  IModArr):
            """Simple Faraday thin source + cable delay + power law fractional polarization"""

            IArr = IModArr.copy()

            freqArr=C/np.sqrt(lamSqArr_m2)

            # Model power law fractional polarization
            pfracArr = (pDict["fracPol"] * np.ones_like(freqArr)) * (freqArr/freqArr.min())**(pDict["gamma"])
            # Calculate the complex fractional q and u spectra
            pArr = pfracArr * IModArr
            # create model v spectrum
            vfracArr = (pDict["fracPol_V"] * np.ones_like(lamSqArr_m2)) * (freqArr/freqArr.min())**(pDict["gamma_V"])
            VModArr = vfracArr * IModArr
            # model differential X,Y response
            gain_X = 1
            gain_Y = gain_X * pDict['gain_diff']
               # model Faraday rotation
            QUArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) + pDict["RM_radm2"] * lamSqArr_m2) )

            QArr = QUArr.real
            UArr = QUArr.imag

            # model cable delay leakage
            U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s"])*VModArr
            V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s"])*VModArr + np.sin(2*np.pi*freqArr*pDict["lag_s"])*UArr
            UArr=U_leak
            VArr=-V_leak
            
            # secondary delay leakage
            U_leak=np.cos(2*np.pi*freqArr*pDict["lag_s2"])*UArr - np.sin(2*np.pi*freqArr*pDict["lag_s2"])*VArr
            V_leak=np.cos(2*np.pi*freqArr*pDict["lag_s2"])*VArr + np.sin(2*np.pi*freqArr*pDict["lag_s2"])*UArr
            UArr=U_leak
            VArr=-V_leak
            
            # model differential X,Y response (see Johnston 2006 for details)
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

        if (name == None) or (name == 'cable_delay'):

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
                
            ]

        if (name == 'cable_delay+response'):
            
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
    
                {"parname":   "gain_diff",
                 "label":     "gain diff",
                 "value":     1.0,
                 "bounds":    [0.1, 10.0],
                 "priortype": "uniform",
                 "wrap":      0},

            ]


        if (name == 'cable_delay+linear'):
            
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

                {"parname":   "gamma",
                 "label":     "$\gamma$",
                 "value":     0.0,
                 "bounds":    [-3.0, 3.0],
                 "priortype": "uniform",
                 "wrap":      0}
    
            ]

        if (name == 'cable_delay+linear+circular'):
            
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

                {"parname":   "gamma",
                 "label":     "$\gamma_L$",
                 "value":     0.0,
                 "bounds":    [-3.0, 3.0],
                 "priortype": "uniform",
                 "wrap":      0},

                {"parname":   "fracPol_V",
                 "label":     "$p_V$",
                 "value":     0.1,
                 "bounds":    [0.001, 1.0],
                 "priortype": "uniform",
                 "wrap":      0},

                {"parname":   "gamma_V",
                 "label":     "$\gamma_V$",
                 "value":     0.0,
                 "bounds":    [-3.0, 3.0],
                 "priortype": "uniform",
                 "wrap":      0}
    
            ]
        
        if (name == 'full_fit'):

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

                {"parname":   "gamma",
                 "label":     "$\gamma_L$",
                 "value":     0.0,
                 "bounds":    [-3.0, 3.0],
                 "priortype": "uniform",
                 "wrap":      0},

                {"parname":   "fracPol_V",
                 "label":     "$p_V$",
                 "value":     0.1,
                 "bounds":    [0.001, 1.0],
                 "priortype": "uniform",
                 "wrap":      0},

                {"parname":   "gamma_V",
                 "label":     "$\gamma_V$",
                 "value":     0.0,
                 "bounds":    [-3.0, 3.0],
                 "priortype": "uniform",
                 "wrap":      0},

                {"parname":   "gain_diff",
                 "label":     "gain diff",
                 "value":     1.0,
                 "bounds":    [0.1, 10.0],
                 "priortype": "uniform",
                 "wrap":      0}    
            
            ]
            
        if (name == 'test_fit'):

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
                 
                {"parname":   "lag_s2",
                 "label":     "lag (sec)",
                 "value":     0.0,
                 "bounds":    [-1e-7, 1e-7],
                 "priortype": "uniform",
                 "wrap":      0},

                {"parname":   "gamma",
                 "label":     "$\gamma_L$",
                 "value":     0.0,
                 "bounds":    [-3.0, 3.0],
                 "priortype": "uniform",
                 "wrap":      0},

                {"parname":   "fracPol_V",
                 "label":     "$p_V$",
                 "value":     0.1,
                 "bounds":    [0.001, 1.0],
                 "priortype": "uniform",
                 "wrap":      0},

                {"parname":   "gamma_V",
                 "label":     "$\gamma_V$",
                 "value":     0.0,
                 "bounds":    [-3.0, 3.0],
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
