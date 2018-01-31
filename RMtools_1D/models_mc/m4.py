#=============================================================================#
#                          MODEL DEFINITION FILE                              #
#=============================================================================#
import numpy as np


#-----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
# Takes in parameters "inParms" in the format defined below and an array of   #
# wavelength-squared values. Returns a complex array containing the real and  #
# imaginary spectra, i.e., the fractional Stokes Q and U spectra.             #
#-----------------------------------------------------------------------------#
def model(inParms, lamSqArr_m2):
    """Two simple sources with independent Burn depolarisation."""

    # Create a dictionary of parameter values from the inParms structure
    kwargs = {}
    for i in range(len(inParms)):
        kwargs[inParms[i]["parname"]] = inParms[i]["value"]
        
    # Calculate the complex fractional q and u spectra
    pArr1 = kwargs["fracPol1"] * np.ones_like(lamSqArr_m2)
    pArr2 = kwargs["fracPol2"] * np.ones_like(lamSqArr_m2)
    quArr1 = pArr1 * np.exp( 2j * (np.radians(kwargs["psi01_deg"]) +
                                   kwargs["RM1_radm2"] * lamSqArr_m2))
    quArr2 = pArr2 * np.exp( 2j * (np.radians(kwargs["psi02_deg"]) +
                                   kwargs["RM2_radm2"] * lamSqArr_m2))
    quArr = (quArr1 * np.exp(-2.0 * kwargs["sigmaRM1_radm2"]**2.0 
                             * lamSqArr_m2**2.0) +
             quArr2 * np.exp(-2.0 * kwargs["sigmaRM2_radm2"]**2.0 
                             * lamSqArr_m2**2.0))
    
    return quArr


#-----------------------------------------------------------------------------#
# Switches controlling the MCMC exploration and convergence detection.        #
#                                                                             #
# The MCMC sampler can be run in two modes: "auto" or "fixed". In auto mode   #
# the algorithm will attempt to detect convergence by checking if the maximum #
# binned standard deviation changes compared to the total stdev AND if the    #
# the binned median is within some limit compared to the total median.        #
#                                                                             #
#   runMode          ...   [auto] or [fixed]                                  #
#   maxSteps         ...   upper limit (auto) or num of steps to run (fixed)  # 
#   nExploreSteps    ...   number of steps to use exploring parameter space   #
#   nPollSteps       ...   poll the chain statistics after every nPollSteps   #
#   nStableCycles    ...   require final nSteps = nPollSteps X nStableCycles  #
#   likeStdLim       ...   standard deviation stability limit for likelihood  #
#   likeMedLim       ...   median stability limit for likelihood              #
#   parmStdLim       ...   standard deviation stability limit for parameters  #
#   parmMedLim       ...   median stability limit for parameters              #
#                                                                             #
#-----------------------------------------------------------------------------#
runParmDict = {"runMode": "auto",
               "maxSteps": 5000,
               "nExploreSteps": 600,
               "nPollSteps": 20,
               "nStableCycles":20,
               "likeStdLim": 1.2,
               "likeMedLim": 0.3,
               "parmStdLim": 1.2,
               "parmMedLim": 0.2}


#-----------------------------------------------------------------------------#
# Parameters for the above model.                                             #
#                                                                             #
# Each parameter is defined by a dictionary with the following keywords:      #
#   parname    ...   parameter name used in the model function above          #
#   label      ...   latex style label used by plotting functions             #
#   value      ...   value of the parameter used if fixed                     #
#   error      ...   uncertainty on the variable, used as a Gaussian prior    #
#   fixed      ...   is this parameter to be held fixed? [True|False]         #
#   seedrng    ...   range in values for initial walkers - should be broad    #
#   bounds     ...   hard boundaries on this parameter (likelihood == 0)      #
#   wrap       ...   [optional] value wrapping limits (e.g., for an angle)    #
#-----------------------------------------------------------------------------#
inParms = [
    {"parname": "fracPol1",
     "label": "$p_1$",
     "value": 0.1,
     "error": 0.0,
     "fixed": False,
     "seedrng": [0.0001, 0.7],
     "bounds": [0.0000001, 1.0]},
    
    {"parname": "fracPol2",
     "label": "$p_2$",
     "value": 0.1,
     "error": 0.0,
     "fixed": False,
     "seedrng": [0.0001, 0.7],
     "bounds": [0.0000001, 1.0]},
    
    {"parname": "psi01_deg",
     "label": "$\psi_{0,1}$ (deg)",
     "value": 0.0,
     "error": 0.0,
     "fixed": False,
     "seedrng": [0.0, 180.0],
     "bounds": [-90.0, 270.0],
     "wrap": [0.0, 180.0]},
    
    {"parname": "psi02_deg",
     "label": "$\psi_{0,2}$ (deg)",
     "value": 0.0,
     "error": 0.0,
     "fixed": False,
     "seedrng": [0.0, 180.0],
     "bounds": [-90.0, 270.0],
     "wrap": [0.0, 180.0]},
    
    {"parname": "RM1_radm2",
     "label": "$\phi_1$ (rad m$^{-2}$)",
     "value": 0.0,
     "error": 0.0,
     "fixed": False,
     "seedrng": [-100.0, 100.1],
     "bounds": [-10000.0, 10000.0]},
    
    {"parname": "RM2_radm2",
     "label": "$\phi_2$ (rad m$^{-2}$)",
     "value": 0.0,
     "error": 0.0,
     "fixed": False,
     "seedrng": [-100.0, 100.1],
     "bounds": [-10000.0, 10000.0]},
    
    {"parname": "sigmaRM1_radm2",
     "label": "$\sigma_{RM,1}$ (rad m$^{-2}$)",
     "value": 0.0,
     "error": 0.0,
     "fixed": False,
     "seedrng": [0.0, 10.0],
     "bounds": [0.0, 1000.0]},
    
    {"parname": "sigmaRM2_radm2",
     "label": "$\sigma_{RM,2}$ (rad m$^{-2}$)",
     "value": 0.0,
     "error": 0.0,
     "fixed": False,
     "seedrng": [0.0, 10.0],
     "bounds": [0.0, 1000.0]}
]
