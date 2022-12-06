#=============================================================================#
#                          MODEL DEFINITION FILE                              #
#=============================================================================#
import numpy as np
import bilby

#-----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms, below. #
#  lamSqArr_m2 = Array of lambda-squared values                               #
#  quArr       = Complex array containing the Re and Im spectra.              #
#-----------------------------------------------------------------------------#
def model(pDict, lamSqArr_m2):
    """
    
    Single Faraday component with internal Faraday dispersion
    
    Ref:
    Burn (1966) Eq 18
    Sokoloff et al. (1998) Eq 34
    O'Sullivan et al. (2012) Eq 10
    Ma et al. (2019a) Eq 15
    
    """
    
    # Calculate the complex fractional q and u spectra
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)
    para_S = 2. * lamSqArr_m2**2 * pDict["sigmaRM_radm2"]**2
             - 2j * lamSqArr_m2 * pDict["deltaRM_radm2"]
    quArr = (pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                  pDict["RM_radm2"] * lamSqArr_m2))
             * (1 - np.exp(-1.*para_S)) / para_S
    
    return quArr


#-----------------------------------------------------------------------------#
# Priors for the above model.                                                 #
# See https://lscsoft.docs.ligo.org/bilby/prior.html for details.             #
#                                                                             #
#-----------------------------------------------------------------------------#
priors = {
    "fracPol": bilby.prior.Uniform(
        minimum=0.001, 
        maximum=1.0, 
        name="fracPol", 
        latex_label="$p$"
    ),
    "psi0_deg": bilby.prior.Uniform(
        minimum=0,
        maximum=180.0,
        name="psi0_deg",
        latex_label="$\psi_0$ (deg)",
        boundary="periodic",
    ),
    "RM_radm2": bilby.prior.Uniform(
        minimum=-1100.0,
        maximum=1100.0,
        name="RM_radm2",
        latex_label="RM (rad m$^{-2}$)",
    ),
    "deltaRM_radm2": bilby.prior.Uniform(
        minimum=0.0,
        maximum=100.0,
        name="deltaRM_radm2",
        latex_label="$\Delta{RM}$ (rad m$^{-2}$)",
    ),
    "sigmaRM_radm2": bilby.prior.Uniform(
        minimum=0.0,
        maximum=100.0,
        name="sigmaRM_radm2",
        latex_label="$\sigma_{RM}$ (rad m$^{-2}$)",
    ),
}
