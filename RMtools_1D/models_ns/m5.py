# =============================================================================#
#                          MODEL DEFINITION FILE                              #
# =============================================================================#
import bilby
import numpy as np


# -----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms, below. #
#  lamSqArr_m2 = Array of lambda-squared values                               #
#  quArr       = Complex array containing the Re and Im spectra.              #
# -----------------------------------------------------------------------------#
def model(pDict, lamSqArr_m2):
    """

    Single Faraday component with differential Faraday rotation
    "Burn slab"

    Ref:
    Burn (1966) Eq 18; with N >> (H_r/2H_z^0)^2
    Sokoloff et al. (1998) Eq 3
    O'Sullivan et al. (2012) Eq 9
    Ma et al. (2019a) Eq 12

    """

    # Calculate the complex fractional q and u spectra
    # fmt: off
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)
    quArr = (pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
                                  (0.5*pDict["deltaRM_radm2"] +
                                   pDict["RM_radm2"]) * lamSqArr_m2))
             * np.sin(pDict["deltaRM_radm2"] * lamSqArr_m2) /
               (pDict["deltaRM_radm2"] * lamSqArr_m2))
    # fmt: on

    return quArr


# -----------------------------------------------------------------------------#
# Priors for the above model.                                                 #
# See https://lscsoft.docs.ligo.org/bilby/prior.html for details.             #
#                                                                             #
# -----------------------------------------------------------------------------#
priors = {
    "fracPol": bilby.prior.Uniform(
        minimum=0.0, maximum=1.0, name="fracPol", latex_label=r"$p$"
    ),
    "psi0_deg": bilby.prior.Uniform(
        minimum=0,
        maximum=180.0,
        name="psi0_deg",
        latex_label=r"$\psi_0$ (deg)",
        boundary="periodic",
    ),
    "RM_radm2": bilby.prior.Uniform(
        minimum=-1100.0,
        maximum=1100.0,
        name="RM_radm2",
        latex_label=r"RM (rad m$^{-2}$)",
    ),
    "deltaRM_radm2": bilby.prior.Uniform(
        minimum=0.0,
        maximum=100.0,
        name="deltaRM_radm2",
        latex_label=r"$\Delta{RM}$ (rad m$^{-2}$)",
    ),
}
