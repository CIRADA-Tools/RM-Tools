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

    Simple Faraday thin source

    Ref:
    Sokoloff et al. (1998) Eq 2
    O'Sullivan et al. (2012) Eq 8
    Ma et al. (2019a) Eq 10

    """

    # Calculate the complex fractional q and u spectra
    # fmt: off
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)
    quArr = pArr * np.exp(
        2j * (np.radians(pDict["psi0_deg"]) + pDict["RM_radm2"] * lamSqArr_m2)
    )
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
}
