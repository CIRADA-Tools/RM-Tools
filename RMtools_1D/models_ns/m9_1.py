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

    Random Slab with a Unif Foreground Screen

    Ref:
    Burn (1966) Eq 17
    Sokoloff et al. (1998) Eq 34,B1,A2

    """

    # fmt: off

    # Calculate the complex fractional q and u spectra
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)
    para_S = (2. * lamSqArr_m2**2 * pDict["sigmaRM_slab_radm2"]**2)

    quArr = (pArr * np.exp(2j * (np.radians(pDict["psi0_deg"]) + pDict["RM_screen_radm2"] * lamSqArr_m2)) * ((1 - np.exp(-1.*para_S)) / para_S))
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
    "sigmaRM_slab_radm2": bilby.prior.Uniform(
        minimum=0.0,
        maximum=100.0,
        name="sigmaRM_SB_radm2",
        latex_label=r"$\sigma_{SB,RM}$ (rad m$^{-2}$)",
    ),
    "RM_screen_radm2": bilby.prior.Uniform(
        minimum=-1100.0,
        maximum=1100.0,
        name="RM_screen_radm2",
        latex_label="$RM_{scr}$ (rad m$^{-2}$)",
    ),
}