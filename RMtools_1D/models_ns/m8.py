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

    Burn Slab with a Linear RM Grad (Gaussian) beam convolved Foreground

    Ref:
    Tribble (1991) Eq 16
    Sokoloff et al. (1998) Eq 39

    """

    # Calculate the complex fractional q and u spectra
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)
    quArr = (
        pArr
        * np.exp(
            2j
            * (np.radians(pDict["psi0_deg"]) + (0.5 * pDict["RM_radm2"] * lamSqArr_m2))
        )
        * (np.sin(pDict["RM_radm2"] * lamSqArr_m2) / (pDict["RM_radm2"] * lamSqArr_m2))
        * np.exp(
            (2j * pDict["RM_screen_radm2"] * lamSqArr_m2)
            - (2.0 * lamSqArr_m2 ** (2) * pDict["sigma_RM_2"] ** 2)
        )
    )

    return quArr


# -----------------------------------------------------------------------------#
# Priors for the above model.                                                 #
# See https://lscsoft.docs.ligo.org/bilby/prior.html for details.             #
#                                                                             #
# -----------------------------------------------------------------------------#
priors = {
    "fracPol": bilby.prior.Uniform(
        minimum=0.0, maximum=1.0, name="fracPol", latex_label="$p$"
    ),
    "psi0_deg": bilby.prior.Uniform(
        minimum=0,
        maximum=180.0,
        name="psi0_deg",
        latex_label="$\psi_0$ (deg)",
        boundary="periodic",
    ),
    "RM_radm2": bilby.prior.Uniform(
        minimum=0.0,
        maximum=100.0,
        name="RM_radm2",
        latex_label="$RM_{src}$ (rad m$^{-2}$)",
    ),
    "RM_screen_radm2": bilby.prior.Uniform(
        minimum=-1100.0,
        maximum=1100.0,
        name="RM_screen_radm2",
        latex_label="$RM_{scr}$ (rad m$^{-2}$)",
    ),
    "sigma_RM_2": bilby.prior.Uniform(
        minimum=0.0,
        maximum=100.0,
        name="sigma_RM_radm2",
        latex_label="sigma_RM (rad m$^{-2}$)",
    ),
}
