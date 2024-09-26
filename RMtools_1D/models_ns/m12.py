#
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
    Single Faraday component with internal Faraday dispersion and
    an foreground external dispersion screen

    Ref:
    Sokoloff et al. (1998) Eq 34
    O'Sullivan et al. (2012) Eq 10
    Oberhelman et al. (in prep) Eq 6
    """

    # Calculate the complex fractional q and u spectra
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)
    para_S = (
        2.0 * lamSqArr_m2**2 * pDict["sigmaRM_radm2"] ** 2
        - 2j * lamSqArr_m2 * pDict["deltaRM_radm2"]
    )

    quArr = (
        pArr
        * np.exp(
            2j * (np.radians(pDict["psi0_deg"]) + pDict["RM_screen"] * lamSqArr_m2)
        )
        * ((1 - np.exp(-1.0 * para_S)) / para_S)
        * np.exp(-2.0 * pDict["sigmaRM_radm2_FG"] ** 2.0 * lamSqArr_m2**2.0)
    )

    return quArr


# -----------------------------------------------------------------------------#
# Priors for the above model.                                                 #
# See https://lscsoft.docs.ligo.org/bilby/prior.html for details.             #
#                                                                             #
# -----------------------------------------------------------------------------#
priors = {
    "fracPol": bilby.prior.Uniform(
        minimum=0.0, maximum=0.6, name="fracPol", latex_label="$p$"
    ),
    "psi0_deg": bilby.prior.Uniform(
        minimum=0,
        maximum=180.0,
        name="psi0_deg",
        latex_label="$\psi_0$ (deg)",
        boundary="periodic",
    ),
    "deltaRM_radm2": bilby.prior.Uniform(
        minimum=0.0,
        maximum=60.0,
        name="deltaRM_radm2",
        latex_label="$\Delta{RM}$ (rad m$^{-2}$)",
    ),
    "sigmaRM_radm2": bilby.prior.Uniform(
        minimum=0.0,
        maximum=50.0,
        name="sigmaRM_radm2",
        latex_label="$\sigma_{RM}$ (rad m$^{-2}$)",
    ),
    "sigmaRM_radm2_FG": bilby.prior.Uniform(
        minimum=0.0,
        maximum=50.0,
        name="sigmaRM_radm2_FG",
        latex_label="$\sigma_{RM,FG}$ (rad m$^{-2}$)",
    ),
    "RM_screen": bilby.prior.Uniform(
        minimum=-1100.0,
        maximum=1100.0,
        name="RM_screen_radm2_FG",
        latex_label="$RM_{screen,FG}$ (rad m$^{-2}$)",
    ),
}
