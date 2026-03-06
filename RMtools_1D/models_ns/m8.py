# =============================================================================#
#                          MODEL DEFINITION FILE                              #
# =============================================================================#
import bilby
import numpy as np
from astropy.constants import c

C = c.value


# -----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms         #
#  lamSqArr_m2 = Array of lambda-squared values                               #
#  quArr       = Complex array containing the Re and Im spectra.              #
# -----------------------------------------------------------------------------#
def model(pDict, lamSqArr_m2):
    """
    Faraday thin source + differential X/Y phase (cable delay)

    Linear polarization:
        p * exp[ 2i (psi0 + RM lambda^2) ]

    Instrumental leakage:
        U <-> V rotation via differential phase between X/Y

    """

    # Frequency array
    freqArr = C / np.sqrt(lamSqArr_m2)

    # Fractional linear polarization
    pArr = pDict["fracPol"] * np.ones_like(lamSqArr_m2)

    # Intrinsic Faraday rotation
    quArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
			         pDict["RM_radm2"] * lamSqArr_m2)  )

    qArr = quArr.real
    uArr = quArr.imag

    # No intrinsic circular polarization
    vArr = np.zeros_like(lamSqArr_m2)

    # Differential phase between X/Y feeds
    phase = 2 * np.pi * freqArr * pDict["lag_s"] + np.radians(pDict["lag_phi"])

    u_leak = np.cos(phase) * uArr - np.sin(phase) * vArr
    v_leak = np.cos(phase) * vArr + np.sin(phase) * uArr

    # Only Q and U are used in fractional system
    quArr = qArr + 1j * uArr
    vArr = -v_leak

    return quArr, vArr


# -----------------------------------------------------------------------------#
# Priors for the above model.                                                 #
# See https://lscsoft.docs.ligo.org/bilby/prior.html for details              #
# -----------------------------------------------------------------------------#
priors = {
    "fracPol": bilby.prior.Uniform(
        minimum=0.001,
        maximum=1.1,
        name="fracPol",
        latex_label=r"$p$",
    ),
    "psi0_deg": bilby.prior.Uniform(
        minimum=0.0,
        maximum=180.0,
        name="psi0_deg",
        latex_label=r"$\psi_0$ (deg)",
        boundary="periodic",
    ),
    "RM_radm2": bilby.prior.Uniform(
        minimum=-4000.0,
        maximum=4000.0,
        name="RM_radm2",
        latex_label=r"RM (rad m$^{-2}$)",
    ),
    "lag_s": bilby.prior.Uniform(
        minimum=-2e-9,
        maximum=0,
        name="lag_s",
        latex_label=r"lag (sec)",
    ),
    "lag_phi": bilby.prior.Uniform(
        minimum=0.0,
        maximum=360.0,
        name="lag_phi",
        latex_label=r"lag$_\phi$ (deg)",
        boundary="periodic",
    ),
}
