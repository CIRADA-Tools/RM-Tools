import argparse
import copy
import imp
import json
import os
import shutil
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pymultinest as pmn

try:
    from mpi4py import MPI

    mpiSwitch = True
except:
    mpiSwitch = False

from RMutils import corner, models
#from baseband_analysis.RMutils import corner, models

######################## loading in version of RMutils built from docker file ##############################################
from RMutils.util_misc import (
    create_frac_spectra,
    create_frac_spectra_test,
    poly5,
    toscalar,
)
from RMutils.util_plotTk import (
    CustomNavbar,
    plot_Ipqu_spectra_chime,
    plot_Ipqu_spectra_fig,
    plot_pqu_spectra_chime,
    plot_q_vs_u_ax_chime,
)

######################## uncomment lines below to load in local version of RMutils tools for testing ##############################################
# from baseband_analysis.RMutils.util_misc import create_frac_spectra, create_frac_spectra_test, poly5, toscalar
# from baseband_analysis.RMutils.util_plotTk import plot_Ipqu_spectra_fig, plot_Ipqu_spectra_chime, plot_pqu_spectra_chime, plot_q_vs_u_ax_chime, CustomNavbar
# from baseband_analysis.RMutils import corner
# from baseband_analysis.RMutils import models

# Fail if script has been started with mpiexec & mpi4py is not installed
if os.environ.get("OMPI_COMM_WORLD_SIZE") is not None:
    if not mpiSwitch:
        print("Script called with mpiexec, but mpi4py not installed")
        sys.exit()

C = 2.997924538e8  # Speed of light [m/s]


def run_qufit(
    data,
    modelName,
    IMod=None,
    polyOrd=3,
    nBits=32,
    verbose=False,
    diagnostic_plots=True,
    values=None,
    bounds=None,
):

    """Function for Nested sampling fitting of Stokes parameters"""

    if mpiSwitch:
        # Get the processing environment
        mpiComm = MPI.COMM_WORLD
        mpiSize = mpiComm.Get_size()
        mpiRank = mpiComm.Get_rank()
    else:
        mpiSize = 1
        mpiRank = 0

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    if isinstance(diagnostic_plots, str):
        outDir = diagnostic_plots
    else:
        # outDir=os.path.expanduser("~")
        outDir = "/tmp"
    nestOut = outDir + "/QUfit_nest/"
    if mpiRank == 0:
        if os.path.exists(nestOut):
            shutil.rmtree(nestOut, True)
        os.mkdir(nestOut)
    if mpiSwitch:
        mpiComm.Barrier()

    # Read the data file in the root process
    if mpiRank == 0:
        dataArr = data.copy()

    if mpiSwitch:
        dataArr = mpiComm.bcast(dataArr, root=0)

    # Parse the data array
    # freq_Hz, I, Q, U, V, dI, dQ, dU, dV
    try:
        (freqArr_Hz, IArr, QArr, UArr, VArr, dIArr, dQArr, dUArr, dVArr) = dataArr
        if mpiRank == 0:
            print("\nFormat [freq_Hz, I, Q, U, V, dI, dQ, dU, dV]")
    except Exception:
        print("pass data in format: [freq_Hz, I, Q, U, V, dI, dQ, dU, dV]")
        return

    # Convert to GHz for convenience
    freqArr_GHz = freqArr_Hz / 1e9
    lamSqArr_m2 = np.power(C / freqArr_Hz, 2.0)

    # Fit the Stokes I spectrum and create the fractional spectra
    if mpiRank == 0:
        if IMod is None:
            dataArr = create_frac_spectra_test(
                freqArr=freqArr_GHz,
                IArr=IArr,
                QArr=QArr,
                UArr=UArr,
                dIArr=dIArr,
                dQArr=dQArr,
                dUArr=dUArr,
                VArr=VArr,
                dVArr=dVArr,
                polyOrd=polyOrd,
                IModArr=None,
                verbose=True,
            )
        else:
            dataArr = create_frac_spectra_test(
                freqArr=freqArr_GHz,
                IArr=IArr,
                QArr=QArr,
                UArr=UArr,
                dIArr=dIArr,
                dQArr=dQArr,
                dUArr=dUArr,
                VArr=VArr,
                dVArr=dVArr,
                polyOrd=polyOrd,
                IModArr=IMod(freqArr_Hz),
                verbose=True,
            )

    else:
        dataArr = None
    if mpiSwitch:
        dataArr = mpiComm.bcast(dataArr, root=0)
    (IModArr, qArr, uArr, vArr, dqArr, duArr, dvArr, IfitDict) = dataArr

    # -------------------------------------------------------------------------#

    # Load the model and parameters from the relevant file
    if mpiSwitch:
        mpiComm.Barrier()
    global model
    model = models.get_model(modelName)
    inParms = models.get_params(modelName)

    # Let's time the sampler
    if mpiRank == 0:
        startTime = time.time()

    # Unpack the inParms structure
    parNames = [x["parname"] for x in inParms]
    labels = [x["label"] for x in inParms]
    if values is None:
        values = [x["value"] for x in inParms]
    if bounds is None:
        bounds = [x["bounds"] for x in inParms]
    priorTypes = [x["priortype"] for x in inParms]
    wraps = [x["wrap"] for x in inParms]
    nDim = len(priorTypes)
    fixedMsk = [0 if x == "fixed" else 1 for x in priorTypes]
    nFree = sum(fixedMsk)

    # Set the prior function given the bounds of each parameter
    prior = prior_call(priorTypes, bounds, values)

    # Set the likelihood function given the data
    lnlike = lnlike_call(
        parNames, lamSqArr_m2, QArr, dQArr, UArr, dUArr, VArr, dVArr, IModArr
    )

    # Let's time the sampler
    if mpiRank == 0:
        startTime = time.time()

    # Run nested sampling using PyMultiNest
    nestArgsDict = merge_two_dicts(init_mnest(), models.nestArgsDict)
    nestArgsDict["n_params"] = nDim
    nestArgsDict["n_dims"] = nDim
    nestArgsDict["outputfiles_basename"] = nestOut
    nestArgsDict["LogLikelihood"] = lnlike
    nestArgsDict["Prior"] = prior
    pmn.run(**nestArgsDict)

    # Do the post-processing on one processor
    if mpiSwitch:
        mpiComm.Barrier()
    if mpiRank == 0:

        # Query the analyser object for results
        aObj = pmn.Analyzer(n_params=nDim, outputfiles_basename=nestOut)
        statDict = aObj.get_stats()
        fitDict = aObj.get_best_fit()
        endTime = time.time()

        # NOTE: The Analyser methods do not work well for parameters with
        # posteriors that overlap the wrap value. Use np.percentile instead.
        pMed = [None] * nDim
        for i in range(nDim):
            pMed[i] = statDict["marginals"][i]["median"]
        lnLike = fitDict["log_likelihood"]
        lnEvidence = statDict["nested sampling global log-evidence"]
        dLnEvidence = statDict["nested sampling global log-evidence error"]

        # Get the best-fitting values & uncertainties directly from chains
        chains = aObj.get_equal_weighted_posterior()
        chains = wrap_chains(chains, wraps, bounds, pMed)
        p = [None] * nDim
        errPlus = [None] * nDim
        errMinus = [None] * nDim
        g = lambda v: (v[1], v[2] - v[1], v[1] - v[0])
        for i in range(nDim):
            p[i], errPlus[i], errMinus[i] = g(
                np.percentile(chains[:, i], [15.72, 50, 84.27])
            )

        # Calculate goodness-of-fit parameters
        nData = 2.0 * len(lamSqArr_m2)
        dof = nData - nFree - 1
        chiSq = chisq_model(
            parNames, p, lamSqArr_m2, QArr, dQArr, UArr, dUArr, VArr, dVArr, IModArr
        )
        chiSqRed = chiSq / dof
        AIC = 2.0 * nFree - 2.0 * lnLike
        AICc = 2.0 * nFree * (nFree + 1) / (nData - nFree - 1) - 2.0 * lnLike
        BIC = nFree * np.log(nData) - 2.0 * lnLike

        # Summary of run
        print("")
        print("-" * 80)
        print("SUMMARY OF SAMPLING RUN:")
        print("#-PROCESSORS  = %d" % mpiSize)
        print("RUN-TIME      = %.2f" % (endTime - startTime))
        print("DOF           = %d" % dof)
        print("CHISQ:        = %.3g" % chiSq)
        print("CHISQ RED     = %.3g" % chiSqRed)
        print("AIC:          = %.3g" % AIC)
        print("AICc          = %.3g" % AICc)
        print("BIC           = %.3g" % BIC)
        print("ln(EVIDENCE)  = %.3g" % lnEvidence)
        print("dLn(EVIDENCE) = %.3g" % dLnEvidence)
        print("")
        print("-" * 80)
        print("RESULTS:\n")
        for i in range(len(p)):
            print(
                "%s = %.4g (+%3g, -%3g)" % (parNames[i], p[i], errPlus[i], errMinus[i])
            )
        print("-" * 80)
        print("")

        # Create a save dictionary and store final p in values
        #         outFile = nestOut + "m%d_nest.json" % modelNum
        outFile = nestOut + "%s_nest.json" % modelName
        IfitDict["p"] = toscalar(IfitDict["p"].tolist())
        saveDict = {
            "parNames": toscalar(parNames),
            "labels": toscalar(labels),
            "values": toscalar(p),
            "errPlus": toscalar(errPlus),
            "errMinus": toscalar(errMinus),
            "bounds": toscalar(bounds),
            "priorTypes": toscalar(priorTypes),
            "wraps": toscalar(wraps),
            "dof": toscalar(dof),
            "chiSq": toscalar(chiSq),
            "chiSqRed": toscalar(chiSqRed),
            "AIC": toscalar(AIC),
            "AICc": toscalar(AICc),
            "BIC": toscalar(BIC),
            "IfitDict": IfitDict,
        }
        json.dump(saveDict, open(outFile, "w"))
        print("Results saved in JSON format to:\n '%s'\n" % outFile)

        # Plot the data and best-fitting model
        lamSqHirArr_m2 = np.linspace(lamSqArr_m2[0], lamSqArr_m2[-1], 10000)
        freqHirArr_Hz = C / np.sqrt(lamSqHirArr_m2)
        pDict = {k: v for k, v in zip(parNames, p)}
        if IMod:
            IModHirArr = IMod(freqHirArr_Hz)
        else:
            IModHirArr = poly5(IfitDict["p"])(freqHirArr_Hz / 1e9)
        quModArr, vModArr = model(pDict, lamSqHirArr_m2, IModHirArr)
        specFig = plt.figure(figsize=(10, 6))
        plot_pqu_spectra_chime(
            freqArr_Hz=freqArr_Hz,
            IArr=IArr,
            qArr=qArr,
            uArr=uArr,
            dIArr=dIArr,
            dqArr=dqArr,
            duArr=duArr,
            freqHirArr_Hz=freqHirArr_Hz,
            IModArr=IModHirArr,
            qModArr=quModArr.real,
            uModArr=quModArr.imag,
            fig=specFig,
        )
        specFig.canvas.draw()

        # Plot the posterior samples in a corner plot
        chains = aObj.get_equal_weighted_posterior()
        chains = wrap_chains(chains, wraps, bounds, p)[:, :nDim]
        iFixed = [i for i, e in enumerate(fixedMsk) if e == 0]
        chains = np.delete(chains, iFixed, 1)
        for i in sorted(iFixed, reverse=True):
            del labels[i]
            del p[i]
        cornerFig = corner.corner(
            xs=chains,
            labels=labels,
            range=[0.99999] * nFree,
            truths=p,
            quantiles=[0.1572, 0.8427],
            bins=30,
        )

        # Plot the stokes Q vs. U (NEEDS WORK)
        qvsuFig = plot_q_vs_u_ax_chime(
            freqArr_Hz=freqArr_Hz,
            qArr=qArr,
            uArr=uArr,
            dqArr=dqArr,
            duArr=duArr,
            freqHirArr_Hz=freqHirArr_Hz,
            qModArr=quModArr.real / IModHirArr,
            uModArr=quModArr.imag / IModHirArr,
        )

        if diagnostic_plots:
            if isinstance(diagnostic_plots, bool):
                qvsuFig.show()
                sys.stdout.flush()
            else:
                outFile = diagnostic_plots + "/fig_%s_specfit.pdf" % modelName
                specFig.savefig(outFile)
                print("Plot of best-fitting model saved to:\n '%s'\n" % outFile)
                outFile = diagnostic_plots + "/fig_%s_corner.pdf" % modelName
                cornerFig.savefig(outFile)
                print("Plot of posterior samples saved to \n '%s'\n" % outFile)
                outFile = diagnostic_plots + "/fig_%s_q_vs_u.pdf" % modelName
                qvsuFig.savefig(outFile)

        pol_prod = zip(p, errPlus, errMinus)

        return (
            list(pol_prod),
            freqHirArr_Hz,
            qArr,
            uArr,
            vArr,
            dqArr,
            duArr,
            dvArr,
            IModArr,
            quModArr.real,
            quModArr.imag,
            vModArr,
        )


def lnlike_call(parNames, lamSqArr_m2, qArr, dqArr, uArr, duArr, vArr, dvArr, IModArr):
    """ Returns a function to evaluate the log-likelihood """

    def lnlike(p, nDim, nParms):

        # Evaluate the model and calculate the joint ln(like)
        # Silva 2006
        pDict = {k: v for k, v in zip(parNames, p)}
        quMod, vMod = model(pDict, lamSqArr_m2, IModArr)
        #         dquArr = np.sqrt(np.power(dqArr, 2) + np.power(duArr, 2))
        dquvArr = np.sqrt(np.power(dqArr, 2) + np.power(duArr, 2) + np.power(dvArr, 2))
        chiSqQ = np.nansum(np.power((qArr - quMod.real) / dqArr, 2))
        chiSqU = np.nansum(np.power((uArr - quMod.imag) / dqArr, 2))
        chiSqV = np.nansum(np.power((vArr - vMod) / dvArr, 2))

        #         nData = len(dquArr)
        nData = len(dquvArr)
        logLike = (
            -nData * np.log(2.0 * np.pi)
            - 2.0 * np.nansum(np.log(dquvArr))
            - chiSqQ / 2.0
            - chiSqU / 2.0
            - chiSqV / 2.0
        )

        return logLike

    return lnlike


# -----------------------------------------------------------------------------#
def chisq_model(
    parNames, p, lamSqArr_m2, qArr, dqArr, uArr, duArr, vArr, dvArr, IModArr
):
    """Calculate the chi^2 for the current model, given the data."""

    # Evaluate the model and calculate chisq
    pDict = {k: v for k, v in zip(parNames, p)}
    quMod, vMod = model(pDict, lamSqArr_m2, IModArr)
    chisq = np.nansum(
        np.power((qArr - quMod.real) / dqArr, 2)
        + np.power((uArr - quMod.imag) / duArr, 2)
        + np.power((vArr - vMod.imag) / dvArr, 2)
    )

    return chisq


def prior_call(priorTypes, bounds, values):
    """Returns a function to transform (0-1) range to the distribution of
    values for each parameter. Note that a numpy vectorised version of this
    function fails because of type-errors."""

    def prior(p, nDim, nParms):
        for i in range(nDim):
            if priorTypes[i] == "log":
                bMin = np.log(np.abs(bounds[i][0]))
                bMax = np.log(np.abs(bounds[i][1]))
                p[i] *= bMax - bMin
                p[i] += bMin
                p[i] = np.exp(p[i])
            elif priorTypes[i] == "normal":
                bMin, bMax = bounds[i]
                sigma = (bMax - bMin) / 2.0
                mu = bMin + sigma
                p[i] = mu + sigma * ndtri(p[i])
            elif priorTypes[i] == "fixed":
                p[i] = values[i]
            else:  # uniform (linear)
                bMin, bMax = bounds[i]
                p[i] = bMin + p[i] * (bMax - bMin)
        return p

    return prior


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def init_mnest():
    """Initialise MultiNest arguments"""

    argsDict = {
        "LogLikelihood": "",
        "Prior": "",
        "n_dims": 0,
        "n_params": 0,
        "n_clustering_params": 0,
        "wrapped_params": None,
        "importance_nested_sampling": False,
        "multimodal": False,
        "const_efficiency_mode": False,
        "n_live_points": 100,
        "evidence_tolerance": 0.5,
        "sampling_efficiency": "model",
        "n_iter_before_update": 500,
        "null_log_evidence": -1.0e90,
        "max_modes": 100,
        "mode_tolerance": -1.0e90,
        "outputfiles_basename": "",
        "seed": -1,
        "verbose": True,
        "resume": True,
        "context": 0,
        "write_output": True,
        "log_zero": -1.0e100,
        "max_iter": 0,
        "init_MPI": False,
        "dump_callback": None,
    }
    return argsDict


def wrap_chains(chains, wraps, bounds, p, verbose=False):

    # Get the indices of the periodic parameters
    iWrap = [i for i, e in enumerate(wraps) if e != 0]

    # Loop through the chains for periodic parameters
    for i in iWrap:
        wrapLow = bounds[i][0]
        wrapHigh = bounds[i][1]
        rng = wrapHigh - wrapLow

        # Shift the wrapping to centre on the best fit value
        wrapCent = wrapLow + (wrapHigh - wrapLow) / 2.0
        wrapLow += p[i] - wrapCent
        wrapHigh += p[i] - wrapCent
        chains[:, i] = ((chains[:, i] - wrapLow) % rng) + wrapLow
        if verbose:
            print(
                "Wrapped parameter '%d' in range [%s, %s] ..." % (i, wrapLow, wrapHigh)
            )

    return chains
