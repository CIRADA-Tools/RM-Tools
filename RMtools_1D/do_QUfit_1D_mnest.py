#!/usr/bin/env python

#=============================================================================#
#                                                                             #
# NAME:     do_QUfit_1D_nest.py                                               #
#                                                                             #
# PURPOSE:  Code to simultaneously fit Stokes Q/I and U/I spectra with a      #
#           Faraday active models.                                            #
#                                                                             #
# MODIFIED: 12-Mar-2018 by C. Purcell                                         #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#   main           ... parse command line arguments and initiate procedure    #
#   run_qufit      ... main function of the QU-fitting procedure              #
#   prior_call     ... return the prior tranform fucntion given prior limits  #
#   lnlike_call    ... return a function to evaluate ln(like) given the data  #
#   chisq_model    ... calculate the chi-squared for the model                #
#   wrap_chains    ... wrap and shift chains of periodic parameters           #
#   init_mnest     ... initialise multinest using a default dictionary        #
#   merge_two_dicts .. merge two dictionaries                                 #
#                                                                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2018 Cormac R. Purcell                                        #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
#=============================================================================#

from shutil import Error
import sys
import os
import shutil
import copy
import time
import importlib
import json
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pymultinest as pmn
try:
    from mpi4py import MPI
    mpiSwitch = True
except:
    mpiSwitch = False

from RMutils.util_misc import create_frac_spectra
from RMutils.util_misc import calculate_StokesI_model
from RMutils.util_misc import toscalar
from RMutils.util_plotTk import plot_Ipqu_spectra_fig
from RMutils.util_plotTk import CustomNavbar
from RMutils import corner

# Fail if script has been started with mpiexec & mpi4py is not installed
if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
    if not mpiSwitch:
        print("Script called with mpiexec, but mpi4py not installed")
        sys.exit()
        
C = 2.997924538e8 # Speed of light [m/s]


#-----------------------------------------------------------------------------#
def main():
    """
    Start the run_qufit procedure if called from the command line.
    
    Run QU-fitting on polarised spectra (1D) stored in an ASCII file. The
    Stokes I spectra is first fit with a polynomial and the resulting model
    used to create fractional q = Q/I and u = U/I spectra. If the 'noStokesI'
    option is given, the input data are assumed to be fractional already.

    The script uses the Nested Sampling algorithm (Skilling 2004) to find
    the best fitting parameters, given a prior function on each free parameter.
    The sampling algorithm also calculates the Bayesian evidence, which can
    be used for model comparison. Factors of >10 between models mean that one
    model is strongly favoured over the other.

    Models and priors are  specified as Python code in files called 'mX.py'
    within the 'models_ns' directory. See the existing files for examples
    drawn from the paper Sokoloff et al. 1998, MNRAS 229, pg 189.

    Main algorithm handles the command line interface, passing all arguments
    one to run_qufit().
    """

    # Help string to be shown using the -h option
    descStr = """
    Run QU-fitting on polarised spectra (1D) stored in an ASCII file. The
    Stokes I spectra is first fit with a polynomial and the resulting model
    used to create fractional q = Q/I and u = U/I spectra. If the 'noStokesI'
    option is given, the input data are assumed to be fractional already.

    The script uses the Nested Sampling algorithm (Skilling 2004) to find
    the best fitting parameters, given a prior function on each free parameter.
    The sampling algorithm also calculates the Bayesian evidence, which can
    be used for model comparison. Factors of >10 between models mean that one
    model is strongly favoured over the other.

    Models and priors are  specified as Python code in files called 'mX.py'
    within the 'models_ns' directory. See the existing files for examples
    drawn from the paper Sokoloff et al. 1998, MNRAS 229, pg 189.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataFile", metavar="dataFile.dat", nargs=1,
                        help="ASCII file containing Stokes spectra & errors.")
    parser.add_argument("-m", dest="modelNum", type=int, default=1,
                        help="model number to fit [1].")
    parser.add_argument("-f", dest="fit_function", type=str, default="log",
                        help="Stokes I fitting function: 'linear' or ['log'] polynomials.")
    parser.add_argument("-o", dest="polyOrd", type=int, default=2,
                        help="polynomial order to fit to I spectrum: 0-5 supported, 2 is default.\nSet to negative number to enable dynamic order selection.")
    parser.add_argument("-i", dest="noStokesI", action="store_true",
                        help="ignore the Stokes I spectrum [False].")
    parser.add_argument("-p", dest="showPlots", action="store_true",
                        help="show the plots [False].")
    parser.add_argument("-d", dest="debug", action="store_true",
                        help="turn on debugging messages/plots [False].")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="verbose mode [False].")
    parser.add_argument("-c", dest="sigma_clip", type=float, default=5,
                        help="How many sigma to clip around parameter modes [5].")
    parser.add_argument("-r", dest="restart", action="store_false",
                        help="Restart fitting and delete previous run [True].")
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.dataFile[0]):
        print("File does not exist: '%s'." % args.dataFile[0])
        sys.exit()
    dataDir, dummy = os.path.split(args.dataFile[0])

    # Run the QU-fitting procedure
    run_qufit(dataFile     = args.dataFile[0],
              modelNum     = args.modelNum,
              outDir       = dataDir,
              polyOrd      = args.polyOrd,
              nBits        = 32,
              noStokesI    = args.noStokesI,
              showPlots    = args.showPlots,
              sigma_clip   = args.sigma_clip,
              debug        = args.debug,
              verbose      = args.verbose,
              restart      = args.restart,
              fit_function = args.fit_function
              )


#-----------------------------------------------------------------------------#
def run_qufit(dataFile, modelNum, outDir="", polyOrd=2, nBits=32,
              noStokesI=False, showPlots=False, sigma_clip=5, 
              debug=False, verbose=False, restart=True,fit_function='log'):
    """Carry out QU-fitting using the supplied parameters:
        dataFile (str, required): relative or absolute path of file containing 
            frequencies and Stokes parameters with errors.
        modelNum (int, required): number of model to be fit to data. Models and
             priors are specified as Python code in files called 'mX.py' within  
            the 'models_ns' directory.
        outDir (str): relative or absolute path to save outputs to. Defaults to
            working directory.
        polyOrd (int): Order of polynomial to fit to Stokes I spectrum (used to
            normalize Q and U values). Defaults to 3 (cubic).
        nBits (int): number of bits to use in internal calculations.
        noStokesI (bool): set True if the Stokes I spectrum should be ignored.
        showPlots (bool): Set true if the spectrum and parameter space plots
            should be displayed.
        sigma_clip (float): How many standard deviations to clip around the 
            mean of each mode in the parameter postierors.
        debug (bool): Display debug messages.
        verbose (bool): Print verbose messages/results to terminal.
        
        Returns: nothing. Results saved to files and/or printed to terminal."""

    # Get the processing environment
    if mpiSwitch:
        mpiComm = MPI.COMM_WORLD
        mpiSize = mpiComm.Get_size()
        mpiRank = mpiComm.Get_rank()
    else:
        mpiSize = 1
        mpiRank = 0
        
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Output prefix is derived from the input file name
    prefixOut, ext = os.path.splitext(dataFile)
    nestOut = f"{prefixOut}_m{modelNum}_nest/"
    if mpiRank==0:
        if os.path.exists(nestOut) and restart:
            shutil.rmtree(nestOut, True)
            os.mkdir(nestOut)
        elif not os.path.exists(nestOut) and restart:
            os.mkdir(nestOut)
        elif not os.path.exists(nestOut) and not restart:
            print("Restart requested, but previous run not found!")
            raise Exception(f"{nestOut} does not exist.")
    if mpiSwitch:
        mpiComm.Barrier()

    # Read the data file in the root process
    if mpiRank==0:
        dataArr = np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
    else:
        dataArr = None
    if mpiSwitch:
        dataArr = mpiComm.bcast(dataArr, root=0)

    # Parse the data array
    # freq_Hz, I, Q, U, dI, dQ, dU
    try:
        (freqArr_Hz, IArr, QArr, UArr,
         dIArr, dQArr, dUArr) = dataArr
        if mpiRank==0:
            print("\nFormat [freq_Hz, I, Q, U, dI, dQ, dU]")
    except Exception:
        # freq_Hz, Q, U, dQ, dU
        try:
            (freqArr_Hz, QArr, UArr, dQArr, dUArr) = dataArr
            if mpiRank==0:
                print("\nFormat [freq_Hz, Q, U,  dQ, dU]")
            noStokesI = True
        except Exception:
            print("\nError: Failed to parse data file!")
            if debug:
                print(traceback.format_exc())
            if mpiSwitch:
                MPI.Finalize()
            return

    # If no Stokes I present, create a dummy spectrum = unity
    if noStokesI:
        if mpiRank==0:
            print("Note: no Stokes I data - assuming fractional polarisation.")
        IArr = np.ones_like(QArr)
        dIArr = np.zeros_like(QArr)

    # Convert to GHz for convenience
    lamSqArr_m2 = np.power(C/freqArr_Hz, 2.0)

    # Fit the Stokes I spectrum and create the fractional spectra
    if mpiRank==0:
        dataArr = create_frac_spectra(freqArr=freqArr_Hz,
                                      IArr=IArr,
                                      QArr=QArr,
                                      UArr=UArr,
                                      dIArr=dIArr,
                                      dQArr=dQArr,
                                      dUArr=dUArr,
                                      polyOrd=polyOrd,
                                      verbose=True,
                                      fit_function=fit_function)
    else:
        dataArr = None
    if mpiSwitch:
        dataArr = mpiComm.bcast(dataArr, root=0)
    (IModArr, qArr, uArr, dqArr, duArr, IfitDict) = dataArr

    # Plot the data and the Stokes I model fit
    if mpiRank==0:
        print("Plotting the input data and spectral index fit.")
        freqHirArr_Hz =  np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)
        IModHirArr = calculate_StokesI_model(IfitDict,freqHirArr_Hz)
        specFig = plt.figure(facecolor='w',figsize=(10, 6))
        plot_Ipqu_spectra_fig(freqArr_Hz     = freqArr_Hz,
                              IArr           = IArr,
                              qArr           = qArr,
                              uArr           = uArr,
                              dIArr          = dIArr,
                              dqArr          = dqArr,
                              duArr          = duArr,
                              freqHirArr_Hz  = freqHirArr_Hz,
                              IModArr        = IModHirArr,
                              fig            = specFig)

        # Use the custom navigation toolbar
        try:
            specFig.canvas.toolbar.pack_forget()
            CustomNavbar(specFig.canvas, specFig.canvas.toolbar.window)
        except Exception:
            pass

        # Display the figure
        if showPlots:
            specFig.canvas.draw()
            specFig.show()

    #-------------------------------------------------------------------------#

    # Load the model and parameters from the relevant file
    if mpiSwitch:
        mpiComm.Barrier()
    if mpiRank==0:
        print("\nLoading the model from 'models_ns/m%d.py' ..."  % modelNum)
    #First check the working directory for a model. Failing that, try the install directory.
    try:
        spec=importlib.util.spec_from_file_location("m%d" % modelNum,"models_ns/m%d.py" % modelNum)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod] = mod
        spec.loader.exec_module(mod)
    except FileNotFoundError:
        try:
            RMtools_dir=os.path.dirname(importlib.util.find_spec('RMtools_1D').origin)
            spec=importlib.util.spec_from_file_location("m%d" % modelNum,RMtools_dir+"/models_ns/m%d.py" % modelNum)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod] = mod
            spec.loader.exec_module(mod)
        except:
            print('Model could not be found! Please make sure model is present either in {}/models_ns/, or in {}/RMtools_1D/models_ns/'.format(os.getcwd(),RMtools_dir))
            sys.exit()

    global model
    model = mod.model


    # Let's time the sampler
    if mpiRank==0:
        startTime = time.time()

    # Unpack the inParms structure
    parNames = [x["parname"] for x in mod.inParms]
    labels = [x["label"] for x in mod.inParms]
    values = [x["value"] for x in mod.inParms]
    bounds = [x["bounds"] for x in mod.inParms]
    priorTypes = [x["priortype"] for x in mod.inParms]
    wraps = [x["wrap"] for x in mod.inParms]
    nDim = len(priorTypes)
    fixedMsk = [0 if x=="fixed" else 1 for x in priorTypes]
    nFree = sum(fixedMsk)

    # Set the prior function given the bounds of each parameter
    prior = prior_call(priorTypes, bounds, values)

    # Set the likelihood function given the data
    lnlike = lnlike_call(parNames, lamSqArr_m2, qArr, dqArr, uArr, duArr)

    # Let's time the sampler
    if mpiRank==0:
        startTime = time.time()

    # Run nested sampling using PyMultiNest
    nestArgsDict = merge_two_dicts(init_mnest(), mod.nestArgsDict)
    nestArgsDict["n_params"]             = nDim
    nestArgsDict["n_dims"]               = nDim
    nestArgsDict["outputfiles_basename"] = nestOut
    nestArgsDict["LogLikelihood"]        = lnlike
    nestArgsDict["Prior"]                = prior
    # Look for multiple modes
    nestArgsDict['multimodal']           = True
    nestArgsDict['n_clustering_params']  = nDim
    nestArgsDict['verbose']              = False
    pmn.run(**nestArgsDict)
    # Save parnames for use with PyMultinest tools
    json.dump(parNames, open(f'{nestOut}/params.json', 'w'))

    # Do the post-processing on one processor
    if mpiSwitch:
        mpiComm.Barrier()
    if mpiRank==0:

        # Query the analyser object for results
        aObj = pmn.Analyzer(n_params=nDim, outputfiles_basename=nestOut)
        statDict = aObj.get_stats()
        fitDict = aObj.get_best_fit()
        endTime = time.time()

        # NOTE: The Analyser methods do not work well for parameters with
        # posteriors that overlap the wrap value. Use np.percentile instead.
        pMed = [None]*nDim
        for i in range(nDim):
            pMed[i] = statDict["marginals"][i]['median']
        lnLike = fitDict["log_likelihood"]
        lnEvidence = statDict["nested sampling global log-evidence"]
        dLnEvidence = statDict["nested sampling global log-evidence error"]

        # Get the best-fitting values & uncertainties directly from chains
        chains =  aObj.get_equal_weighted_posterior()
        chains = wrap_chains(chains, wraps, bounds, pMed)
        # Find the mode with the highest evidence
        mode_evidence = [mode['strictly local log-evidence'] for mode in statDict['modes']]
        # Get the max and std for modal value
        modes = np.array(statDict['modes'][np.argmax(mode_evidence)]['maximum a posterior'])
        sigmas = np.array(statDict['modes'][np.argmax(mode_evidence)]['sigma'])
        upper = modes + sigma_clip * sigmas
        lower = modes - sigma_clip * sigmas

        p = [None]*nDim
        errPlus = [None]*nDim
        errMinus = [None]*nDim
        g = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        for i in range(nDim):
            # Get stats around modal value
            idx = (chains[:,i] > lower[i]) & (chains[:,i] < upper[i])
            p[i], errPlus[i], errMinus[i] = \
                        g(np.percentile(chains[idx, i], [15.72, 50, 84.27]))

        # Calculate goodness-of-fit parameters
        nData = 2.0 * len(lamSqArr_m2)
        dof = nData - nFree -1
        chiSq = chisq_model(parNames, p, lamSqArr_m2, qArr, dqArr, uArr, duArr)
        chiSqRed = chiSq/dof
        AIC = 2.0*nFree - 2.0 * lnLike
        AICc = 2.0*nFree*(nFree+1)/(nData-nFree-1) - 2.0 * lnLike
        BIC = nFree * np.log(nData) - 2.0 * lnLike
        
        # Summary of run
        print("")
        print("-"*80)
        print("SUMMARY OF SAMPLING RUN:")
        print("#-PROCESSORS  = %d" % mpiSize)
        print("RUN-TIME      = %.2f" % (endTime-startTime))
        print("DOF           = %d" % dof)
        print("CHISQ:        = %.3g" % chiSq)
        print("CHISQ RED     = %.3g" % chiSqRed)
        print("AIC:          = %.3g" % AIC)
        print("AICc          = %.3g" % AICc)
        print("BIC           = %.3g" % BIC)
        print("ln(EVIDENCE)  = %.3g" % lnEvidence)
        print("dLn(EVIDENCE) = %.3g" % dLnEvidence)
        print("")
        print("-"*80)
        print("RESULTS:\n")
        for i in range(len(p)):
            print("%s = %.4g (+%3g, -%3g)" % \
                  (parNames[i], p[i], errPlus[i], errMinus[i]))
        print("-"*80)
        print("")

        # Create a save dictionary and store final p in values
        outFile = prefixOut + "_m%d_nest.json" % modelNum
        IfitDict["p"] = toscalar(IfitDict["p"].tolist())
        saveDict = {"parNames":      toscalar(parNames),
                    "labels":        toscalar(labels),
                    "values":        toscalar(p),
                    "errPlus":       toscalar(errPlus),
                    "errMinus":      toscalar(errMinus),
                    "bounds":        toscalar(bounds),
                    "priorTypes":    toscalar(priorTypes),
                    "wraps":         toscalar(wraps),
                    "dof":           toscalar(dof),
                    "chiSq":         toscalar(chiSq),
                    "chiSqRed":      toscalar(chiSqRed),
                    "AIC":           toscalar(AIC),
                    "AICc":          toscalar(AICc),
                    "BIC":           toscalar(BIC),
                    "ln(EVIDENCE) ": toscalar(lnEvidence),
                    "dLn(EVIDENCE)": toscalar(dLnEvidence),
                    "nFree":         toscalar(nFree),
                    "Imodel":        toscalar(IfitDict["p"]),
                    "IfitChiSq":     toscalar(IfitDict["chiSq"]),
                    "IfitChiSqRed":  toscalar(IfitDict["chiSqRed"]),
                    "IfitPolyOrd":   toscalar(IfitDict["polyOrd"]),
                    "Ifitfreq0":     toscalar(IfitDict["reference_frequency_Hz"])
                    }
        json.dump(saveDict, open(outFile, "w"))
        outFile = prefixOut + "_m%d_nest.dat" % modelNum
        FH = open(outFile, "w")
        for k, v in saveDict.items():
            FH.write("%s=%s\n" % (k, v))
        FH.close()
        print("Results saved in JSON and .dat format to:\n '%s'\n" % outFile)


        # Plot the posterior samples in a corner plot
        chains =  aObj.get_equal_weighted_posterior()
        chains = wrap_chains(chains, wraps, bounds, p)[:, :nDim]
        iFixed = [i for i, e in enumerate(fixedMsk) if e==0]
        chains = np.delete(chains, iFixed, 1)
        for i in sorted(iFixed, reverse=True):
            del(labels[i])
            del(p[i])

        cornerFig = corner.corner(xs      = chains,
                                  labels  = labels,
                                  range   = [(l,u) for l,u in zip(upper,lower)],
                                  truths  = p,
                                  quantiles = [0.1572, 0.8427],
                                  bins    = 30)

        # Save the posterior chains to ASCII file
        if verbose: print("Saving the posterior chains to ASCII file.")
        outFile = prefixOut + "_m%d_posteriorChains.dat" % modelNum
        if verbose: print("> %s" % outFile)
        np.savetxt(outFile, chains)

        # Plot the data and best-fitting model
        lamSqHirArr_m2 =  np.linspace(lamSqArr_m2[0], lamSqArr_m2[-1], 10000)
        freqHirArr_Hz = C / np.sqrt(lamSqHirArr_m2)
        IModArr = calculate_StokesI_model(IfitDict,freqHirArr_Hz)
        pDict = {k:v for k, v in zip(parNames, p)}
        quModArr = model(pDict, lamSqHirArr_m2)
        model_dict = {
            'chains': chains,
            'model': model,
            'parNames': parNames,
            'values': values
        }
        specFig.clf()
        plot_Ipqu_spectra_fig(freqArr_Hz     = freqArr_Hz,
                              IArr           = IArr,
                              qArr           = qArr,
                              uArr           = uArr,
                              dIArr          = dIArr,
                              dqArr          = dqArr,
                              duArr          = duArr,
                              freqHirArr_Hz  = freqHirArr_Hz,
                              IModArr        = IModArr,
                              qModArr        = quModArr.real,
                              uModArr        = quModArr.imag,
                              model_dict     = model_dict,
                              fig            = specFig)
        specFig.canvas.draw()

        # Save the figures
        outFile = prefixOut + "fig_m%d_specfit.pdf" % modelNum
        specFig.savefig(outFile)
        print("Plot of best-fitting model saved to:\n '%s'\n" % outFile)
        outFile = prefixOut + "fig_m%d_corner.pdf" % modelNum
        cornerFig.savefig(outFile)
        print("Plot of posterior samples saved to \n '%s'\n" % outFile)

        # Display the figures
        if showPlots:
            plt.show()
            #cornerFig.show()

        # Clean up

    # Clean up MPI environment
    if mpiSwitch:
        MPI.Finalize()


#-----------------------------------------------------------------------------#
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
                sigma = (bMax - bMin)/2.0
                mu = bMin + sigma
                p[i] = mu + sigma * ndtri(p[i])
            elif priorTypes[i] == "fixed":
                p[i] = values[i]
            else: # uniform (linear)
                bMin, bMax = bounds[i]
                p[i] = bMin + p[i] * (bMax - bMin)
        return p

    return prior


#-----------------------------------------------------------------------------#
def lnlike_call(parNames, lamSqArr_m2, qArr, dqArr, uArr, duArr):
    """ Returns a function to evaluate the log-likelihood """

    def lnlike(p, nDim, nParms):

        # Evaluate the model and calculate the joint ln(like)
        # Silva 2006
        pDict = {k:v for k, v in zip(parNames, p)}
        quMod = model(pDict, lamSqArr_m2)
        dquArr = np.sqrt(np.power(dqArr, 2) + np.power(duArr, 2))
        chiSqQ = np.nansum(np.power((qArr-quMod.real)/dqArr, 2))
        chiSqU = np.nansum(np.power((uArr-quMod.imag)/dqArr, 2))
        nData = len(dquArr)
        logLike = (-nData * np.log(2.0*np.pi)
                   -2.0 * np.nansum(np.log(dquArr))
                   -chiSqQ/2.0 -chiSqU/2.0)

        return logLike

    return lnlike


#-----------------------------------------------------------------------------#
def chisq_model(parNames, p, lamSqArr_m2, qArr, dqArr, uArr, duArr):
    """Calculate the chi^2 for the current model, given the data."""

    # Evaluate the model and calculate chisq
    pDict = {k:v for k, v in zip(parNames, p)}
    quMod = model(pDict, lamSqArr_m2)
    chisq = np.nansum( np.power((qArr-quMod.real)/dqArr, 2) +
                       np.power((uArr-quMod.imag)/duArr, 2))

    return chisq


#-----------------------------------------------------------------------------#
def wrap_chains(chains, wraps, bounds, p, verbose=False):

    # Get the indices of the periodic parameters
    iWrap = [i for i, e in enumerate(wraps) if e != 0]

    # Loop through the chains for periodic parameters
    for i in iWrap:
        wrapLow = bounds[i][0]
        wrapHigh = bounds[i][1]
        rng = wrapHigh - wrapLow

        # Shift the wrapping to centre on the best fit value
        wrapCent = wrapLow + (wrapHigh - wrapLow)/2.0
        wrapLow += (p[i] - wrapCent)
        wrapHigh += (p[i] - wrapCent)
        chains[:, i] = ((chains[:, i]-wrapLow) % rng) + wrapLow
        if verbose:
            print("Wrapped parameter '%d' in range [%s, %s] ..." %
                  (i, wrapLow, wrapHigh))

    return chains


#-----------------------------------------------------------------------------#
def init_mnest():
    """Initialise MultiNest arguments"""

    argsDict = {'LogLikelihood':              '',
                'Prior':                      '',
                'n_dims':                     0,
                'n_params':                   0,
                'n_clustering_params':        0,
                'wrapped_params':             None,
                'importance_nested_sampling': False,
                'multimodal':                 False,
                'const_efficiency_mode':      False,
                'n_live_points':              100,
                'evidence_tolerance':         0.5,
                'sampling_efficiency':        'model',
                'n_iter_before_update':       500,
                'null_log_evidence':          -1.e90,
                'max_modes':                  100,
                'mode_tolerance':             -1.e90,
                'outputfiles_basename':       '',
                'seed':                       -1,
                'verbose':                    True,
                'resume':                     True,
                'context':                    0,
                'write_output':               True,
                'log_zero':                   -1.e100,
                'max_iter':                   0,
                'init_MPI':                   False,
                'dump_callback':              None}
    return argsDict


#-----------------------------------------------------------------------------#
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
