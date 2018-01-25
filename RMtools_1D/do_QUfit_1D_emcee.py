#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_QUfit_1D.py                                                    #
#                                                                             #
# PURPOSE:  Code to simultaneously fit Stokes I, Q and U spectra with a suite #
#           of Faraday active models.                                         #
#                                                                             #
# MODIFIED: 03-Oct-2017 by C. Purcell                                         #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#   main           ... parse command line arguments and initiate procedure    #
#   inParmClass    ... class containing the control parameters for MCMC       #
#   lnlike_priors  ... calculate the ln(likelihood) for gaussian priors       #
#   lnlike_bounds  ... calculate the ln(likelihood) for uniform priors        #
#   lnlike_model   ... calculate the ln(likelihood for the model              #
#   lnlike_total   ... calculate the total ln(likelihood)                     #
#   plot_trace     ... plot the chains versus step (time)                     #
#   plot_like_stats .. plot the likelihood statistics                         #
#   plot_trace_stats  plot the statistics of the MCMC chains                  #
#   gelman_rubin   ... calculate the Gelman Rubin statistic and others        #
#   unwrap_lines   ... unwrap a line in a periodic plot                       #
#   wrap_arr       ... wrap the periodic values in an array                   #
#   wrap_chains    ... wrap all if the walker chains, shifting if necessary   #
#   chk_trace_stable . check if a binned time series has flattened            #
#   run_qufit      ... main function of the QU-fitting procedure              #
#                                                                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2016 Cormac R. Purcell                                        #
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

import sys
import os
import copy
import time
import imp
import argparse
import traceback
import math as m
import numpy as np
import numpy.ma as ma
import scipy.optimize as op
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from RMutils import emcee
from RMutils.util_misc import create_frac_spectra
from RMutils.util_misc import poly5
from RMutils.util_plotTk import plot_Ipqu_spectra_fig
from RMutils.util_plotTk import CustomNavbar
from RMutils.util_plotTk import tweakAxFormat

C = 2.997924538e8 # Speed of light [m/s]


#-----------------------------------------------------------------------------#
def main():
    """
    Start the run_qufit procedure if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataFile", metavar="dataFile.dat", nargs=1,
                        help="ASCII file containing Stokes spectra & errors.")
    parser.add_argument("-m", dest="modelNum", type=int, default=1,
                        help="model number to fit [1].")
    parser.add_argument("-w", dest="nWalkers", type=int, default=200,
                        help="number of EMCEE parallel walkers [200].")
    parser.add_argument("-t", dest="nThreads", type=int, default=3,
                        help="number of parallel threads to execute [2].")    
    parser.add_argument("-o", dest="polyOrd", type=int, default=2,
                        help="polynomial order to fit to I spectrum [2].")
    parser.add_argument("-i", dest="noStokesI", action="store_true",
                        help="ignore the Stokes I spectrum [False].")
    parser.add_argument("-p", dest="showPlots", action="store_true",
                        help="show the plots [False].")
    parser.add_argument("-d", dest="debug", action="store_true",
                        help="turn on debugging messages/plots [False].")
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.dataFile[0]):
        print "File does not exist: '%s'." % args.dataFile[0]
        sys.exit()
    dataDir, dummy = os.path.split(args.dataFile[0])

    # Run the QU-fitting procedure
    run_qufit(dataFile     = args.dataFile[0],
              modelNum     = args.modelNum,
              polyOrd      = args.polyOrd,
              nWalkers     = args.nWalkers,
              nThreads     = args.nThreads,
              outDir       = dataDir,
              noStokesI    = args.noStokesI,
              nBits        = 32,
              showPlots    = args.showPlots,
              debug        = args.debug)

#-----------------------------------------------------------------------------#
class inParmClass():

    # Constructor method contains input parameters
    def __init__(self, inParms, runParmDict):

        # Input parameters
        self.inParms = inParms
        self.runParmDict = runParmDict
        
        # Set properties of currently selected model
        self.nParms = len(self.inParms)
        self.fxi = [i for i in range(self.nParms) 
                    if not self.inParms[i]['fixed'] ]
        self.nDim = len(self.fxi)
        self.runMode = self.runParmDict.get("runMode", "auto")
        self.maxSteps = self.runParmDict.get("maxSteps", 2000)
        self.nExploreSteps = self.runParmDict.get("nExploreSteps", 500)
        self.nPollSteps = self.runParmDict.get("nPollSteps", 10)
        self.nStableCycles = self.runParmDict.get("nStableCycles", 40)
        self.likeStdLim = self.runParmDict.get("likeStdLim", 1.2)
        self.likeMedLim = self.runParmDict.get("likeMedLim", 0.3)
        self.parmStdLim = self.runParmDict.get("parmStdLim", 1.2)
        self.parmMedLim = self.runParmDict.get("parmMedLim", 0.3)
        self.nSteps = self.nStableCycles * self.nPollSteps

    def seed_walkers(self, nWalkers):
        """Create initial walker vectors according to the seed ranges."""
        
        rngLst = [self.inParms[self.fxi[j]]['seedrng'][1] -
                  self.inParms[self.fxi[j]]['seedrng'][0]
                  for j in range(self.nDim)]
        lowLst = [self.inParms[self.fxi[j]]['seedrng'][0] 
                  for j in range(self.nDim)]
        p0 = [lowLst + rngLst * np.random.rand(self.nDim) 
              for i in range(nWalkers)]

        return p0

    def update_values(self, walker):
        """Update the values of the inparms given a walker vector."""
        
        j = 0
        for i in range(len(self.inParms)):
            if not self.inParms[i]['fixed']:
                self.inParms[i]['value'] = walker[j]
                j += 1
    

#-----------------------------------------------------------------------------#
def lnlike_priors(walker, inParms):
    """Calculate ln(L) for priors assuming Gaussian errors."""

    chiSqLst = []
    j = 0
    for i in range(len(inParms)):
        if not inParms[i]['fixed']:
            
            # Priors are inParms with a value and error 
            if inParms[i]['error']>0.0:
                chiSqNrm = ((walker[j]-inParms[i]['value'])**2.0
                          / inParms[i]['error']**2.0 +
                          2*m.pi*inParms[i]['error']**2.0)
                chiSqLst.append(lnLike)
            else:
                chiSqLst.append(0.0)

            # Increment to the next walker (free parameter)
            j +=1   

    return -np.sum(chiSqLst)/2.0


#-----------------------------------------------------------------------------#
def lnlike_bounds(walker, inParms):
    """Impose boundaries on the likelihood space by setting an infinitely high
    lnLike outside of set ranges."""
    
    chiSqLst = []
    j = 0
    for i in range(len(inParms)):
        if not inParms[i]['fixed']:
            
            # If walker vector is outside bounds, lnLike = +inf
            if (walker[j]<inParms[i]['bounds'][0] or
                walker[j]>inParms[i]['bounds'][1]):
                chiSqLst.append(np.inf)
            else:
                chiSqLst.append(0.0)
                
            # Increment to the next walker (free parameter)
            j +=1   
                
    return -np.sum(chiSqLst)/2.0
    

#-----------------------------------------------------------------------------#
def lnlike_model(inParms, lamSqArr_m2, qArr, dqArr, uArr, duArr):
    """Calculate the ln(likelihood) for the current model, given the data."""
    
    # Calculate the model
    quMod = model(inParms, lamSqArr_m2)
    
    # Calculate the ln(Like) for the model assuming a normal distribution
    dquArr = np.sqrt(np.power(dqArr, 2) + np.power(duArr, 2))
    chiSqNrm = np.nansum( np.power((qArr-quMod.real)/dqArr, 2) +
                          np.power((uArr-quMod.imag)/duArr, 2) +
                          np.log(2 * np.pi * np.power(dquArr, 2)) )
    
    return -chiSqNrm/2.0


#-----------------------------------------------------------------------------#
def chisq_model(inParms, lamSqArr_m2, qArr, dqArr, uArr,
                 duArr):
    """Calculate the chi^2 for the current model, given the data."""
    
    # Calculate the model
    quMod = model(inParms, lamSqArr_m2)
    
    # Calculate the chi^2 for the model assuming a normal distribution
    chisq = np.nansum( np.power((qArr-quMod.real)/dqArr, 2) +
                        np.power((uArr-quMod.imag)/duArr, 2))

    return chisq


#-----------------------------------------------------------------------------#
def lnlike_total(walker, ip, lamSqArr_m2, qArr, dqArr, uArr,
                 duArr):
    """Calculate the total ln(likelihood), including priors."""

    # Update free parameters in ip.inParms to the latest walker values
    ip.update_values(walker)

    # Calculate the chi-sq for priors, model and bounds
    lnLikePriors = lnlike_priors(walker, ip.inParms)
    lnLikeBounds = lnlike_bounds(walker, ip.inParms)
    lnLikeModel = lnlike_model(ip.inParms, lamSqArr_m2,
                               qArr, dqArr, uArr, duArr)
    
    return lnLikePriors + lnLikeBounds + lnLikeModel


#-----------------------------------------------------------------------------#
def plot_trace(sampler, inParms, chop=None, fig=None, title="Walker Chains"):
    """Plot the MCMC sampler chains coloured by ln(likelihood)."""

    fxi = [i for i in range(len(inParms)) if not inParms[i]['fixed'] ]
    nDim = len(fxi)
    nSteps = sampler.chain.shape[1]
    
    # Initialise the figure
    if fig is None:
        fig=plt.figure(figsize=(8, 8))
    else:
        fig.clear()

    # Plot a trace for each parameter
    for j in range(nDim):
        chain = sampler.chain[:,:,j].transpose()
        like = sampler.lnprobability.transpose()        
	ax = fig.add_subplot(nDim, 1, j+1)
        stepArr = np.arange(chain.shape[0], dtype="f4")+1
	#ax.plot(chain,'k', alpha = 0.3)
        for i in range(chain.shape[1]):
            ax.scatter(x=stepArr, y=chain[:,i], c=like[:,i],
                       cmap=plt.cm.jet, marker="D", edgecolor='none',
                       alpha=0.2, s=4)
        
        ax.set_ylabel(inParms[fxi[j]]['label'], rotation=0, fontsize=12)
        ax.yaxis.set_label_coords(-0.3, 0.5)
        if j==0:
            ax.set_title(title)
        if j<nDim-1:
            [label.set_visible(False) for label in ax.get_xticklabels()]
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        yRange = np.max(chain) - np.min(chain)
        yMinPlt = np.min(chain) - yRange*0.1
        yMaxPlt = np.max(chain) + yRange*0.1
        ax.set_ylim(yMinPlt, yMaxPlt)
        ax.set_xlim(np.min(stepArr)-nSteps*0.01, np.max(stepArr)+nSteps*0.01)
        ax.autoscale_view(True,True,True)
    ax.set_xlabel('Steps', fontsize=15)
    fig.subplots_adjust(left=0.30, bottom=0.07, right=0.97, top=0.94)
    fig.canvas.draw()
    fig.show()


#-----------------------------------------------------------------------------#
def plot_like_stats(statDict, fig=None, nSteps=None,
                    title="Likelihood Trace & Statistics"):
    """Plot the binned Likelihood trace and statistics."""
    
    # Initialise the figure
    if fig is None:
        fig = plt.figure(figsize=(8, 10))
    else:
        fig.clear()
    
    # Plot the median likelihood
    ax1 = fig.add_subplot(311)
    [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax1.step(x=statDict["stepBin"], y=statDict["medBin"],
             marker="None", where="mid", color="k")
    ax1.set_ylabel("$\\langle$ln(like)$\\rangle$", fontsize=12)
    ax1.autoscale_view(True,True,True)
        
    # Plot stdev(likelihood)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.step(x=statDict["stepBin"], y=statDict["stdBin"],
             marker="None", where="mid", color="r")
    ax2.set_ylabel("$\\sigma$ ln(like)", fontsize=12)
    ax2.autoscale_view(True,True,True)

    # Plot the rolling stdev and median metrics
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.step(x=statDict["stepBin"], y=statDict["stat1"],
             marker="None", where="mid", color="r",
             label="$\\sigma$ stability")
    ax3.step(x=statDict["stepBin"], y=statDict["stat2"],
             marker="None", where="mid", color="b",
             label="Median stability")
    ax3.set_ylabel("Stability", fontsize=12)
    ax3.autoscale_view(True,True,True)
    
    # Final formatting
    ax3.set_xlabel('Steps')
    ax3 = tweakAxFormat(ax3,loc='upper left', bbox_to_anchor=(0.00, 1.00))
    fig.subplots_adjust(left=0.18, bottom=0.08, right=0.97, top=0.94)
    fig.canvas.draw()
    fig.show()

    
#-----------------------------------------------------------------------------#
def plot_trace_stats(sampler, inParms, figLst=[], nSteps=None,
                     title="Chains & Statistics", statLst=[]):
    """For each free parameter create a plot showng the MCMC sampler chains 
    coloured by ln(likelihood), the binned median & stdev likelihood, and the
    Gelman-Rubin statistics."""

    fxi = [i for i in range(len(inParms)) if not inParms[i]['fixed'] ]
    nDim = len(fxi)
    if nSteps is None:
        nSteps = sampler.chain.shape[1]

    # How many bins equivalent to nSteps?
    nBins = np.where(np.array(statLst[0]["stepBin"])>=nSteps)[0][0]

    # Initialise the figure clear the current windows
    if figLst==[]:
        for j in range(nDim):
            figLst.append(plt.figure(figsize=(8, 12)))

    # Insert the axes on first call
    if figLst[0].axes==[]:
        for j in range(nDim):
            figLst[j].add_subplot(411)
            figLst[j].add_subplot(412, sharex=figLst[j].axes[0])
            figLst[j].add_subplot(413, sharex=figLst[j].axes[0])
            figLst[j].add_subplot(414, sharex=figLst[j].axes[0])
    else:
        for j in range(nDim):
            figLst[j].axes[0].cla()
            figLst[j].axes[1].cla()
            figLst[j].axes[2].cla()
            figLst[j].axes[3].cla()
            
    # Plot a trace and statistics for each parameter
    for j in range(nDim):
        chain = sampler.chain[:,:,j].transpose()
        like = sampler.lnprobability.transpose()
        stepArr = np.arange(chain.shape[0], dtype="f4")+1

        # Plot the chain trace
        ax1 = figLst[j].axes[0]
        for i in range(chain.shape[1]):
            ax1.scatter(x=stepArr, y=chain[:,i], c=like[:,i],
                        cmap=plt.cm.jet, marker="D", edgecolor='none',
                        alpha=0.2, s=4)        
        ax1.set_ylabel(inParms[fxi[j]]['label'], rotation=90)
        ax1.set_title(title)
        [label.set_visible(False) for label in ax1.get_xticklabels()]
        ax1.yaxis.set_major_locator(MaxNLocator(5))

        # Plot the median of the trace
        if statLst!=[]:
            ax1.step(x=statLst[j]["stepBin"], y=statLst[j]["medAll"],
                     marker="None", where="mid", color="grey",
                     linewidth=1.5)
            ax1.errorbar(x=statLst[j]["stepBin"], y=statLst[j]["medAll"],
                         yerr=statLst[j]["stdAll"], ecolor="grey",
                         fmt="none", capsize=0, elinewidth=1.5)
            
        # Force the y-scaling
        yRange = np.max(chain[-nSteps:,:]) - np.min(chain[-nSteps:,:])
        yMinPlt = np.min(chain[-nSteps:,:]) - yRange*0.1
        yMaxPlt = np.max(chain[-nSteps:,:]) + yRange*0.1
        ax1.set_ylim(yMinPlt, yMaxPlt)
        
        # Plot the Stdev
        ax2 = figLst[j].axes[1]
        [label.set_visible(False) for label in ax2.get_xticklabels()]
        ax2.step(x=statLst[j]["stepBin"], y=statLst[j]["stdAll"], marker="s",
                 where="mid")
        ax2.scatter(x=statLst[j]["stepBin"], y=np.sqrt(statLst[j]["W"]),
                    marker="D", color="r")
        ax2.scatter(x=statLst[j]["stepBin"], y=np.sqrt(statLst[j]["B"]),
                    marker="x", s=180, linewidth=1.5, edgecolor="g",
                    facecolor="g", zorder=10)
        ax2.set_ylabel("Standard Deviation", rotation=90)
        ax2.yaxis.set_major_locator(MaxNLocator(5))
        
        # Force the y-scaling
        yMax = [np.max(statLst[j]["stdAll"][-nBins:])]
        yMin = [np.min(statLst[j]["stdAll"][-nBins:])]
        yMax.append(np.max(np.sqrt(statLst[j]["W"][-nBins:])))
        yMin.append(np.min(np.sqrt(statLst[j]["W"][-nBins:])))
        yMax.append(np.max(np.sqrt(statLst[j]["B"][-nBins:])))
        yMin.append(np.min(np.sqrt(statLst[j]["B"][-nBins:])))
        yRange = np.max(yMax) - np.min(yMin)
        yMinPlt = np.min(yMin) - yRange*0.1
        yMaxPlt = np.max(yMax) + yRange*0.1
        ax2.set_ylim(yMinPlt, yMaxPlt)
        
        # Plot the Gelman Rubin statistic
        ax3 = figLst[j].axes[2]
        [label.set_visible(False) for label in ax3.get_xticklabels()]
        ax3.step(x=statLst[j]["stepBin"], y= statLst[j]["R"], marker="None",
                 where="mid", color="r") 
        ax3.set_ylabel("Gelman-Rubin", rotation=90)
        ax3.yaxis.set_major_locator(MaxNLocator(5))
        ax3.autoscale_view(True,True,True)

        # Plot the stability traces
        ax4 = figLst[j].axes[3]
        ax4.step(x=statLst[j]["stepBin"], y=statLst[j]["stat1"],
                 marker="None", where="mid", color="r",
                 label="$\\sigma$ stability")
        ax4.step(x=statLst[j]["stepBin"], y=statLst[j]["stat2"],
                 marker="None", where="mid", color="b",
                 label="Median stability")
        ax4.set_ylabel("Stability", fontsize=12)
        ax4.autoscale_view(True,True,True)
        
        # Final formatting
        ax4.set_xlabel('Steps')        
        #ax4.set_xlim(sampler.chain.shape[1]-nSteps, sampler.chain.shape[1])
        ax1.axvline(sampler.chain.shape[1]-nSteps, color='grey', linewidth=2)
        ax2.axvline(sampler.chain.shape[1]-nSteps, color='grey', linewidth=2)
        ax3.axvline(sampler.chain.shape[1]-nSteps, color='grey', linewidth=2)
        ax4.axvline(sampler.chain.shape[1]-nSteps, color='grey', linewidth=2)
        
        figLst[j].subplots_adjust(left=0.18, bottom=0.08, right=0.97, top=0.94)
        figLst[j].canvas.draw()
        figLst[j].show()

        
#-----------------------------------------------------------------------------#
def gelman_rubin(chain):
    
    # Transpose from default EMCEE shape
    chain = chain.transpose()
        
    # Calculate values from all samples in the chain
    nStep = int(chain.shape[0])
    nChain = int(chain.shape[1])
    medAll = np.median(chain)
    stdAll = np.std(chain)
    
    # Calculate values per-chain
    meanEachChain = np.mean(chain, axis=0)
    stdEachChain = np.std(chain, ddof=1.0, axis=0)
    varEachChain = np.var(chain, ddof=1.0, axis=0)
    
    # Variance between the mean of each chain
    B = np.var(meanEachChain, ddof=1.0)

    # Mean variance of the chains in the window
    W = np.mean(varEachChain[-nChain:])
    
    # Estimate of variance
    s2 = W * (nStep - 1) / nStep + B
    V = s2 + B / nChain
    R =  V / W

    # Return dictionary
    mDict = {"nStep":         nStep,
             "nChain":        nChain,
             "medAll":        medAll,
             "stdAll":        stdAll,
             "meanEachChain": meanEachChain,
             "stdEachChain":  stdEachChain,
             "varEachChain":  varEachChain,
             "B":             B,
             "W":             W,
             "s2":            s2,
             "V":             V,
             "R":             R}

    return mDict


#-----------------------------------------------------------------------------#
def unwrap_lines(dat, lims=[-90.0, 90.0], thresh = 0.95):
    
    jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))
    

#-----------------------------------------------------------------------------#
def wrap_arr(arr, wrapLow=-90.0, wrapHigh=90.0):
    """Wrap the values in an array (e.g., angles)."""
    
    rng = wrapHigh - wrapLow
    arr = ((arr-wrapLow) % rng) + wrapLow
    return arr


#-----------------------------------------------------------------------------#
def wrap_chains(inParms, sampler, pos=None, shift=False, verbose=False):
    """Wrap the walker chains if the key 'fixed' is set True in inParms. The
    argument shifts the wrap values to centre on the median chain position.
    This is necessary to allow stop the final sampler chains running into
    the edge if the max(L) value is near the wrap limit."""
    
    # Get the indices of the free parameters (not fixed)
    nWalkers, nSteps, nDim = sampler.chain.shape
    fxi = [i for i in range(len(inParms)) if not inParms[i]['fixed'] ]

    # Loop through the free parameters
    for j in range(nDim):
        if "wrap" in inParms[fxi[j]]:
            
            wrapLow = inParms[fxi[j]]["wrap"][0]
            wrapHigh = inParms[fxi[j]]["wrap"][1]

            # Shift the wrapping to centre on the chains
            if pos is not None and shift==True:
                wrapCent = wrapLow + (wrapHigh - wrapLow)/2.0
                med = np.median(pos[:,j])
                wrapLow += (med - wrapCent)
                wrapHigh += (med - wrapCent)
            if verbose:
                print "> Wrapping parameter '%s' in range [%s, %s] ..." % \
                    (inParms[fxi[j]]["parname"], wrapLow, wrapHigh),
            for i in range(nSteps):
                sampler.chain[:,i,j] = wrap_arr(sampler.chain[:,i,j],
                                                wrapLow, wrapHigh)
            if pos is not None:
                pos[:,j] = wrap_arr(pos[:,j], wrapLow, wrapHigh)
            if verbose:
                print "done."
            
    if pos is None:
        return sampler
    else:
        return sampler, pos

    
#-----------------------------------------------------------------------------#
def chk_trace_stable(statDict, nCycles, stdLim=1.1, medLim=0.3):
    """Check that the statistics of a MCMC trace have stabilised. This is used
    to assess convergence. WARNING: stability does not necessarilly mean the
    chain has converged if it is stuck in a meta-stable state."""
    
    status = False
    stdMax = np.nan
    stdMed = np.nan
    stat1 = np.nan
    stat2 = np.nan
    if len(statDict["stepBin"])>=nCycles:        
        stdArr = np.array(statDict["stdBin"][-nCycles:], dtype="f8")
        medArr = np.array(statDict["medBin"][-nCycles:], dtype="f8")        
        stdMax = np.max(stdArr)
        stdMed = np.median(stdArr)
        medMed = np.median(medArr)
        diffArr = np.abs(medArr-medMed)

        # Condition 1: Standard deviation has stabilised within some limit
        stat1 = stdMax/(stdMed*stdLim)
        c1 = stdMax/stdMed<stdLim
        
        # Condition 2: the median has stabilised within some limit
        stat2 = np.max(diffArr)/(stdMed*stdLim)
        c2 = np.max(diffArr)<stdMed*medLim
        if c1 and c2:
            status = True

    return status, stat1, stat2 


#-----------------------------------------------------------------------------#
def run_qufit(dataFile, modelNum, nWalkers=200, nThreads=2, outDir="",
              polyOrd=3, nBits=32, noStokesI=False, showPlots=False,
              debug=False):
    """Root function controlling the fitting porcedure."""
    
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)
    
    # Output prefix is derived from the input file name
    prefixOut, ext = os.path.splitext(dataFile)
    
    # Read the data-file. Format=space-delimited, comments='#'.
    print "Reading the data file '%s':" % dataFile
    # freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy
    try:
        print "> Trying [freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy]",
        (freqArr_Hz, IArr_Jy, QArr_Jy, UArr_Jy,
         dIArr_Jy, dQArr_Jy, dUArr_Jy) = \
         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
        print "... success."
    except Exception:
        print "...failed."
        # freq_Hz, Q_Jy, U_Jy, dQ_Jy, dU_Jy
        try:
            print "Reading [freq_Hz, Q_Jy, U_Jy,  dQ_Jy, dU_Jy]",
            (freqArr_Hz, QArr_Jy, UArr_Jy, dQArr_Jy, dUArr_Jy) = \
                         np.loadtxt(dataFile, unpack=True, dtype=dtFloat)
            print "... success."
            noStokesI = True
        except Exception:
            print "...failed."
            if debug:
                print traceback.format_exc()
            sys.exit()
    
    # If no Stokes I present, create a dummy spectrum = unity
    if noStokesI:
        print "Warn: no Stokes I data in use."
        IArr_Jy = np.ones_like(QArr_Jy)
        dIArr_Jy = np.zeros_like(QArr_Jy)
        
    # Convert to GHz and mJy for convenience
    print "Successfully read in the Stokes spectra."
    freqArr_GHz = freqArr_Hz / 1e9
    lamSqArr_m2 = np.power(C/freqArr_Hz, 2.0)
    IArr_mJy = IArr_Jy * 1e3
    QArr_mJy = QArr_Jy * 1e3
    UArr_mJy = UArr_Jy * 1e3
    dIArr_mJy = dIArr_Jy * 1e3
    dQArr_mJy = dQArr_Jy * 1e3
    dUArr_mJy = dUArr_Jy * 1e3

    # Fit the Stokes I spectrum and create the fractional spectra
    IModArr, qArr, uArr, dqArr, duArr, IfitDict = \
             create_frac_spectra(freqArr=freqArr_GHz,
                                 IArr=IArr_mJy,
                                 QArr=QArr_mJy,
                                 UArr=UArr_mJy,
                                 dIArr=dIArr_mJy,
                                 dQArr=dQArr_mJy,
                                 dUArr=dUArr_mJy,
                                 polyOrd=polyOrd,
                                 verbose=True)
    
    # Plot the data and the Stokes I model fit
    if showPlots:
        print "Plotting the input data and spectral index fit."
        freqHirArr_Hz =  np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)     
        IModHirArr_mJy = poly5(IfitDict["p"])(freqHirArr_Hz/1e9)    
        specFig = plt.figure(figsize=(12, 8))
        plot_Ipqu_spectra_fig(freqArr_Hz     = freqArr_Hz,
                              IArr_mJy       = IArr_mJy, 
                              qArr           = qArr, 
                              uArr           = uArr, 
                              dIArr_mJy      = dIArr_mJy,
                              dqArr          = dqArr,
                              duArr          = duArr,
                              freqHirArr_Hz  = freqHirArr_Hz,
                              IModArr_mJy    = IModHirArr_mJy,
                              fig            = specFig)
        
        # Use the custom navigation toolbar
        try:
            specFig.canvas.toolbar.pack_forget()
            CustomNavbar(specFig.canvas, specFig.canvas.toolbar.window)
        except Exception:
            pass

        # Display the figure
        specFig.canvas.draw()
        specFig.show()

    #-------------------------------------------------------------------------#

    # Load the model and parameters from the relevant file
    print "\nLoading the model from file 'models/m%d.py' ..."  % modelNum
    mod = imp.load_source("m%d" % modelNum, "models/m%d.py" % modelNum)
    global model
    model = mod.model
    
    # Select the inputs to the chosen model by creating an instance of
    # inParmClass. Seed walker vectors based on the preset seed-range.
    ip = inParmClass(mod.inParms, mod.runParmDict)
    p0 = ip.seed_walkers(nWalkers)

    # Call the lnlike_total function to test it works OK
    print "> Calling ln(likelihood) as a test: L = ",
    L = lnlike_total(p0[0], ip, lamSqArr_m2,
                     qArr, dqArr, uArr, duArr)
    print L
    if np.isnan(L):
        print "> Err: ln(likelihood) function returned NaN."
        sys.exit()

    # Define an MCMC sampler object. 3rd argument is ln(likelihood) function
    # and 4th is a list of additional arguments to lnlike() after walker.
    sampler = emcee.EnsembleSampler(nWalkers,
                                    ip.nDim,
                                    lnlike_total,
                                    args=[ip,
                                          lamSqArr_m2,
                                          qArr,
                                          dqArr,
                                          uArr,
                                          duArr],
                                    threads=nThreads)
    
    # Initialise the trace figure
    if showPlots:
        chainFigLst = []
        for i in range(ip.nDim):
            chainFigLst.append(plt.figure(figsize=(8, 8)))
        
    # Run the sampler to explore parameter space
    print 'Explore parameter space for %d steps ...' % ip.nExploreSteps,
    sys.stdout.flush()
    pos, prob, state = sampler.run_mcmc(p0, ip.nExploreSteps)
    print 'done.'
    
    # Reset the samplers to a small range around the max(likelihood)
    maxPos = pos[np.argmax(prob, 0)]
    pos = [maxPos + 1e-9 * np.random.rand(ip.nDim) for i in range(nWalkers)]
    
    # Plot the chains for the exploration step
    if showPlots:
        print 'Plotting the walker chains for the wide exploration step.'
        titleStr = "Exploring all likely parameter space."
        plot_trace(sampler, ip.inParms, title=titleStr)
    sampler.reset()
    
    # Initialise the structure for holding the binned statistics
    # List of (list of dictionaries)
    statLst = []
    for i in range(ip.nDim):
        statLst.append({"stepBin": [],
                        "medBin": [],
                        "stdBin": [],
                        "medAll": [],
                        "stdAll": [],
                        "B": [],
                        "W": [],
                        "R": [],
                        "stat1": [],
                        "stat2": []})
    likeStatDict = {"stepBin": [],
                    "medBin": [],
                    "stdBin": [],
                    "stat1": [],
                    "stat2": []}
    
    # Run the sampler, polling the statistics every nPollSteps
    print "Running the sampler and polling every %d steps:" % (ip.nPollSteps)
    if ip.runMode=="auto":
        print "> Will attempt to detect MCMC chain stability."
    print "Maximum steps set to %d." % ip.maxSteps
    print ""
    while True:
        convergeFlg = False
        convergeFlgLst = []
        print ".",
        sys.stdout.flush()
    
        # Run the sampler for nPollSteps
        pos, prob, state = sampler.run_mcmc(pos, ip.nPollSteps)
        
        # Perform wrapping if ip.inParms[n]['wrap'] is set.
        sampler, pos = wrap_chains(ip.inParms, sampler, pos, shift=True)
        
        # Measure the statistics of the binned likelihood
        stepBin = sampler.chain.shape[1] -(ip.nPollSteps/2.0)
        likeWin = sampler.lnprobability[:,-ip.nPollSteps:]
        likeStatDict["stepBin"].append(stepBin)
        likeStatDict["medBin"].append(np.median(likeWin))
        likeStatDict["stdBin"].append(np.std(likeWin))
            
        # Measure the statistics of the binned chains
        chainWin = sampler.chain[:,-ip.nPollSteps:,:]
        for i in range(ip.nDim):
            mDict = gelman_rubin(chainWin[:,:,i])
            statLst[i]["stepBin"].append(stepBin)
            statLst[i]["medBin"].append(np.median(chainWin[:,:,i]))
            statLst[i]["stdBin"].append(np.std(chainWin[:,:,i]))
            statLst[i]["medAll"].append(mDict["medAll"])
            statLst[i]["stdAll"].append(mDict["stdAll"])
            statLst[i]["B"].append(mDict["B"])
            statLst[i]["W"].append(mDict["W"])
            statLst[i]["R"].append(mDict["R"])
            
            # Check for convergence in each parameter trace
            convergeFlg, stat1, stat2 = \
                chk_trace_stable(statDict=statLst[i],
                                 nCycles=ip.nStableCycles,
                                 stdLim=ip.parmStdLim,
                                 medLim=ip.parmMedLim)
            convergeFlgLst.append(convergeFlg)
            statLst[i]["stat1"].append(stat1)
            statLst[i]["stat2"].append(stat2)

        # Check for convergence in the likelihood trace
        convergeFlg, stat1, stat2 = \
            chk_trace_stable(statDict=likeStatDict,
                             nCycles=ip.nStableCycles,
                             stdLim=ip.likeStdLim,
                             medLim=ip.likeMedLim)
        convergeFlgLst.append(convergeFlg)
        likeStatDict["stat1"].append(stat1)
        likeStatDict["stat2"].append(stat2)

        # If all traces have converged, continue
        if ip.runMode=="auto" and np.all(convergeFlgLst):
            print "\n>Stability threshold passed!"
            break

        # Continue at the upper step limit
        if sampler.chain.shape[1]>ip.maxSteps:
            print "\nMaximum number of steps performed."
            break

    # Plot the likelihood trace and statistics
    if debug:
        plot_like_stats(likeStatDict)
        if not showPlots:
            print "Press <RETURN> ...",
            raw_input()
        
    # Discard the burn-in section of the chain
    print "\nUsing the last %d steps to sample the posterior.\n" % ip.nSteps
    chainCut = sampler.chain[:,-ip.nSteps:,:]
    s = chainCut.shape
    flatChainCut = chainCut.reshape(s[0] * s[1], s[2])
    lnprobCut = sampler.lnprobability[-ip.nSteps:,:]
    flatLnprobCut = lnprobCut.flatten()
    
    # Plot the chains
    if showPlots:
        print 'Plotting the walker chains after polling ...'
        plot_trace_stats(sampler, ip.inParms, figLst=chainFigLst,
                         nSteps=ip.nSteps, statLst=statLst)
        
    # Determine the best-fit values from the 16th, 50th and 84th percentile
    # Marginalizing in MCMC is simple: select the axis of the parameter.
    # Update ip.inParms with the best-fitting values.
    pBest = []
    print
    print '-'*80
    print 'RESULTS:\n'
    for i in range(len(ip.fxi)):
        fChain = flatChainCut[:, i]
        g = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        best, errPlus, errMinus = g(np.percentile(fChain, [15.72, 50, 84.27]))
        pBest.append(best)
        ip.inParms[ip.fxi[i]]['value'] = best
        ip.inParms[ip.fxi[i]]['errPlus'] = errPlus
        ip.inParms[ip.fxi[i]]['errMinus'] = errMinus
        print '%s = %.4g (+%3g, -%3g)' % (ip.inParms[ip.fxi[i]]['parname'],
                                          best, errPlus, errMinus)
    
    # Calculate goodness-of-fit parameters
    nSamp = len(lamSqArr_m2)*2.0
    dof = nSamp - ip.nDim -1
    chiSq = chisq_model(ip.inParms, lamSqArr_m2, qArr, dqArr, uArr, duArr)
    chiSqRed = chiSq/dof

    # Calculate the information criteria
    lnLike = lnlike_model(ip.inParms, lamSqArr_m2, qArr, dqArr, uArr, duArr)
    AIC = 2.0*ip.nDim - 2.0 * lnLike
    AICc = 2.0*ip.nDim*(ip.nDim+1)/(nSamp-ip.nDim-1) - 2.0 * lnLike
    BIC = ip.nDim * np.log(nSamp) - 2.0 * lnLike
    print
    print "DOF:", dof
    print "CHISQ:", chiSq
    print "CHISQ RED:", chiSqRed
    print "AIC:", AIC
    print "AICc", AICc
    print "BIC", BIC
    print
    print '-'*80
    
    # Create a save dictionary
    saveObj = {"inParms": ip.inParms,
               "flatchain": flatChainCut,
               "flatlnprob": flatLnprobCut,
               "chain": chainCut,
               "lnprob": lnprobCut,
               "convergeFlg": np.all(convergeFlgLst),
               "dof": dof,
               "chiSq": chiSq, 
               "chiSqRed": chiSqRed, 
               "AIC": AIC, 
               "AICc": AICc, 
               "BIC": BIC, 
               "IfitDict": IfitDict}
    
    # Save the Markov chain and results to a Python Pickle
    outFile = prefixOut + "_MCMC.pkl"
    if os.path.exists(outFile):
        os.remove(outFile)
    fh = open(outFile, "wb")
    pkl.dump(saveObj, fh)
    fh.close()
    print "> Results and MCMC chains saved in pickle file '%s'" % outFile
    
    # Plot the results
    if showPlots:
        print "Plotting the best-fitting model."
        lamSqHirArr_m2 =  np.linspace(lamSqArr_m2[0], lamSqArr_m2[-1], 10000)
        freqHirArr_Hz = C / np.sqrt(lamSqHirArr_m2)
        IModArr_mJy = poly5(IfitDict["p"])(freqHirArr_Hz/1e9)
        quModArr = model(ip.inParms, lamSqHirArr_m2)
        specFig.clf()
        plot_Ipqu_spectra_fig(freqArr_Hz     = freqArr_Hz,
                              IArr_mJy       = IArr_mJy, 
                              qArr           = qArr, 
                              uArr           = uArr, 
                              dIArr_mJy      = dIArr_mJy,
                              dqArr          = dqArr,
                              duArr          = duArr,
                              freqHirArr_Hz  = freqHirArr_Hz,
                              IModArr_mJy    = IModArr_mJy,
                              qModArr        = quModArr.real, 
                              uModArr        = quModArr.imag,
                              fig            = specFig)
        specFig.canvas.draw()
        print "> Press <RETURN> to exit ...",
        raw_input()


#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
