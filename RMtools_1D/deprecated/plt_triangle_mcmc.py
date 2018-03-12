#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     plt_triangle_mcmc.py                                              #
#                                                                             #
# PURPOSE:  Create a triangle plot showing the results of the MCMC fitting.   #
#           Reads in a python pickle containing the chain and parameters.     #
#                                                                             #
# MODIFIED: 18-Nov-2016 by C. Purcell                                         #
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

# Y axis label offset
yAxLabOff = -0.15

#----------------------------------------------------------------------------#

import sys
import os
import argparse
import traceback
import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


#-----------------------------------------------------------------------------#
def main():
    """
    Start the plot_triangle procedure if called from the command line.
    """
    
    # Help string to be shown using the -h option
    descStr = """
    """
    
    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("pickleFile", metavar="MCMC.pkl", nargs=1,
                        help="Pickle file containing MCMC results.")
    parser.add_argument("-n", dest="nBins", type=int, default=40,
                        help="number of X & Y bins in correlation plots [40].")
    parser.add_argument("-o", dest="outFile", type=str, default="",
                        help="save a figure to an output file [None].")
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.pickleFile[0]):
        print "File does not exist: '%s'." % args.pickleFile[0]
        sys.exit()
    dataDir, dummy = os.path.split(args.pickleFile[0])

    # Run the plot_triangle procedure
    plot_triangle(pklFile = args.pickleFile[0],
                  nBins   = args.nBins)

#-----------------------------------------------------------------------------#
def plot_triangle(pklFile, nBins=40, outFile="", figHeight=10):
    """
    Create a triangle or 'corner' plot from MCMC results store in a pickle.
    """
    
    # Load in the pickle and unpack
    fh = open(pklFile, "rb")
    results = pkl.load(fh)
    inParms = results['inParms']
    flatchain = results['flatchain']
    flatlnprob = results['flatlnprob']    
    
    # Set out the plots based on the number of variables
    fxi = [i for i in range(len(inParms)) if not inParms[i]["fixed"] ]
    nDim = len(fxi)
    
    # Setup the figure page
    figHeight
    fig = plt.figure(figsize=(1.2*figHeight, figHeight))

    # Alter the default linewidths etc
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['xtick.major.size'] = 6.0
    mpl.rcParams['xtick.minor.size'] = 4.0
    mpl.rcParams['ytick.major.size'] = 6.0
    mpl.rcParams['ytick.minor.size'] = 4.0
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 12.0
    
    # Loop through the rows and columns
    axLst = []
    xAnchorDict = {}
    yAnchorDict = {}
    for i in range(nDim**2):
        xi = i%nDim
        yi = i/nDim
        doHist = int(xi==yi)
        mskAx = int(xi>yi)

        # Get the x-data and labels
        xData = flatchain[:, xi]
        yData = flatchain[:, yi]
        zData = flatlnprob
        xLabel = inParms[fxi[xi]]["label"]
        yLabel = inParms[fxi[yi]]["label"]
        
        # Read off scales
        xMinData = min(xData)
        xMaxData = max(xData)
        yMinData = min(yData)
        yMaxData = max(yData)

        # Bin the data
        xBins = np.linspace(xMinData, xMaxData, int(nBins+1))
        yBins = np.linspace(yMinData, yMaxData, int(nBins+1))
        nXY, yBins, xBins = np.histogram2d(yData, xData, (yBins, xBins))

        # Skip redundant plots above diagonal
        if mskAx:
            axLst.append(None)
            continue

        # Anchor x-scale to the histograms and the y-scales to the 1st row
        sharex = None
        sharey = None
        if xi==0 and not doHist:
            yAnchorDict[yi] = i
        if doHist:
            xAnchorDict[xi] = i
        else:
            sharex = axLst[xAnchorDict[xi]]
            if xi>0:
                sharey = axLst[yAnchorDict[yi]]

        # Draw the axis
        #axLst[i].set_title("(%s %s) %s" % (xi, yi, i))  # DEBUG
        axLst.append(fig.add_subplot(nDim, nDim, i+1, sharex=sharex,
                                     sharey=sharey))
        
        # Choose which tickmarks and labels appear on plot
        if not(yi==nDim-1):
            [label.set_visible(False) for label in axLst[i].get_xticklabels()]
        else:
            plt.setp(axLst[i].xaxis.get_majorticklabels(), rotation=45)
            axLst[i].set_xlabel(xLabel, rotation=45, fontsize = 10)
        if xi==0:            
            plt.setp(axLst[i].yaxis.get_majorticklabels(), rotation=45)
            axLst[i].set_ylabel(yLabel, rotation=45, fontsize=10)
            axLst[i].yaxis.set_label_coords(yAxLabOff*nDim, 0.5)
        else:
            [label.set_visible(False) for label in axLst[i].get_yticklabels()]
        if doHist:
            [label.set_visible(False) for label in axLst[i].get_yticklabels()]
            
        # Plot the data as a histogram on the diagonal or as a density map
        if doHist:
            (h,d1, d2) = axLst[i].hist(xData, bins=xBins, color="silver",
                                       histtype="stepfilled", linewidth=1.0)
            axLst[i].set_xlim(xMinData, xMaxData)
            axLst[i].set_ylim(0.01, max(h)*1.1)            
        else:
            im = axLst[i].imshow(nXY, interpolation='nearest', origin='lower',
                                 extent=[xMinData, xMaxData,
                                         yMinData, yMaxData],
                                 aspect="auto", cmap="gray_r")
            axLst[i].set_ylim(yMinData, yMaxData)
            axLst[i].set_xlim(xMinData, xMaxData)

            # Plot the contours
            try:
                maxDensity = np.max(nXY)
                levels = get_sigma_levels(nXY)
                levels.sort()
                cax2 = axLst[i].contour(nXY, levels=levels,
                                        colors=['b','r','g'], lw=3.0,
                                        extent=[xMinData, xMaxData,
                                                yMinData, yMaxData])
            except Exception:
                print "Contour plotting failed!"

        # Common formatting
        axLst[i].xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axLst[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axLst[i].yaxis.set_major_locator(MaxNLocator(4))
        axLst[i].xaxis.set_major_locator(MaxNLocator(4))
        format_ticks(axLst[i], 10, 1.2)

    # Final formatting and save
    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.18,
                       wspace=0.15, hspace=0.05)
    #pl.savefig('triangle.eps')
    fig.show()
    print "> Press <RETURN> to exit ...",
    raw_input()

    
#-----------------------------------------------------------------------------#
def get_sigma_levels(img, cLevels=[0.682689492, 0.954499736, 0.997300204]):

    ind = np.unravel_index(np.argsort(img, axis=None)[::-1], img.shape)
    cumsum = np.cumsum(img[ind])/np.sum(img[ind])
    aLevels = []
    for cLevel in cLevels:
        aLevels.append(img[ind][np.where(cumsum<cLevel)[0][-1]])
        
    return aLevels
    
    
#-----------------------------------------------------------------------------#
def format_ticks(ax, pad=10, w=1.0):
    
    ax.tick_params(pad=pad)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markeredgewidth(w)


#-----------------------------------------------------------------------------#
main()
