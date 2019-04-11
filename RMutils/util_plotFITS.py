#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     util_plotFITS.py                                                  #
#                                                                             #
# PURPOSE:  Common function for plotting fits images.                         #
#                                                                             #
# MODIFIED: 15-May-2016 by C. Purcell                                         #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#  label_format_dms                                                           #
#  label_format_hms                                                           #
#  label_format_deg                                                           #
#  plot_fits_map                                                              #
#                                                                             #
#=============================================================================#

import math as m
import numpy as np
import astropy.io.fits as pf
import astropy.wcs.wcs as pw
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon

from .normalize import APLpyNormalize
from .util_FITS import strip_fits_dims
from .util_FITS import mkWCSDict
from .util_misc  import calc_stats


#-----------------------------------------------------------------------------#
def label_format_dms(deg, pos):
    """
    Format decimal->DD:MM:SS. Called by the label formatter. 
    """
    
    angle = abs(deg)
    sign=1
    if angle!=0: sign = angle/deg

    # Calcuate the degrees, min and sec
    dd = int(angle)
    rmndr = 60.0*(angle - dd)
    mm = int(rmndr)
    ss = 60.0*(rmndr-mm)

    # If rounding up to 60, carry to the next term
    if float("%05.2f" % ss) >=60.0:
        ss = 0.0
        mm+=1.0
    if float("%02d" % mm) >=60.0:
        mm = 0.0
        dd+=1.0
        
    if sign > 0:
        return "%02dd%02dm" % (sign*dd, mm)
    else:
        return "%03dd%02dm" % (sign*dd, mm)

    
#-----------------------------------------------------------------------------#
def label_format_hms(deg, pos):
    """
    Format decimal->HH:MM:SS. Called by the label formatter. 
    """

    hrs = deg/15.0
    
    angle = abs(hrs)
    sign=1
    if angle!=0: sign = angle/hrs

    # Calcuate the hrsrees, min and sec
    dd = int(angle)
    rmndr = 60.0*(angle - dd)
    mm = int(rmndr)
    ss = 60.0*(rmndr-mm)

    # If rounding up to 60, carry to the next term
    if float("%05.2f" % ss) >=60.0:
        ss = 0.0
        mm+=1.0
    if float("%02d" % mm) >=60.0:
        mm = 0.0
        dd+=1.0

    if sign > 0:
        return "%02dh%02dm%02.0fs" % (sign*dd, mm, ss)
    else:
        return "%03dh%02dm%02.0fs" % (sign*dd, mm, ss)

    
#-----------------------------------------------------------------------------#
def label_format_deg(deg, pos):
    
    return "%.3f" % deg


#-----------------------------------------------------------------------------#
def plot_fits_map(data, header, stretch='auto', exponent=2, scaleFrac=0.9,
                  cmapName='gist_heat', zMin=None, zMax=None,
                  annEllipseLst=[], annPolyLst=[], bunit=None,
                  lw=1.0, interpolation='Nearest', fig=None, dpi=100,
                  doColbar=True):

    """
    Plot a colourscale image of a FITS map.
    
    annEllipseLst is a list of lists:
        annEllipseLst[0][i] = x_deg
        annEllipseLst[1][i] = y_deg
        annEllipseLst[2][i] = minor_deg
        annEllipseLst[3][i] = major_deg
        annEllipseLst[4][i] = pa_deg
        annEllipseLst[5][i] = colour ... optional, default to 'g'
        
    annPolyLst is also a list of lists:
        annPolyLst[0][i] = list of polygon coords = [[x1,y1], [x2, y2] ...]
        annPolyLst[1][i] = colour of polygon e.g., 'w'
    """
    
    # Strip unused dimensions from the array
    data, header = strip_fits_dims(data, header, 2, 5)

    # Parse the WCS information
    w = mkWCSDict(header)
    wcs = pw.WCS(w['header2D'])

    # Calculate the image vmin and vmax by measuring the range in the inner
    # 'scale_frac' of the image
    s = data.shape
    boxMaxX = int( s[-1]/2.0 + s[-1] * scaleFrac/2.0 + 1.0 )
    boxMinX = int( s[-1]/2.0 - s[-1] * scaleFrac/2.0 + 1.0 )
    boxMaxY = int( s[-2]/2.0 + s[-2] * scaleFrac/2.0 + 1.0 )
    boxMinY = int( s[-2]/2.0 - s[-2] * scaleFrac/2.0 + 1.0 )
    dataSample = data[boxMinY:boxMaxY, boxMinX:boxMaxX]
    measures = calc_stats(dataSample)
    sigma = abs(measures['max'] / measures['madfm'])
    if stretch=='auto':
        if sigma <= 20:
            vMin = measures['madfm'] * (-1.5)
            vMax = measures['madfm'] * 10.0
            stretch='linear'
        elif sigma > 20:
            vMin = measures['madfm'] * (-3.0)
            vMax = measures['madfm'] * 40.0
            stretch='linear'
        elif sigma > 500:
            vMin = measures['madfm'] * (-7.0)
            vMax = measures['madfm'] * 200.0
            stretch='sqrt'
    if not zMax is None:
        vMax = max(zMax, measures['max'])
    if not zMax is None:
        vMin = zMin
        
    # Set the colourscale using an normalizer object
    normalizer = APLpyNormalize(stretch=stretch, exponent=exponent,
                                vmin=vMin, vmax=vMax)

    # Setup the figure
    if fig is None:
        fig = plt.figure(figsize=(9.5, 8))
    ax = fig.add_axes([0.1, 0.08, 0.9, 0.87])
    if w['coord_type']=='EQU':
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
    elif w['coord_type']=='GAL':
        ax.set_xlabel('Galactic Longitude (deg)') 
        ax.set_ylabel('Galactic Latitude (deg)')
    else:
        ax.set_xlabel('Unknown')
        ax.set_ylabel('Unknown')
    cosY = m.cos( m.radians(w['ycent']) )
    aspect = abs( w['ydelt'] / (w['xdelt'] * cosY))

    # Set the format of the major tick mark and labels
    if w['coord_type']=='EQU':
        f = 15.0
        majorFormatterX = FuncFormatter(label_format_hms)
        minorFormatterX = None
        majorFormatterY = FuncFormatter(label_format_dms)
        minorFormattery = None
    else:
        f = 1.0
        majorFormatterX = FuncFormatter(label_format_deg)
        minorFormatterX = None
        majorFormatterY = FuncFormatter(label_format_deg)
        minorFormattery = None
    ax.xaxis.set_major_formatter(majorFormatterX)
    ax.yaxis.set_major_formatter(majorFormatterY)
    
    # Set the location of the the major tick marks
    #xrangeArcmin = abs(w['xmax']-w['xmin'])*(60.0*f)
    #xmultiple = m.ceil(xrangeArcmin/3.0)/(60.0*f)
    #yrangeArcmin = abs(w['ymax']-w['ymin'])*60.0
    #ymultiple = m.ceil(yrangeArcmin/3.0)/60.0
    #majorLocatorX = MultipleLocator(xmultiple)
    #ax.xaxis.set_major_locator(majorLocatorX)
    #majorLocatorY = MultipleLocator(ymultiple)
    #ax.yaxis.set_major_locator(majorLocatorY)
    
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    # Print the image to the axis
    im = ax.imshow(data, interpolation=interpolation, origin='lower',
                   aspect=aspect, 
                   extent=[w['xmax'], w['xmin'], w['ymin'], w['ymax']],
                   cmap=plt.get_cmap(cmapName), norm=normalizer) 
    
    # Add the colorbar
    if doColbar:
        cbar = fig.colorbar(im, pad=0.0)
        if 'BUNIT' in header:
            cbar.set_label(header['BUNIT'])
        else:
            cbar.set_label('Unknown')
        if not bunit is None:
            cbar.set_label(bunit)
        
    # Format the colourbar labels - TODO

    # Set white ticks
    ax.tick_params(pad=5)
    for line in ax.xaxis.get_ticklines() + ax.get_yticklines():
        line.set_markeredgewidth(1)
        line.set_color('w')

    # Create the ellipse source annotations
    if len(annEllipseLst) > 0:
        if len(annEllipseLst) >= 5:
            srcXLst = np.array(annEllipseLst[0])
            srcYLst = np.array(annEllipseLst[1])
            srcMinLst = np.array(annEllipseLst[2])
            srcMajLst = np.array(annEllipseLst[3])
            srcPALst = np.array(annEllipseLst[4])
        if len(annEllipseLst) >= 6:
            if type(annEllipseLst[5]) is str:
                srcEColLst = [annEllipseLst[5]] * len(srcXLst)
            elif type(annEllipseLst[5]) is list:
                srcEColLst = annEllipseLst[5]
            else:
                rcEColLst = ['g'] * len(srcXLst)
        else:
            srcEColLst = ['g'] * len(srcXLst)
        for i in range(len(srcXLst)):
            try:
                el = Ellipse((srcXLst[i], srcYLst[i]), srcMinLst[i],
                             srcMajLst[i], angle=180.0-srcPALst[i],
                             edgecolor=srcEColLst[i],
                             linewidth=lw, facecolor='none')
                ax.add_artist(el)
            except Exception:
                pass
        
    # Create the polygon source annotations
    if len(annPolyLst) > 0:
        annPolyCoordLst = annPolyLst[0]
        if len(annPolyLst) > 1:
            if type(annPolyLst[1]) is str:
                annPolyColorLst = [annPolyLst[1]] * len(annPolyCoordLst)
            elif type(annPolyLst[1]) is list:
                annPolyColorLst = annPolyLst[1]
            else:
                annPolyColorLst = ['g'] * len(annPolyCoordLst)
        else:
            annPolyColorLst = ['g'] * len(annPolyCoordLst)
        for i in range(len(annPolyCoordLst)):
            cpoly = Polygon(annPolyCoordLst[i], animated=False, linewidth=lw)
            cpoly.set_edgecolor(annPolyColorLst[i])
            cpoly.set_facecolor('none')
            ax.add_patch(cpoly)
    
    return fig
