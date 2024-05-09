#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     util_plotTk.py                                                    #
#                                                                             #
# PURPOSE:  Plotting functions for the POSSUM pipeline                        #
#                                                                             #
# MODIFIED: 31-Jan-2018 by C. Purcell                                         #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
# xfloat                                                                      #
# xint                                                                        #
# filter_range_indx                                                           #
# tweakAxFormat                                                               #
# format_ticks                                                                #
# CustomNavbar                                                                #
# plot_I_vs_nu_ax                                                             #
# plot_PQU_vs_nu_ax                                                           #
# plot_rmsIQU_vs_nu_ax                                                        #
# plot_pqu_vs_lamsq_ax                                                        #
# plot_psi_vs_lamsq_ax                                                        #
# plot_q_vs_u_ax                                                              #
# plot_RMSF_ax                                                                #
# gauss                                                                       #
# plot_dirtyFDF_ax                                                            #
# plot_cleanFDF_ax                                                            #
# plot_hist4_ax                                                               #
#                                                                             #
# #-------------------------------------------------------------------------# #
#                                                                             #
# plotSpecIPQU                                                                #
# plotSpecRMS                                                                 #
# plotPolang                                                                  #
# plotFracPol                                                                 #
# plotFracQvsU                                                                #
# plot_Ipqu_spectra_fig                                                       #
# plotPolsummary                                                              #
# plotPolresidual                                                             #
# plot_rmsf_fdf_fig                                                           #
# plotRMSF                                                                    #
# plotDirtyFDF                                                                #
# plotCleanFDF                                                                #
# plotStampI                                                                  #
# plotStampP                                                                  #
# plotSctHstQuery                                                             #
# mk_hist_poly                                                                #
# label_format_exp                                                            #
# plot_complexity_fig                                                         #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 -2018 Cormac R. Purcell                                  #
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
# =============================================================================#

import math as m
import os
import sys
import tkinter.ttk
import traceback

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c as speed_of_light
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .normalize import APLpyNormalize
from .util_misc import nanmedian, norm_cdf, xfloat
from .util_plotFITS import plot_fits_map

# Alter the default linewidths etc.
mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["xtick.major.size"] = 8.0
mpl.rcParams["xtick.minor.size"] = 4.0
mpl.rcParams["ytick.major.size"] = 8.0
mpl.rcParams["ytick.minor.size"] = 4.0
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.size"] = 12.0

# Quick workaround to check if there is a chance for matplotlin to catch an
# X DISPLAY
try:
    if os.environ["rmsy_mpl_backend"] == "Agg":
        print('Environment variable rmsy_mpl_backend="Agg" detected.')
        print('Using matplotlib "Agg" backend in order to save the plots.')
        mpl.use("Agg")
except Exception:
    pass


# -----------------------------------------------------------------------------#
def xint(x, default=None):
    if x is None:
        return default
    return int(x)


# -----------------------------------------------------------------------------#
def filter_range_indx(a, dataMin=None, dataMax=None, filterNans=False):
    """Return a boolean array [True, ...] where data falls outside of the
    range [dataMin <= a <= dataMax]."""

    if filterNans:
        iNaN = np.zeros_like(a, dtype="bool")
    else:
        iNaN = a != a
    if dataMin is None:
        i1 = np.ones_like(a, dtype="bool")
    else:
        i1 = a >= dataMin
        i1 += iNaN
    if dataMax is None:
        i2 = np.ones_like(a, dtype="bool")
    else:
        i2 = a <= dataMax
        i2 += iNaN
    return ~i1 + ~i2


# -----------------------------------------------------------------------------#
def tweakAxFormat(
    ax,
    pad=10,
    loc="upper right",
    linewidth=1,
    ncol=1,
    bbox_to_anchor=(1.00, 1.00),
    showLeg=True,
):
    # Axis/tic formatting
    ax.tick_params(pad=pad)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markeredgewidth(linewidth)

    # Legend formatting
    if showLeg:
        leg = ax.legend(
            numpoints=1,
            loc=loc,
            shadow=False,
            borderaxespad=0.3,
            ncol=ncol,
            bbox_to_anchor=bbox_to_anchor,
        )
        for t in leg.get_texts():
            t.set_fontsize("small")
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_alpha(0.5)

    return ax


# -----------------------------------------------------------------------------#
def format_ticks(ax, pad=10, w=1.0):
    ax.tick_params(pad=pad)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markeredgewidth(w)


# -----------------------------------------------------------------------------#
class CustomNavbar(NavigationToolbar2Tk):
    """Custom navigation toolbar subclass"""

    def __init__(self, canvas, window):
        NavigationToolbar2Tk.__init__(self, canvas, window)
        self.legStat = []
        for i in range(len(self.canvas.figure.axes)):
            ax = self.canvas.figure.axes[i]
            if ax.get_legend() is None:
                self.legStat.append(None)
            else:
                self.legStat.append(True)

    def _init_toolbar(self):
        NavigationToolbar2Tk._init_toolbar(self)

        # Add the legend toggle button
        self.legBtn = tkinter.ttk.Button(
            self, text="Hide Legend", command=self.toggle_legend
        )
        self.legBtn.pack(side="left")

        # Remove the back and forward buttons
        # List of buttons is in self.toolitems
        buttonLst = self.pack_slaves()
        buttonLst[1].pack_forget()
        buttonLst[2].pack_forget()

    def toggle_legend(self):
        for i in range(len(self.canvas.figure.axes)):
            ax = self.canvas.figure.axes[i]
            if self.legStat[i] is not None:
                ax.get_legend().set_visible(not self.legStat[i])
                self.legStat[i] = not self.legStat[i]
        if self.legBtn["text"] == "Hide Legend":
            self.legBtn["text"] = "Show Legend"
        else:
            self.legBtn["text"] = "Hide Legend"
        self.canvas.draw()


# -----------------------------------------------------------------------------#
def plot_I_vs_nu_ax(
    ax,
    freqArr_Hz,
    IArr,
    dIArr=None,
    freqHirArr_Hz=None,
    IModArr=None,
    axisYright=False,
    axisXtop=False,
    units="",
):
    """Plot the I spectrum and an optional model."""

    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Default to non-high-resolution inputs
    if freqHirArr_Hz is None:
        freqHirArr_Hz = freqArr_Hz

    # Plot I versus frequency
    ax.errorbar(
        x=freqArr_Hz / 1e9,
        y=IArr,
        yerr=dIArr,
        mfc="none",
        ms=2,
        fmt="D",
        mec="k",
        ecolor="k",
        alpha=0.5,
        elinewidth=1.0,
        capsize=2,
        label="Stokes I",
    )

    # Plot the model
    if IModArr is not None:
        ax.plot(freqHirArr_Hz / 1e9, IModArr, color="tab:red", lw=2.5, label="I Model")

    # Scaling & formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = (np.nanmax(freqArr_Hz) - np.nanmin(freqArr_Hz)) / 1e9
    ax.set_xlim(
        np.nanmin(freqArr_Hz) / 1e9 - xRange * 0.05,
        np.nanmax(freqArr_Hz) / 1e9 + xRange * 0.05,
    )
    ax.set_xlabel(r"$\nu$ (GHz)")
    ax.set_ylabel(rf"Flux Density ({units})")
    ax.minorticks_on()

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------#
def plot_PQU_vs_nu_ax(
    ax,
    freqArr_Hz,
    QArr,
    UArr,
    dQArr=None,
    dUArr=None,
    freqHirArr_Hz=None,
    QmodArr=None,
    UmodArr=None,
    axisYright=False,
    axisXtop=False,
):
    """Plot the P, Q & U spectrum and an optional model."""

    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Default to non-high-resolution inputs
    if freqHirArr_Hz is None:
        freqHirArr_Hz = freqArr_Hz

    # Calculate P and errors
    PArr = np.sqrt(np.power(QArr, 2) + np.power(UArr, 2))
    if dQArr is None or dUArr is None:
        dPArr = None
    else:
        dPArr = np.sqrt(np.power(dQArr, 2) + np.power(dUArr, 2))

    # Plot P, Q, U versus frequency
    ax.errorbar(
        x=freqArr_Hz / 1e9,
        y=QArr,
        yerr=dQArr,
        mec="tab:blue",
        mfc="none",
        ms=2,
        fmt="D",
        color="g",
        elinewidth=1.0,
        capsize=2,
        label="Stokes Q",
    )
    ax.errorbar(
        x=freqArr_Hz / 1e9,
        y=UArr,
        yerr=dUArr,
        mec="tab:red",
        mfc="none",
        ms=2,
        fmt="D",
        color="tab:red",
        elinewidth=1.0,
        capsize=2,
        label="Stokes U",
    )
    ax.errorbar(
        x=freqArr_Hz / 1e9,
        y=PArr,
        yerr=dPArr,
        mec="k",
        mfc="none",
        ms=2,
        fmt="D",
        color="k",
        elinewidth=1.0,
        capsize=2,
        label="Intensity P",
    )

    # Plot the models
    if QmodArr is not None:
        ax.plot(freqHirArr_Hz / 1e9, QmodArr, color="tab:blue", lw=0.5, label="Model Q")
    if UmodArr is not None:
        ax.plot(freqHirArr_Hz / 1e9, UmodArr, color="tab:red", lw=0.5, label="Model U")
    if QmodArr is not None and UmodArr is not None:
        PmodArr = np.sqrt(QmodArr**2.0 + UmodArr**2.0)
        ax.plot(freqHirArr_Hz / 1e9, PmodArr, color="k", lw=0.5, label="Model P")

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = (np.nanmax(freqArr_Hz) - np.nanmin(freqArr_Hz)) / 1e9
    ax.set_xlim(
        np.nanmin(freqArr_Hz) / 1e9 - xRange * 0.05,
        np.nanmax(freqArr_Hz) / 1e9 + xRange * 0.05,
    )
    ax.set_xlabel(r"$\nu$ (GHz)")
    ax.set_ylabel("Flux Density")
    ax.axhline(0, color="grey")

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------#
def plot_rmsIQU_vs_nu_ax(
    ax, freqArr_Hz, rmsIArr, rmsQArr, rmsUArr, axisYright=False, axisXtop=False
):
    """Plot the noise spectra in Stokes I, Q & U."""

    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot the rms spectra in GHz and flux units
    ax.plot(freqArr_Hz / 1e9, rmsIArr, marker="o", color="k", lw=0.5, label="rms I")
    ax.plot(
        freqArr_Hz / 1e9, rmsQArr, marker="o", color="tab:blue", lw=0.5, label="rms Q"
    )
    ax.plot(
        freqArr_Hz / 1e9, rmsUArr, marker="o", color="tab:red", lw=0.5, label="rms U"
    )
    # ax.text(0.05, 0.94, 'I, Q & U RMS', transform=ax.transAxes)

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = (np.nanmax(freqArr_Hz) - np.nanmin(freqArr_Hz)) / 1e9
    ax.set_xlim(
        np.nanmin(freqArr_Hz) / 1e9 - xRange * 0.05,
        np.nanmax(freqArr_Hz) / 1e9 + xRange * 0.05,
    )
    ax.set_xlabel(r"$\nu$ (GHz)")
    ax.set_ylabel("Flux Density")

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------#
def plot_pqu_vs_lamsq_ax(
    ax,
    lamSqArr_m2,
    qArr,
    uArr,
    pArr=None,
    dqArr=None,
    duArr=None,
    dpArr=None,
    lamSqHirArr_m2=None,
    qModArr=None,
    uModArr=None,
    model_dict=None,
    axisYright=False,
    axisXtop=False,
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Default to non-high-resolution inputs
    if lamSqHirArr_m2 is None:
        lamSqHirArr_m2 = lamSqArr_m2

    # Calculate p and errors
    if pArr is None:
        pArr = np.sqrt(qArr**2.0 + uArr**2.0)
        pArr = np.where(np.isfinite(pArr), pArr, np.nan)
    if dpArr is None:
        if dqArr is None or duArr is None:
            dpArr = None
        else:
            dpArr = np.sqrt(dqArr**2.0 + duArr**2.0)
            dpArr = np.where(np.isfinite(dpArr), dpArr, np.nan)

    # Plot p, q, u versus lambda^2
    #    """
    ax.errorbar(
        x=lamSqArr_m2,
        y=qArr,
        yerr=dqArr,
        mec="tab:blue",
        mfc="none",
        ms=2,
        fmt="D",
        ecolor="tab:blue",
        alpha=0.5,
        elinewidth=1.0,
        capsize=2,
        label="Stokes q",
    )
    ax.errorbar(
        x=lamSqArr_m2,
        y=uArr,
        yerr=duArr,
        mec="tab:red",
        mfc="none",
        ms=2,
        fmt="D",
        ecolor="tab:red",
        alpha=0.5,
        elinewidth=1.0,
        capsize=2,
        label="Stokes u",
    )
    ax.errorbar(
        x=lamSqArr_m2,
        y=pArr,
        yerr=dpArr,
        mec="k",
        mfc="none",
        ms=2,
        fmt="D",
        ecolor="k",
        alpha=0.5,
        elinewidth=1.0,
        capsize=2,
        label="Intensity p",
    )
    """
    ax.errorbar(x=lamSqArr_m2, y=pArr, yerr=dpArr, mec='k', mfc='tab:red', ms=2,
                fmt='D', ecolor='k', elinewidth=1.0, capsize=2,
                label='Intensity p')

    ax.errorbar(x=lamSqArr_m2, y=qArr, yerr=dqArr, mec='tab:blue', mfc='tab:red', ms=2,
                fmt='D', ecolor='tab:blue', elinewidth=1.0, capsize=2,
                label='Stokes q')

    ax.errorbar(x=lamSqArr_m2, y=uArr, yerr=duArr, mec='tab:red', mfc='tab:blue', ms=2,
                fmt='D', ecolor='tab:red', elinewidth=1.0, capsize=2,
                label='Stokes u')
    """

    # Plot the models
    if qModArr is not None:
        ax.plot(
            lamSqHirArr_m2, qModArr, color="tab:blue", alpha=1, lw=0.1, label="Model q"
        )
    if uModArr is not None:
        ax.plot(
            lamSqHirArr_m2, uModArr, color="tab:red", alpha=1, lw=0.1, label="Model u"
        )
    if qModArr is not None and uModArr is not None:
        pModArr = np.sqrt(qModArr**2.0 + uModArr**2.0)
        ax.plot(lamSqHirArr_m2, pModArr, color="k", alpha=1, lw=0.1, label="Model p")
    if model_dict is not None:
        errDict = {}
        QUerrmodel = []
        # Sample the posterior randomly 100 times
        for i in range(1000):
            idx = np.random.choice(np.arange(model_dict["posterior"].shape[0]))
            for j, name in enumerate(model_dict["parNames"]):
                errDict[name] = model_dict["posterior"][name][idx]
            QUerrmodel.append(model_dict["model"](errDict, lamSqHirArr_m2))
        QUerrmodel = np.array(QUerrmodel)
        low_re, med_re, high_re = np.percentile(
            np.real(QUerrmodel), [16, 50, 84], axis=0
        )
        low_im, med_im, high_im = np.percentile(
            np.imag(QUerrmodel), [16, 50, 84], axis=0
        )
        low_abs, med_abs, high_abs = np.percentile(
            np.abs(QUerrmodel), [16, 50, 84], axis=0
        )

        ax.plot(lamSqHirArr_m2, med_re, "-", color="tab:blue", linewidth=0.1, alpha=1)
        ax.fill_between(lamSqHirArr_m2, low_re, high_re, color="tab:blue", alpha=0.5)
        ax.plot(lamSqHirArr_m2, med_im, "-", color="tab:red", linewidth=0.1, alpha=1)
        ax.fill_between(lamSqHirArr_m2, low_im, high_im, color="tab:red", alpha=0.5)
        if qModArr is not None and uModArr is not None:
            ax.plot(lamSqHirArr_m2, med_abs, "-", color="k", linewidth=0.1, alpha=1)
            ax.fill_between(lamSqHirArr_m2, low_abs, high_abs, color="k", alpha=0.5)

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(lamSqArr_m2) - np.nanmin(lamSqArr_m2)
    ax.set_xlim(
        np.nanmin(lamSqArr_m2) - xRange * 0.05, np.nanmax(lamSqArr_m2) + xRange * 0.05
    )
    yDataMax = max(np.nanmax(pArr), np.nanmax(qArr), np.nanmax(uArr))
    yDataMin = min(np.nanmin(pArr), np.nanmin(qArr), np.nanmin(uArr))
    yRange = yDataMax - yDataMin
    medErrBar = np.max(
        [float(nanmedian(dpArr)), float(nanmedian(dqArr)), float(nanmedian(duArr))]
    )
    ax.set_ylim(
        yDataMin - 2 * medErrBar - yRange * 0.05,
        yDataMax + 2 * medErrBar + yRange * 0.1,
    )
    ax.set_xlabel(r"$\lambda^2$ (m$^2$)")
    ax.set_ylabel("Fractional Polarisation")
    ax.axhline(0, linestyle="--", color="grey")
    ax.minorticks_on()

    # Format tweaks
    ax = tweakAxFormat(ax)


# -----------------------------------------------------------------------------#
def plot_psi_vs_lamsq_ax(
    ax,
    lamSqArr_m2,
    qArr,
    uArr,
    dqArr=None,
    duArr=None,
    lamSqHirArr_m2=None,
    qModArr=None,
    uModArr=None,
    model_dict=None,
    axisYright=False,
    axisXtop=False,
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Default to non-high-resolution inputs
    if lamSqHirArr_m2 is None:
        lamSqHirArr_m2 = lamSqArr_m2

    # Calculate the angle and errors
    pArr = np.sqrt(qArr**2.0 + uArr**2.0)
    psiArr_deg = np.degrees(np.arctan2(uArr, qArr) / 2.0)
    if dqArr is None or duArr is None:
        dQUArr = None
        dPsiArr_deg = None
    else:
        dQUArr = np.sqrt(dqArr**2.0 + duArr**2.0)
        dPsiArr_deg = np.degrees(
            np.sqrt((qArr * duArr) ** 2.0 + (uArr * dqArr) ** 2.0) / (2.0 * pArr**2.0)
        )

    # Plot psi versus lambda^2
    ax.errorbar(
        x=lamSqArr_m2,
        y=psiArr_deg,
        yerr=dPsiArr_deg,
        mec="k",
        mfc="none",
        ms=2,
        fmt="D",
        ecolor="k",
        alpha=0.3,
        elinewidth=1.0,
        capsize=2,
    )

    # Plot the model
    if qModArr is not None and uModArr is not None:
        psiHirArr_deg = np.degrees(np.arctan2(uModArr, qModArr) / 2.0)
        ax.plot(
            lamSqHirArr_m2,
            psiHirArr_deg,
            color="tab:red",
            lw=0.1,
            label=r"Model $\psi$",
        )
    if model_dict is not None:
        errDict = {}
        psi_errmodel = []
        # Sample the posterior randomly 100 times
        for i in range(1000):
            idx = np.random.choice(np.arange(model_dict["posterior"].shape[0]))
            for j, name in enumerate(model_dict["parNames"]):
                errDict[name] = model_dict["posterior"][name][idx]
            QUerrmodel = model_dict["model"](errDict, lamSqHirArr_m2)
            Qerrmodel = np.real(QUerrmodel)
            Uerrmodel = np.imag(QUerrmodel)
            psi_errmodel.append(np.degrees(np.arctan2(Uerrmodel, Qerrmodel) / 2.0))

        psi_errmodel = np.array(psi_errmodel)
        low, med, high = np.percentile(psi_errmodel, [16, 50, 84], axis=0)

        ax.plot(lamSqHirArr_m2, med, "-", color="tab:red", linewidth=0.1, alpha=1)
        ax.fill_between(
            lamSqHirArr_m2, low, high, color="tab:red", linewidth=0.1, alpha=0.5
        )

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(lamSqArr_m2) - np.nanmin(lamSqArr_m2)
    ax.set_xlim(
        np.nanmin(lamSqArr_m2) - xRange * 0.05, np.nanmax(lamSqArr_m2) + xRange * 0.05
    )
    ax.set_ylim(-99.9, 99.9)
    ax.set_xlabel(r"$\lambda^2$ (m$^2$)")
    ax.set_ylabel(r"$\psi$ (degrees)")
    ax.axhline(0, linestyle="--", color="grey")
    ax.minorticks_on()

    # Format tweaks
    ax = tweakAxFormat(ax, showLeg=False)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------#
def plot_q_vs_u_ax(
    ax,
    lamSqArr_m2,
    qArr,
    uArr,
    dqArr=None,
    duArr=None,
    lamSqHirArr_m2=None,
    qModArr=None,
    uModArr=None,
    model_dict=None,
    axisYright=False,
    axisXtop=False,
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot u versus q
    ax.errorbar(
        x=qArr,
        y=uArr,
        xerr=dqArr,
        yerr=duArr,
        mec="grey",
        mfc="none",
        ms=1,
        fmt=".",
        ecolor="grey",
        elinewidth=1.0,
        capsize=2,
        zorder=1,
    )
    freqArr_Hz = speed_of_light.value / np.sqrt(lamSqArr_m2)
    ax.scatter(
        x=qArr,
        y=uArr,
        c=freqArr_Hz,
        cmap="rainbow_r",
        s=30,
        marker="D",
        edgecolor="none",
        linewidth=0.1,
        zorder=2,
    )

    # Plot the model
    if qModArr is not None and uModArr is not None:
        ax.plot(qModArr, uModArr, color="k", lw=0.1, label="Model q & u", zorder=2)
    if model_dict is not None:
        errDict = {}
        # Sample the posterior randomly 100 times
        for i in range(1000):
            idx = np.random.choice(np.arange(model_dict["posterior"].shape[0]))
            for j, name in enumerate(model_dict["parNames"]):
                errDict[name] = model_dict["posterior"][name][idx]
            QUerrmodel = model_dict["model"](errDict, lamSqHirArr_m2)
            Qerrmodel = np.real(QUerrmodel)
            Uerrmodel = np.imag(QUerrmodel)
            ax.plot(Qerrmodel, Uerrmodel, color="k", lw=0.1, alpha=0.5, zorder=2)

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(qArr) - np.nanmin(qArr)
    ax.set_xlim(np.nanmin(qArr) - xRange * 0.05, np.nanmax(qArr) + xRange * 0.05)
    yRange = np.nanmax(uArr) - np.nanmin(uArr)
    ax.set_ylim(np.nanmin(uArr) - yRange * 0.05, np.nanmax(uArr) + yRange * 0.05)
    ax.set_xlabel("Stokes q")
    ax.set_ylabel("Stokes u")
    ax.axhline(0, linestyle="--", color="grey")
    ax.axvline(0, linestyle="--", color="grey")
    ax.axis("equal")
    ax.minorticks_on()
    # Format tweaks
    ax = tweakAxFormat(ax, showLeg=False)


# -----------------------------------------------------------------------------#
def unwrap_lines(dat, lims=[-90.0, 90.0], thresh=0.95):
    jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))


# -----------------------------------------------------------------------------#
def plot_RMSF_ax(
    ax, phiArr, RMSFArr, fwhmRMSF=None, axisYright=False, axisXtop=False, doTitle=False
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot the RMSF
    ax.step(phiArr, RMSFArr.real, where="mid", color="tab:blue", lw=0.5, label="Real")
    ax.step(
        phiArr, RMSFArr.imag, where="mid", color="tab:red", lw=0.5, label="Imaginary"
    )
    ax.step(phiArr, np.abs(RMSFArr), where="mid", color="k", lw=1.0, label="PI")
    ax.axhline(0, color="grey")
    if doTitle:
        ax.text(0.05, 0.84, "RMSF", transform=ax.transAxes)

    # Plot the Gaussian fit
    if fwhmRMSF is not None:
        yGauss = gauss([1.0, 0.0, fwhmRMSF])(phiArr)
        ax.plot(
            phiArr,
            yGauss,
            color="g",
            marker="None",
            mfc="w",
            mec="g",
            ms=10,
            label="Gaussian",
            lw=2.0,
            ls="--",
        )

    # Scaling
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(phiArr) - np.nanmin(phiArr)
    ax.set_xlim(np.nanmin(phiArr) - xRange * 0.01, np.nanmax(phiArr) + xRange * 0.01)
    ax.set_ylabel("Normalised Units")
    ax.set_xlabel(r"$\phi$ rad m$^{-2}$")
    ax.axhline(0, color="grey")

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------
def gauss(p):
    """Return a fucntion to evaluate a Gaussian with parameters
    p = [amp, mean, FWHM]"""

    a, b, w = p
    gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
    s = w / gfactor

    def rfunc(x):
        y = a * np.exp(-((x - b) ** 2.0) / (2.0 * s**2.0))
        return y

    return rfunc


# -----------------------------------------------------------------------------#
def plot_dirtyFDF_ax(
    ax,
    phiArr,
    FDFArr,
    gaussParm=[],
    vLine=None,
    title="Dirty FDF",
    axisYright=False,
    axisXtop=False,
    doTitle=False,
    units="",
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot the FDF
    FDFpiArr = np.sqrt(np.power(FDFArr.real, 2.0) + np.power(FDFArr.imag, 2.0))
    ax.step(phiArr, FDFArr.real, where="mid", color="tab:blue", lw=0.5, label="Real")
    ax.step(
        phiArr, FDFArr.imag, where="mid", color="tab:red", lw=0.5, label="Imaginary"
    )
    ax.step(phiArr, FDFpiArr, where="mid", color="k", lw=1.0, label="PI")
    if doTitle == True:
        ax.text(0.05, 0.84, "Dirty FDF", transform=ax.transAxes)

    # Plot the Gaussian peak
    if len(gaussParm) == 3:
        # [amp, mean, FWHM]
        phiTrunkArr = np.where(
            phiArr >= gaussParm[1] - gaussParm[2] / 3.0, phiArr, np.nan
        )
        phiTrunkArr = np.where(
            phiArr <= gaussParm[1] + gaussParm[2] / 3.0, phiTrunkArr, np.nan
        )
        yGauss = gauss(gaussParm)(phiTrunkArr)
        ax.plot(
            phiArr,
            yGauss,
            color="magenta",
            marker="None",
            mfc="w",
            mec="g",
            ms=10,
            label="Peak Fit",
            lw=2.5,
            ls="-",
        )

    # Plot a vertical line
    if vLine:
        ax.axvline(vLine, color="magenta", ls="--", linewidth=1.5)

    # Scaling
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(8))
    xRange = np.nanmax(phiArr) - np.nanmin(phiArr)
    ax.set_xlim(np.nanmin(phiArr) - xRange * 0.01, np.nanmax(phiArr) + xRange * 0.01)
    ax.set_ylabel("Flux Density (" + units + ")")
    ax.set_xlabel(r"$\phi$ (rad m$^{-2}$)")
    ax.axhline(0, color="grey")

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------#
def plot_cleanFDF_ax(
    ax,
    phiArr,
    cleanFDFArr=None,
    ccFDFArr=None,
    dirtyFDFArr=None,
    gaussParm=[],
    title="Clean FDF",
    cutoff=None,
    axisYright=False,
    axisXtop=False,
    showComplex=True,
    units="",
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot the dirty FDF in the background
    if dirtyFDFArr is not None:
        dirtyFDFpiArr = np.sqrt(
            np.power(dirtyFDFArr.real, 2.0) + np.power(dirtyFDFArr.imag, 2.0)
        )

        ax.step(phiArr, dirtyFDFpiArr, where="mid", color="grey", lw=0.5, label="Dirty")

    # Plot the clean FDF
    if cleanFDFArr is not None:
        ax.step(phiArr, np.abs(cleanFDFArr), where="mid", color="k", lw=1.0, label="PI")
        if showComplex:
            ax.step(
                phiArr,
                cleanFDFArr.real,
                where="mid",
                color="tab:blue",
                lw=0.5,
                label="Real",
            )
            ax.step(
                phiArr,
                cleanFDFArr.imag,
                where="mid",
                color="tab:red",
                lw=0.5,
                label="Imaginary",
            )
            # ax.text(0.05, 0.94, title, transform=ax.transAxes)

    # Plot the CC spectrum
    if ccFDFArr is not None:
        ax.step(phiArr, ccFDFArr, where="mid", color="g", lw=0.5, label="CC")

    # Plot the Gaussian peak
    if len(gaussParm) == 3:
        # [amp, mean, FWHM]
        phiTrunkArr = np.where(
            phiArr >= gaussParm[1] - gaussParm[2] / 3.0, phiArr, np.nan
        )
        phiTrunkArr = np.where(
            phiArr <= gaussParm[1] + gaussParm[2] / 3.0, phiTrunkArr, np.nan
        )
        yGauss = gauss(gaussParm)(phiTrunkArr)
        ax.plot(
            phiArr,
            yGauss,
            color="magenta",
            marker="None",
            mfc="w",
            mec="g",
            ms=10,
            label="Peak Fit",
            lw=2.5,
            ls="-",
        )

    # Plot the clean cutoff line
    if cutoff is not None:
        ax.axhline(cutoff, color="tab:red", ls="--")

    # Scaling
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(phiArr) - np.nanmin(phiArr)
    ax.set_xlim(np.nanmin(phiArr) - xRange * 0.01, np.nanmax(phiArr) + xRange * 0.01)
    ax.set_ylabel("Flux Density (" + units + ")")
    ax.set_xlabel(r"$\phi$ (rad m$^{-2}$)")
    ax.axhline(0, color="grey")

    # Format tweaks
    ax = tweakAxFormat(ax)
    # ax.autoscale_view(True,True,True)


# -----------------------------------------------------------------------------#
def plot_hist4_ax(
    ax,
    popLst,
    nBins=10,
    doXlog=False,
    doYlog=False,
    styIndx=0,
    xMin=None,
    xMax=None,
    yMax=None,
    xLabel="",
    yLabel="",
    title="",
    legLabLst=[],
    legLoc="tr",
    verbose=False,
):
    # Format of the histogram lines and shading.
    # Up to four histograms are supported and two alternative styles
    edgeColourLst = [
        ["black", "red", "blue", "black"],
        ["grey", "red", "blue", "black"],
    ]
    fillColourLst = [
        ["#dddddd", "none", "none", "none"],
        ["none", "none", "none", "none"],
    ]
    hatchStyleLst = [["", "/", "\\", ""], ["/", "\\", "", ""]]
    histLinewidthLst = [[1.0, 1.0, 1.0, 1.5], [1.0, 1.0, 1.0, 1.5]]

    # Translate the legend location code
    if legLoc not in ["tl", "tr", "bl", "br"]:
        legLoc = "tr"
    locTab = {
        "tl": "upper left",
        "tr": "upper right",
        "bl": "lower left",
        "br": "lower right",
    }
    legLoc = locTab[legLoc]

    # TODO: Remove extra columns in the recarrays

    # Determine the max and min of the ensemble population
    popEnsemble = np.concatenate(popLst).astype(np.float)
    xMinData = float(np.nanmin(popEnsemble))
    xMaxData = float(np.nanmax(popEnsemble))

    # All valid data must have the same sign for log plots
    if doXlog:
        if not (xMinData < 0) == (xMaxData < 0):
            print("\nErr: for log axis all data must have the same sign!")
            return
    sign = np.sign(popEnsemble)[0]

    # Calculate the bin edges
    if doXlog:
        logBins = np.linspace(
            m.log10(abs(xMinData)), m.log10(abs(xMaxData)), int(nBins + 1)
        )
        b = np.power(10.0, logBins) * sign
    else:
        b = np.linspace(xMinData, xMaxData, int(nBins + 1))

    # Bin the data in each population
    nLst = []
    for p in popLst:
        n, b = np.histogram(p.astype(np.float), bins=b)
        n = np.array(n, dtype=np.float)
        nLst.append(n)

    # Print the binned values to the screen
    if verbose:
        print("\n#BIN, COUNTS ...")
        binCentreArr = b[:-1] + np.diff(b) / 2.0
        for i in range(len(binCentreArr)):
            print(binCentreArr[i], end=" ")
            for j in range(len(nLst)):
                print(nLst[j][i], end=" ")
            print()

    # Set the Y-axis limits
    nEnsemble = np.concatenate(nLst)
    if doYlog:
        yZeroPt = 0.8
        yMin = yZeroPt
        if yMax is None:
            yMaxData = float(np.nanmax(nEnsemble))
            yFac = abs(yMaxData / yZeroPt)
            yMax = yMaxData * (1 + m.log10(yFac) * 0.3)
    else:
        yZeroPt = 0.0
        yMin = yZeroPt
        if yMax is None:
            yMax = float(np.nanmax(nEnsemble)) * 1.2

    # Set the X-axis limits, incorporating a single padding bin
    xFac = (len(b) - 1) * 0.05
    if doXlog:
        sign = np.sign(b)[0]
        logBins = np.log10(b * sign)
        logBinWidth = np.nanmax(np.diff(logBins))
        if xMin is None:
            xMin = 10 ** (logBins[0] - logBinWidth * xFac) * sign
        if xMax is None:
            xMax = 10 ** (logBins[-1] + logBinWidth * xFac) * sign
    else:
        linBinWidth = np.nanmax(np.diff(b))
        if xMin is None:
            xMin = b[0] - linBinWidth * xFac
        if xMax is None:
            xMax = b[-1] + linBinWidth * xFac

    # Set the axis formatter for log scale axes
    if doXlog:
        ax.set_xscale("symlog")
        majorFormatterX = FuncFormatter(label_format_exp(5.0))
        ax.xaxis.set_major_formatter(majorFormatterX)
    if doYlog:
        ax.set_yscale("symlog")
        majorFormatterY = FuncFormatter(label_format_exp(3.0))
        ax.yaxis.set_major_formatter(majorFormatterY)

    # Create individual histogram polygons. Manually creating histograms gives
    # more control than inbuilt matplotlib function - when originally writing
    # this code the fill styles did not work well.
    for i in range(len(nLst)):
        # Set the legend labels
        try:
            legLabel = legLabLst[i]
            if legLabLst[i] == "":
                raise Exception
        except Exception:
            legLabel = "Query %s" % (i + 1)

        # Create the histograms from line-segments
        polyCoords = mk_hist_poly(b, nLst[i], doYlog, zeroPt=0.7)
        hist = Polygon(
            polyCoords, closed=True, animated=False, linewidth=2.7, label=legLabel
        )
        hist.set_linewidth(histLinewidthLst[styIndx][i])
        hist.set_edgecolor(edgeColourLst[styIndx][i])
        hist.set_facecolor(fillColourLst[styIndx][i])
        hist.set_hatch(hatchStyleLst[styIndx][i])
        ax.add_patch(hist)

    # Set the X axis limits
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)

    # Draw the labels on the plot
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title, size=14)

    # Format tweaks
    tweakAxFormat(ax, showLeg=True, loc=legLoc)


# -----------------------------------------------------------------------------#
def plot_scatter4_ax(
    ax,
    popLst,
    doXlog=False,
    doYlog=False,
    zPower=1.0,
    styIndx=0,
    xMin=None,
    xMax=None,
    yMin=None,
    yMax=None,
    zMin=None,
    zMax=None,
    xLabel="",
    yLabel="",
    zLabel="",
    title="",
    legLabLst=[],
    showCbar=False,
    show11Line=False,
    legLoc="tr",
    verbose=False,
):
    # Format of the scatter points and shading.
    # Up to four populations are supported and two alternative styles
    edgeColourLst = [
        ["black", "black", "black", "black"],
        ["black", "red", "green", "blue"],
    ]
    fillColourLst = [
        ["black", "red", "green", "blue"],
        ["none", "none", "none", "none"],
    ]
    symbolLst = [["o", "s", "d", "^"], ["o", "+", "s", "d"]]
    symbolSize = [[45, 30, 30, 30], [45, 80, 30, 30]]

    # Translate the legend location code
    if legLoc not in ["tl", "tr", "bl", "br"]:
        legLoc = "tr"
    locTab = {
        "tl": "upper left",
        "tr": "upper right",
        "bl": "lower left",
        "br": "lower right",
    }
    legLoc = locTab[legLoc]

    # Separate out the X Y and Z data
    xLst = []
    yLst = []
    zLst = []
    for i in range(len(popLst)):
        colNames = popLst[i].dtype.names
        nCols = len(colNames)
        xLst.append(popLst[i][colNames[0]])
        yLst.append(popLst[i][colNames[1]])
        if nCols > 2:
            yLst.append(popLst[i][colNames[2]])

    # Determine the max and min of the ensemble population
    xEnsemble = np.concatenate(xLst).astype(np.float)
    signX = np.sign(xEnsemble)[0]
    xMinData = float(np.nanmin(xEnsemble))
    xMaxData = float(np.nanmax(xEnsemble))
    yEnsemble = np.concatenate(yLst).astype(np.float)
    signY = np.sign(yEnsemble)[0]
    yMinData = float(np.nanmin(yEnsemble))
    yMaxData = float(np.nanmax(yEnsemble))
    if not zLst == []:
        zEnsemble = np.concatenate(zLst).astype(np.float)
        signZ = np.sign(zEnsemble)[0]
        zMinData = float(np.nanmin(zEnsemble))
        zMaxData = float(np.nanmax(zEnsemble))

    # All valid data must have the same sign for log plots
    if doXlog:
        if not (xMinData < 0) == (xMaxData < 0):
            print("\nErr: for log X-axis all data must have the same sign!")
            sys.exit()
    if doYlog:
        if not (yMinData < 0) == (yMaxData < 0):
            print("\nErr: for log Y-axis all data must have the same sign!")
            sys.exit()
    if zLst is not None and zPower != 1.0:
        if not (zMinData < 0) == (zMaxData < 0):
            print("\nErr: for log Z-axis all data must have the same sign!")
            sys.exit()

    # Set the plotting ranges (& colour limits)
    if doXlog:
        xFac = abs(xMaxData / xMinData)
        if xMin is None:
            xMin = xMinData / (1 + m.log10(xFac) * 0.1)
        if xMax is None:
            xMax = xMaxData * (1 + m.log10(xFac) * 0.1)
    else:
        xPad = abs(xMaxData - xMinData) * 0.04
        if xMin is None:
            xMin = xMinData - xPad
        if xMax is None:
            xMax = xMaxData + xPad
    if doYlog:
        yFac = abs(yMaxData / yMinData)
        if yMin is None:
            yMin = yMinData / (1 + m.log10(yFac) * 0.1)
        if yMax is None:
            yMax = yMaxData * (1 + m.log10(yFac) * 0.1)
    else:
        yPad = abs(yMaxData - yMinData) * 0.05
        if yMin is None:
            yMin = yMinData - yPad
        if yMax is None:
            yMax = yMaxData + yPad

    # Set the z-colour range
    if not zLst == []:
        if not np.all(np.isnan(zEnsemble)):
            if zMin is None:
                zMin = zMinData
            if zMax is None:
                zMax = zMaxData

    # Set the axis formatter for log scale axes
    if doXlog:
        ax.set_xscale("log")
        majorFormatterX = FuncFormatter(label_format_exp(5.0))
        ax.xaxis.set_major_formatter(majorFormatterX)
    if doYlog:
        ax.set_yscale("log")
        majorFormatterY = FuncFormatter(label_format_exp(3.0))
        ax.yaxis.set_major_formatter(majorFormatterY)
    norm = APLpyNormalize(stretch="power", exponent=zPower, vmin=zMin, vmax=zMax)

    # Plot each set of points in turn
    sc3D = None
    zMap = "r"
    for i in range(len(xLst)):
        # Map the z axis to the colours
        if not zLst == []:
            if np.all(np.isnan(zLst[i])):
                zMap = fillColourLst[styIndx][i]
            else:
                zMap = zLst[i]

        # Set the legend labels
        try:
            legLabel = legLabLst[i]
            if legLabLst[i] == "":
                raise Exception
        except Exception:
            legLabel = "Query %s" % (i + 1)

        # Add the points to the plot
        sc = ax.scatter(
            xLst[i],
            yLst[i],
            marker=symbolLst[styIndx][i],
            s=symbolSize[styIndx][i],
            c=zMap,
            norm=norm,
            vmin=zMin,
            vmax=zMax,
            linewidths=0.5,
            edgecolors=edgeColourLst[styIndx][i],
            label=legLabel,
        )
        if not zLst == []:
            if not np.all(np.isnan(zLst[i])):
                sc3D = sc

    # Set the X axis limits
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)

    # Draw the labels on the plot
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)

    # Format tweaks
    tweakAxFormat(ax, showLeg=True, loc=legLoc)


# Axis code above
# =============================================================================#
# Figure code below


# -----------------------------------------------------------------------------#
def plotSpecIPQU(dataMan, indx, io="fig"):
    # Get the data
    freqArr_Hz, IArr, rmsIArr = dataMan.get_specI_byindx(indx)
    dummy, QArr, rmsQArr = dataMan.get_specQ_byindx(indx)
    dummy, UArr, rmsUArr = dataMan.get_specU_byindx(indx)

    # Get the models to overplot
    freqHirArr_Hz, qModArr, uModArr = dataMan.get_thin_qumodel_byindx(
        indx, oversample=True
    )
    freqHirArr_Hz, IModArr = dataMan.get_modI_byindx(indx, oversample=True)
    QmodArr = qModArr * IModArr
    UmodArr = uModArr * IModArr

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot I versus nu,
    ax1 = fig.add_subplot(211)
    plot_I_vs_nu_ax(
        ax=ax1,
        freqArr_Hz=freqArr_Hz,
        IArr=IArr,
        dIArr=rmsIArr,
        freqHirArr_Hz=freqHirArr_Hz,
        IModArr=IModArr,
    )
    ax1.set_xlabel("")
    [label.set_visible(False) for label in ax1.get_xticklabels()]

    # Plot Stokes P, Q & U
    ax2 = fig.add_subplot(212, sharex=ax1)
    plot_PQU_vs_nu_ax(
        ax=ax2,
        freqArr_Hz=freqArr_Hz,
        QArr=QArr,
        UArr=UArr,
        dQArr=rmsQArr,
        dUArr=rmsUArr,
        freqHirArr_Hz=freqHirArr_Hz,
        QmodArr=QmodArr,
        UmodArr=UmodArr,
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotSpecRMS(dataMan, indx, io="fig"):
    # Get the data
    freqArr_Hz, IArr, rmsIArr = dataMan.get_specI_byindx(indx)
    dummy, QArr, rmsQArr = dataMan.get_specQ_byindx(indx)
    dummy, UArr, rmsUArr = dataMan.get_specU_byindx(indx)

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot Stokes I, Q & U
    ax1 = fig.add_subplot(111)
    plot_rmsIQU_vs_nu_ax(
        ax=ax1, freqArr_Hz=freqArr_Hz, rmsIArr=rmsIArr, rmsQArr=rmsQArr, rmsUArr=rmsUArr
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotPolang(dataMan, indx, io="fig"):
    # Get the data
    freqArr_Hz, IArr, rmsIArr = dataMan.get_specI_byindx(indx)
    dummy, modIArr = dataMan.get_modI_byindx(indx)
    dummy, QArr, rmsQArr = dataMan.get_specQ_byindx(indx)
    dummy, UArr, rmsUArr = dataMan.get_specU_byindx(indx)

    # Calculate fractional polarisation spectra
    qArr = QArr / modIArr
    uArr = UArr / modIArr
    dqArr = qArr * np.sqrt((rmsQArr / QArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    duArr = uArr * np.sqrt((rmsUArr / UArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    lamSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)

    # Get the models to overplot
    freqHirArr_Hz, qModArr, uModArr = dataMan.get_thin_qumodel_byindx(
        indx, oversample=True
    )
    lamSqHirArr_m2 = np.power(speed_of_light.value / freqHirArr_Hz, 2.0)

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot psi versus lambda^2
    ax1 = fig.add_subplot(111)
    plot_psi_vs_lamsq_ax(
        ax=ax1,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
        axisYright=False,
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotFracPol(dataMan, indx, io="fig"):
    # Get the data
    freqArr_Hz, IArr, rmsIArr = dataMan.get_specI_byindx(indx)
    dummy, modIArr = dataMan.get_modI_byindx(indx)
    dummy, QArr, rmsQArr = dataMan.get_specQ_byindx(indx)
    dummy, UArr, rmsUArr = dataMan.get_specU_byindx(indx)

    # Calculate fractional polarisation spectra
    qArr = QArr / modIArr
    uArr = UArr / modIArr
    dqArr = qArr * np.sqrt((rmsQArr / QArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    duArr = uArr * np.sqrt((rmsUArr / UArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    lamSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)

    # Get the models to overplot
    freqHirArr_Hz, qModArr, uModArr = dataMan.get_thin_qumodel_byindx(
        indx, oversample=True
    )
    lamSqHirArr_m2 = np.power(speed_of_light.value / freqHirArr_Hz, 2.0)

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot p, q, u versus lambda^2
    ax2 = fig.add_subplot(111)
    plot_pqu_vs_lamsq_ax(
        ax=ax2,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotFracQvsU(dataMan, indx, io="fig"):
    # Get the data
    freqArr_Hz, IArr, rmsIArr = dataMan.get_specI_byindx(indx)
    dummy, modIArr = dataMan.get_modI_byindx(indx)
    dummy, QArr, rmsQArr = dataMan.get_specQ_byindx(indx)
    dummy, UArr, rmsUArr = dataMan.get_specU_byindx(indx)

    # Calculate fractional polarisation spectra
    qArr = QArr / modIArr
    uArr = UArr / modIArr
    dqArr = qArr * np.sqrt((rmsQArr / QArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    duArr = uArr * np.sqrt((rmsUArr / UArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    lamSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)

    # Get the models to overplot
    freqHirArr_Hz, qModArr, uModArr = dataMan.get_thin_qumodel_byindx(
        indx, oversample=True
    )
    lamSqHirArr_m2 = np.power(speed_of_light.value / freqHirArr_Hz, 2.0)

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot U versus Q
    ax1 = fig.add_subplot(111)
    plot_q_vs_u_ax(
        ax=ax1,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
        axisYright=False,
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plot_Ipqu_spectra_fig(
    freqArr_Hz,
    IArr,
    qArr,
    uArr,
    dIArr=None,
    dqArr=None,
    duArr=None,
    freqHirArr_Hz=None,
    IModArr=None,
    qModArr=None,
    uModArr=None,
    model_dict=None,
    fig=None,
    units="",
):
    """Plot the Stokes I, Q/I & U/I spectral summary plots."""

    # Default to a pyplot figure
    if fig == None:
        fig = plt.figure(facecolor="w", figsize=(12.0, 8))

    # Default to non-high-resolution inputs
    if freqHirArr_Hz is None:
        freqHirArr_Hz = freqArr_Hz
    lamSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)
    lamSqHirArr_m2 = np.power(speed_of_light.value / freqHirArr_Hz, 2.0)

    # Plot I versus nu axis
    ax1 = fig.add_subplot(221)
    plot_I_vs_nu_ax(
        ax=ax1,
        freqArr_Hz=freqArr_Hz,
        IArr=IArr,
        dIArr=dIArr,
        freqHirArr_Hz=freqHirArr_Hz,
        IModArr=IModArr,
        axisXtop=True,
        units=units,
    )

    # Plot p, q, u versus lambda^2 axis
    ax2 = fig.add_subplot(223)
    plot_pqu_vs_lamsq_ax(
        ax=ax2,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
        model_dict=model_dict,
    )

    # Plot psi versus lambda^2 axis
    ax3 = fig.add_subplot(222, sharex=ax2)
    plot_psi_vs_lamsq_ax(
        ax=ax3,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
        model_dict=model_dict,
        axisYright=True,
        axisXtop=True,
    )

    # Plot q versus u axis
    ax4 = fig.add_subplot(224)
    plot_q_vs_u_ax(
        ax=ax4,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
        model_dict=model_dict,
        axisYright=True,
    )

    # Adjust subplot spacing
    fig.subplots_adjust(
        left=0.1, bottom=0.08, right=0.90, top=0.92, wspace=0.08, hspace=0.08
    )

    fig.tight_layout()

    return fig


# -----------------------------------------------------------------------------#
def plotPolsummary(dataMan, indx, io="fig"):
    # Get the data
    freqArr_Hz, IArr, rmsIArr = dataMan.get_specI_byindx(indx)
    dummy, modIArr = dataMan.get_modI_byindx(indx)
    dummy, QArr, rmsQArr = dataMan.get_specQ_byindx(indx)
    dummy, UArr, rmsUArr = dataMan.get_specU_byindx(indx)

    # Calculate fractional polarisation spectra
    qArr = QArr / modIArr
    uArr = UArr / modIArr
    dqArr = qArr * np.sqrt((rmsQArr / QArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    duArr = uArr * np.sqrt((rmsUArr / UArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    lamSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)

    # Get the models to overplot
    dummy, modIHirArr = dataMan.get_modI_byindx(indx, oversample=True)
    freqHirArr_Hz, qModArr, uModArr = dataMan.get_thin_qumodel_byindx(
        indx, oversample=True
    )

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Call the function to create the 4-panel figure
    plot_Ipqu_spectra_fig(
        freqArr_Hz=freqArr_Hz,
        IArr=IArr,
        qArr=qArr,
        uArr=uArr,
        dIArr=rmsIArr,
        dqArr=dqArr,
        duArr=duArr,
        freqHirArr_Hz=freqHirArr_Hz,
        IModArr=modIHirArr,
        qModArr=qModArr,
        uModArr=uModArr,
        fig=fig,
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotPolresidual(dataMan, indx, io="fig"):
    # Get the data
    freqArr_Hz, IArr, rmsIArr = dataMan.get_specI_byindx(indx)
    dummy, modIArr = dataMan.get_modI_byindx(indx)
    dummy, QArr, rmsQArr = dataMan.get_specQ_byindx(indx)
    dummy, UArr, rmsUArr = dataMan.get_specU_byindx(indx)
    dummy, qModArr, uModArr = dataMan.get_thin_qumodel_byindx(indx)

    # Calculate fractional polarisation spectra
    qArr = QArr / modIArr
    uArr = UArr / modIArr
    dqArr = qArr * np.sqrt((rmsQArr / QArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    duArr = uArr * np.sqrt((rmsUArr / UArr) ** 2.0 + (rmsIArr / IArr) ** 2.0)
    lamSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)

    # Form the residuals
    qResidArr = qArr - qModArr
    uResidArr = uArr - uModArr
    pResidArr = np.sqrt(qResidArr**2.0 + uResidArr**2.0)
    IResidArr = IArr - modIArr

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot I versus nu,
    ax1 = fig.add_subplot(221)
    plot_I_vs_nu_ax(
        ax=ax1, freqArr_Hz=freqArr_Hz, IArr=IResidArr, dIArr=rmsIArr, axisXtop=True
    )

    # Plot p, q, u versus lambda^2
    ax2 = fig.add_subplot(223)
    plot_pqu_vs_lamsq_ax(
        ax=ax2,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qResidArr,
        uArr=uResidArr,
        dqArr=dqArr,
        duArr=duArr,
    )

    # Plot psi versus lambda^2
    ax3 = fig.add_subplot(222)
    plot_psi_vs_lamsq_ax(
        ax=ax3,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qResidArr,
        uArr=uResidArr,
        dqArr=dqArr,
        duArr=duArr,
        axisYright=True,
        axisXtop=True,
    )

    # Plot U versus Q
    ax4 = fig.add_subplot(224)
    plot_q_vs_u_ax(
        ax=ax4,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qResidArr,
        uArr=uResidArr,
        dqArr=dqArr,
        duArr=duArr,
        axisYright=True,
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plot_rmsf_fdf_fig(
    phiArr,
    FDF,
    phi2Arr,
    RMSFArr,
    fwhmRMSF=None,
    gaussParm=[],
    vLine=None,
    fig=None,
    units="flux units",
):
    """Plot the RMSF and FDF on a single figure."""

    # Default to a pyplot figure
    if fig == None:
        fig = plt.figure(facecolor="w", figsize=(12.0, 8))
    # Plot the RMSF
    ax1 = fig.add_subplot(211)
    plot_RMSF_ax(
        ax=ax1, phiArr=phi2Arr, RMSFArr=RMSFArr, fwhmRMSF=fwhmRMSF, doTitle=True
    )
    [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax1.set_xlabel("")

    # Plot the FDF
    # Why are these next two lines here? Removing as part of units fix.
    #    if len(gaussParm)==3:
    #        gaussParm[0] *= 1e3
    ax2 = fig.add_subplot(212, sharex=ax1)
    plot_dirtyFDF_ax(
        ax=ax2,
        phiArr=phiArr,
        FDFArr=FDF,
        gaussParm=gaussParm,
        vLine=vLine,
        doTitle=True,
        units=units,
    )

    return fig


# -----------------------------------------------------------------------------#
def plotRMSF(dataMan, indx, io="fig"):
    # Get the data and Gaussian fit to RMSF
    phiArr, RMSFArr = dataMan.get_RMSF_byindx(indx)
    pDict = dataMan.get_RMSF_params_byindx(indx)

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot the RMSF
    ax1 = fig.add_subplot(111)
    plot_RMSF_ax(ax=ax1, phiArr=phiArr, RMSFArr=RMSFArr, fwhmRMSF=pDict["fwhmRMSF"])

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotDirtyFDF(dataMan, indx, io="fig"):
    # Get the data
    phiArr, FDFArr = dataMan.get_dirtyFDF_byindx(indx)

    # Get the peak results
    pDict = dataMan.get_FDF_peak_params_byindx(indx)
    pDict1 = dataMan.get_RMSF_params_byindx(indx)
    gaussParm = [pDict["ampPeakPIfit"], pDict["phiPeakPIfit_rm2"], pDict1["fwhmRMSF"]]

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot the FDF
    ax1 = fig.add_subplot(111)
    plot_dirtyFDF_ax(
        ax=ax1,
        phiArr=phiArr,
        FDFArr=FDFArr,
        gaussParm=gaussParm,
        title="Dirty Faraday Dispersion Function",
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotCleanFDF(dataMan, indx, io="fig"):
    # Get the data
    phiArr, dirtyFDFArr = dataMan.get_dirtyFDF_byindx(indx)
    dummy, cleanFDFArr = dataMan.get_cleanFDF_byindx(indx)
    dummy, ccFDF = dataMan.get_ccFDF_byindx(indx)

    # Get the peak results
    pDict = dataMan.get_FDF_peak_params_byindx(indx, doClean=True)
    pDict1 = dataMan.get_RMSF_params_byindx(indx)
    gaussParm = [pDict["ampPeakPIfit"], pDict["phiPeakPIfit_rm2"], pDict1["fwhmRMSF"]]

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    # Plot the clean FDF
    ax1 = fig.add_subplot(111)
    plot_cleanFDF_ax(
        ax=ax1,
        phiArr=phiArr,
        cleanFDFArr=cleanFDFArr,
        ccFDFArr=ccFDF,
        dirtyFDFArr=dirtyFDFArr,
        gaussParm=gaussParm,
        title="Clean Faraday Dispersion Function",
        cutoff=pDict["cleanCutoff"],
    )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plot_cleanFDF_fig(
    phiArr,
    cleanFDFArr,
    ccFDFArr=None,
    dirtyFDFArr=None,
    residFDFArr=None,
    gaussParm=[],
    title="Clean FDF",
    cutoff=None,
    showComplex=True,
    fig=None,
):
    # Default to a pyplot figure
    if fig == None:
        fig = plt.figure(facecolor="w", figsize=(12.0, 8))

    # Plot the clean FDF
    ax1 = fig.add_subplot(211)
    plot_cleanFDF_ax(
        ax=ax1,
        phiArr=phiArr,
        cleanFDFArr=cleanFDFArr,
        ccFDFArr=ccFDFArr,
        dirtyFDFArr=dirtyFDFArr,
        gaussParm=gaussParm,
        title=title,
        cutoff=cutoff,
        showComplex=False,
    )

    # Plot the residual
    ax2 = fig.add_subplot(212)
    plot_cleanFDF_ax(
        ax=ax2,
        phiArr=phiArr,
        cleanFDFArr=cleanFDFArr,
        ccFDFArr=ccFDFArr,
        dirtyFDFArr=residFDFArr,
        gaussParm=gaussParm,
        title=title,
        cutoff=cutoff,
        showComplex=False,
    )

    return fig


# -----------------------------------------------------------------------------#
def plotStampI(dataMan, indx, io="fig"):
    # Get the data & header of the saved postage stamp
    data, head = dataMan.get_stampI_byindx(indx)

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    fig = plot_fits_map(data, head, fig=fig)

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotStampP(dataMan, indx, io="fig"):
    # Get the data & header of the saved postage stamp
    data, head = dataMan.get_stampP_byindx(indx)

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])

    fig = plot_fits_map(data, head, fig=fig)

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def plotSctHstQuery(dataMan, plotParm, io="fig"):
    # What type of plot are we creating?
    plotType = plotParm.configDict.get("TYPE", "Histogram")

    # Execute each query in turn and store results in list of recarrays
    popLst = []
    names = []
    nCols = 0
    for i in range(len(plotParm.queryLst) - 1, -1, -1):
        sql = plotParm.queryLst[i]
        try:
            resultArr = dataMan.query_database(sql)
            colNames = resultArr.dtype.names
            nCols = max(len(colNames), nCols)
            popLst.append(resultArr)
        except Exception:
            popLst.append(None)
            print("\nWarn: failed to execute query:")
            print("'%s'\n" % sql)
            print(traceback.format_exc(), "\n")
    popLst.reverse()
    popLst = popLst[:4]

    # Filter data for limits given in the driving file (default None)
    xMinDataCmd = plotParm.configDict.get("XDATAMIN", None)
    xMaxDataCmd = plotParm.configDict.get("XDATAMAX", None)
    xMinData = xfloat(xMinDataCmd, None)
    xMaxData = xfloat(xMaxDataCmd, None)
    yMinDataCmd = plotParm.configDict.get("YDATAMIN", None)
    yMaxDataCmd = plotParm.configDict.get("YDATAMAX", None)
    yMinData = xfloat(yMinDataCmd, None)
    yMaxData = xfloat(yMaxDataCmd, None)
    zMinDataCmd = plotParm.configDict.get("ZDATAMIN", None)
    zMaxDataCmd = plotParm.configDict.get("ZDATAMAX", None)
    zMinData = xfloat(zMinDataCmd, None)
    zMaxData = xfloat(zMaxDataCmd, None)
    for i in range(len(popLst)):
        msk = filter_range_indx(popLst[i][colNames[0]], xMinData, xMaxData)
        if plotType == "Scatter" and nCols > 1:
            msk += filter_range_indx(popLst[i][colNames[1]], yMinData, yMaxData)
        if plotType == "Scatter" and nCols > 2:
            msk += filter_range_indx(popLst[i][colNames[2]], zMinData, zMaxData)
        popLst[i] = popLst[i][~msk]

    # Labels from driving file (default column name in DB)
    if plotParm.configDict["XLABEL"] == "":
        plotParm.configDict["XLABEL"] = colNames[0]
    xLabel = plotParm.configDict.get("XLABEL", colNames[0])
    if plotParm.configDict["YLABEL"] == "":
        if plotType == "Scatter":
            plotParm.configDict["YLABEL"] = colNames[1]
        else:
            plotParm.configDict["YLABEL"] = "Count"
    yLabel = plotParm.configDict.get("YLABEL", "Count")
    if plotParm.configDict["ZLABEL"] == "":
        if plotType == "Scatter":
            plotParm.configDict["ZLABEL"] = colNames[2]
    zLabel = plotParm.configDict.get("ZLABEL", "")
    plotTitle = plotParm.configDict.get("TITLE", "")

    # Other driving parameters
    nBins = xint(plotParm.configDict.get("NBINS", 10))
    doXlog = xint(plotParm.configDict.get("DOLOGX", 0))
    doYlog = xint(plotParm.configDict.get("DOLOGY", 0))
    zPower = xfloat(plotParm.configDict.get("ZPOWER", 1.0))

    # Setup the figure
    fig = Figure()
    fig.set_size_inches([8, 8])
    ax = fig.add_subplot(111)

    # Bin the data and create the histogram
    if plotType == "Histogram":
        plot_hist4_ax(
            ax,
            popLst=popLst,
            nBins=nBins,
            doXlog=doXlog,
            doYlog=doYlog,
            styIndx=0,
            xMin=None,
            xMax=None,
            yMax=None,
            xLabel=xLabel,
            yLabel=yLabel,
            title=plotTitle,
            legLabLst=plotParm.queryLabLst,
        )
    if plotType == "Scatter":
        plot_scatter4_ax(
            ax,
            popLst=popLst,
            doXlog=doXlog,
            doYlog=doYlog,
            zPower=zPower,
            styIndx=0,
            xMin=None,
            xMax=None,
            yMin=None,
            yMax=None,
            zMin=None,
            zMax=None,
            xLabel=xLabel,
            yLabel=yLabel,
            zLabel=zLabel,
            title=plotTitle,
            legLabLst=plotParm.queryLabLst,
            showCbar=False,
            show11Line=False,
            legLoc="tr",
            verbose=False,
        )

    # Write to the pipe
    if io == "string":
        sio = io.StringIO()
        setattr(sio, "name", "foo.jpg")
        fig.savefig(sio, format="jpg")
        return sio
    else:
        return fig


# -----------------------------------------------------------------------------#
def mk_hist_poly(bins, n, logScaleY=False, zeroPt=0.8, addZeroPt=True):
    """Create the line segments for the a polygon used to draw a histogram"""

    if logScaleY is True:
        for i in range(len(n)):
            if n[i] <= 0.0:
                n[i] = zeroPt
    else:
        zeroPt = 0.0

    # Starting position
    polyCoordLst = []
    if addZeroPt:
        polyCoordLst.append([bins[0], zeroPt])

    # Form the line segments
    i = 0
    j = 0
    while i <= len(bins) - 1:
        if j < len(n):
            polyCoordLst.append([bins[i], n[j]])
        if i == j:
            i += 1
        else:
            j += 1

    # Ground the polygon line and close
    if addZeroPt:
        polyCoordLst.append([bins[-1], zeroPt])
        polyCoordLst.append([bins[0], zeroPt])
    polyCoords = np.array(polyCoordLst)

    return polyCoords


# -----------------------------------------------------------------------------#
def label_format_exp(switchExp=3.0):
    """Return a function to format labels for log axes. Switches to a power
    format for log10(|number|) >= switchExp."""

    def rfunc(num, pos=None):
        absNum = 0.0
        sign = ""
        exponent = 0.0
        if num != 0.0:
            absNum = abs(num)
            sign = "-" if int(num / absNum) < 0 else ""
            exponent = m.log10(absNum)
        if abs(exponent) >= switchExp:
            return r"$%s10^{%i}$" % (sign, m.log10(absNum))
        else:
            return r"$%s%g$" % (sign, absNum)

    return rfunc


# -----------------------------------------------------------------------------#
def plot_complexity_fig(
    xArr,
    qArr,
    dqArr,
    sigmaAddqArr,
    chiSqRedqArr,
    probqArr,
    uArr,
    duArr,
    sigmaAdduArr,
    chiSqReduArr,
    probuArr,
    mDict,
    med=0.0,
    noise=1.0,
    fig=None,
):
    """Create the residual Stokes q and u complexity plots."""

    # Default to a pyplot figure
    if fig == None:
        fig = plt.figure(facecolor="w", figsize=(16.0, 8.0))

    # Plot the data and the +/- 1-sigma levels
    ax1 = fig.add_subplot(231)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.set_major_locator(MaxNLocator(7))
    ax1.errorbar(
        x=xArr,
        y=qArr,
        yerr=dqArr,
        ms=3,
        color="tab:blue",
        fmt="o",
        alpha=0.5,
        capsize=0,
    )
    ax1.errorbar(
        x=xArr, y=uArr, yerr=duArr, ms=3, color="tab:red", fmt="o", alpha=0.5, capsize=0
    )
    ax1.axhline(med, color="grey", zorder=10)
    ax1.axhline(1.0, color="k", linestyle="--", zorder=10)
    ax1.axhline(-1.0, color="k", linestyle="--", zorder=10)
    ax1.set_xlabel(r"$\lambda^2$")
    ax1.set_ylabel("Normalised Residual")

    # Plot the histogram of the data overlaid by the normal distribution
    H = 1.0 / np.sqrt(2.0 * np.pi * noise**2.0)
    xNorm = np.linspace(med - 3 * noise, med + 3 * noise, 1000)
    yNorm = H * np.exp(-0.5 * ((xNorm - med) / noise) ** 2.0)
    fwhm = noise * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    ax2 = fig.add_subplot(232)
    ax2.tick_params(labelbottom="off")
    nBins = 15
    yMin = np.nanmin([np.nanmin(qArr), np.nanmin(uArr)])
    yMax = np.nanmax([np.nanmax(qArr), np.nanmax(uArr)])
    n, b, p = ax2.hist(
        qArr,
        nBins,
        range=(yMin, yMax),
        density=1,
        histtype="step",
        color="tab:blue",
        linewidth=1.0,
    )
    ax2.plot(xNorm, yNorm, color="k", linestyle="--", linewidth=1.5)
    n, b, p = ax2.hist(
        uArr,
        nBins,
        range=(yMin, yMax),
        density=1,
        histtype="step",
        color="tab:red",
        linewidth=1.0,
    )
    ax2.axvline(med, color="grey", zorder=11)
    ax2.set_title(r"Distribution of Data vs Normal")
    ax2.set_ylabel(r"Normalised Counts")

    # Plot the ECDF versus a normal CDF
    nData = len(xArr)
    ecdfArr = np.array(list(range(nData))) / float(nData)
    qSrtArr = np.sort(qArr)
    uSrtArr = np.sort(uArr)
    ax3 = fig.add_subplot(235, sharex=ax2)
    ax3.step(qSrtArr, ecdfArr, where="mid", color="tab:blue")
    ax3.step(uSrtArr, ecdfArr, where="mid", color="tab:red")
    x, y = norm_cdf(mean=med, std=noise, N=1000)
    ax3.plot(x, y, color="k", linewidth=1.5, linestyle="--", zorder=1)
    ax3.set_ylim(0, 1.05)
    ax3.axvline(med, color="grey", zorder=11)
    ax3.set_xlabel(r"Normalised Residual")
    ax3.set_ylabel(r"Normalised Counts")

    # Plot reduced chi-squared
    ax4 = fig.add_subplot(234)
    ax4.step(
        x=sigmaAddqArr, y=chiSqRedqArr, color="tab:blue", linewidth=1.0, where="mid"
    )
    ax4.step(
        x=sigmaAddqArr, y=chiSqReduArr, color="tab:red", linewidth=1.0, where="mid"
    )
    ax4.axhline(1.0, color="k", linestyle="--")
    ax4.set_xlabel(r"$\sigma_{\rm add}$")
    ax4.set_ylabel(r"$\chi^2_{\rm reduced}$")

    # Plot the probability distribution function
    ax5 = fig.add_subplot(233)
    ax5.tick_params(labelbottom="off")
    ax5.step(
        x=sigmaAddqArr,
        y=probqArr,
        linewidth=1.0,
        where="mid",
        color="tab:blue",
        alpha=0.5,
    )
    ax5.step(
        x=sigmaAdduArr,
        y=probuArr,
        linewidth=1.0,
        where="mid",
        color="tab:red",
        alpha=0.5,
    )
    ax5.axvline(mDict["sigmaAddQ"], color="tab:blue", linestyle="-", linewidth=1.5)
    ax5.axvline(
        mDict["sigmaAddQ"] + mDict["dSigmaAddPlusQ"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    ax5.axvline(
        mDict["sigmaAddQ"] - mDict["dSigmaAddMinusQ"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    ax5.axvline(mDict["sigmaAddU"], color="tab:red", linestyle="-", linewidth=1.5)
    ax5.axvline(
        mDict["sigmaAddU"] + mDict["dSigmaAddPlusU"],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    ax5.axvline(
        mDict["sigmaAddU"] - mDict["dSigmaAddMinusU"],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    ax5.set_title("Likelihood Distribution")
    ax5.set_ylabel(r"P($\sigma_{\rm add}$|data)")

    # Plot the CPDF
    CPDFq = np.cumsum(probqArr) / np.sum(probqArr)
    CPDFu = np.cumsum(probuArr) / np.sum(probuArr)
    ax6 = fig.add_subplot(236, sharex=ax5)
    ax6.step(x=sigmaAddqArr, y=CPDFq, linewidth=1.0, where="mid", color="tab:blue")
    ax6.step(x=sigmaAdduArr, y=CPDFu, linewidth=1.0, where="mid", color="tab:red")
    ax6.set_ylim(0, 1.05)
    ax6.axhline(0.5, color="grey", linestyle="-", linewidth=1.5)
    ax6.axvline(mDict["sigmaAddQ"], color="tab:blue", linestyle="-", linewidth=1.5)
    ax6.axvline(
        mDict["sigmaAddQ"] + mDict["dSigmaAddPlusQ"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    ax6.axvline(
        mDict["sigmaAddQ"] - mDict["dSigmaAddMinusQ"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    ax6.axvline(mDict["sigmaAddU"], color="tab:red", linestyle="-", linewidth=1.5)
    ax6.axvline(
        mDict["sigmaAddU"] + mDict["dSigmaAddPlusU"],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    ax6.axvline(
        mDict["sigmaAddU"] - mDict["dSigmaAddMinusU"],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    ax6.set_xlabel(r"$\sigma_{\rm add}$")
    ax6.set_ylabel(r"Cumulative Likelihood")

    # Zoom in
    xLim1 = np.nanmin(
        [
            mDict["sigmaAddQ"] - mDict["dSigmaAddMinusQ"] * 4.0,
            mDict["sigmaAddU"] - mDict["dSigmaAddMinusU"] * 4.0,
        ]
    )
    xLim1 = max(0.0, xLim1)
    xLim2 = np.nanmax(
        [
            mDict["sigmaAddQ"] + mDict["dSigmaAddPlusQ"] * 4.0,
            mDict["sigmaAddU"] + mDict["dSigmaAddPlusU"] * 4.0,
        ]
    )
    ax6.set_xlim(xLim1, xLim2)

    # Show the figure
    fig.subplots_adjust(
        left=0.07, bottom=0.09, right=0.97, top=0.92, wspace=0.25, hspace=0.05
    )

    return fig
