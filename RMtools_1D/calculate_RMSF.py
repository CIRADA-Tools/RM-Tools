#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:01:48 2019

@author: cvaneck

This routine will determine the RMSF and related parameters,
giving the following input information. One of:
a file with channel frequencies and weights
OR
a file with channel frequencies (assumes equal weights)
OR
Input values for mininum frequency, maximum frequency, and channel width.
(assumes equal weights and all channels present)

The outputs are a list of relavant RMSF properties, and a plot of the RMSF
shape.
"""

# import sys
import argparse

import numpy as np
from astropy.constants import c as speed_of_light
from matplotlib import pyplot as plt

from RMutils.util_RM import get_rmsf_planes


def main():
    """
    Determines what set of input parameters were defined, reads in file or
    generates frequency array as appropriate, and passes frequency and weight
    arrays to the function that works out the RMSF properties.
    """

    descStr = """
    Calculate and plot RMSF and report main properties, given a supplied
    frequency coverage and optional weights (either as second column of
    frequency file, or as separate file)."""

    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "freqFile",
        metavar="freqFile.dat",
        nargs="?",
        default=None,
        help="ASCII file containing frequencies and optionally weights.",
    )
    parser.add_argument(
        "weightFile",
        metavar="weightFile.dat",
        nargs="?",
        help="Optional ASCII file containing weights.",
    )
    parser.add_argument(
        "-f",
        dest=("freq_parms"),
        nargs=3,
        default=None,
        help="Generate frequencies (in Hz): minfreq, maxfreq, channel_width",
    )
    parser.add_argument(
        "-m",
        dest="phiMax_radm2",
        type=float,
        default=None,
        help="absolute max Faraday depth sampled [Auto, ~10xFWHM].",
    )
    parser.add_argument(
        "-d",
        dest="dphi_radm2",
        type=float,
        default=None,
        help="Delta phi [Auto, ~10/FWHM].",
    )
    parser.add_argument(
        "-s",
        dest="plotfile",
        default=None,
        help="Filename to save plot to. [do not save]",
    )
    parser.add_argument(
        "-n", dest="plotname", default=None, help='Name of plot ["Simulated RMSF"]'
    )
    parser.add_argument(
        "-r",
        "--super-resolution",
        action="store_true",
        help="Optimise the resolution of the RMSF (as per Rudnick & Cotton). ",
    )
    args = parser.parse_args()

    # Check that at least one frequency input has been given:
    if args.freqFile == None and args.freq_parms == None:
        print("Please supply either a file with frequency values or use the -f flag.")
        raise (Exception("No frequency input! Use -h flag for help on inputs."))

    # if args.phiMax_radm2 != None:
    #     if args.phiMax_radm2

    # Order of priority: frequency file takes precedence over -i flag.
    #                   weight file takes precedence over 2nd column of frequency file.
    if args.freqFile != None:
        data = np.genfromtxt(args.freqFile, encoding=None, dtype=None)
        if len(data.shape) == 2:
            freq_array = data[:, 0]
            weights_array = data[:, 1]
        else:
            freq_array = data
            weights_array = np.ones_like(freq_array)
    else:
        # Generate frequency and weight arrays from intput values.
        freq_array = np.arange(
            float(args.freq_parms[0]),
            float(args.freq_parms[1]),
            float(args.freq_parms[2]),
        )
        weights_array = np.ones_like(freq_array)

    if args.weightFile != None:
        weights_array = np.genfromtxt(args.weightFile, encoding=None, dtype=None)
        if len(weights_array) != len(freq_array):
            raise Exception(
                "Weights file does not have same number of channels as frequency source"
            )

    determine_RMSF_parameters(
        freq_array,
        weights_array,
        args.phiMax_radm2,
        args.dphi_radm2,
        args.plotfile,
        args.plotname,
        args.super_resolution,
    )


def determine_RMSF_parameters(
    freq_array,
    weights_array,
    phi_max,
    dphi,
    plotfile=None,
    plotname=None,
    super_resolution=False,
):
    """
    Characterizes an RMSF given the supplied frequency and weight arrays.
    Prints the results to terminal and produces a plot.
    Inputs:
        freq_array: array of frequency values (in Hz)
        weights_array: array of channel weights (arbitrary units)
        phi_max (float): maximum Faraday depth to compute RMSF out to.
        dphi (float): step size in Faraday depth
        plotfile (str): file name and path to save RMSF plot.
        plotname (str): title of plot
    """
    lambda2_array = speed_of_light.value**2 / freq_array**2
    l2_min = np.min(lambda2_array)
    l2_max = np.max(lambda2_array)
    dl2 = np.median(np.abs(np.diff(lambda2_array)))

    if phi_max == None:
        phi_max = 10 * 2 * np.sqrt(3.0) / (l2_max - l2_min)  # ~10*FWHM
    if dphi == None:
        dphi = 0.1 * 2 * np.sqrt(3.0) / (l2_max - l2_min)  # ~10*FWHM

    phi_array = np.arange(
        -1 * phi_max / 2, phi_max / 2 + 1e-6, dphi
    )  # division by two accounts for how RMSF is always twice as wide as FDF.

    rmsf_results = get_rmsf_planes(
        lambda2_array,
        phi_array,
        weightArr=weights_array,
        fitRMSF=True,
        fitRMSFreal=super_resolution,
        lam0Sq_m2=0 if super_resolution else None,
    )

    # Output key results to terminal:
    print("RMSF PROPERTIES:")
    print(
        "Theoretical (unweighted) FWHM:       {:.4g} rad m^-2".format(
            3.8 / (l2_max - l2_min)
        )
    )
    print(
        "Measured FWHM:                       {:.4g} rad m^-2".format(
            rmsf_results.fwhmRMSFArr
        )
    )
    print("Theoretical largest FD scale probed: {:.4g} rad m^-2".format(np.pi / l2_min))
    print(
        "Theoretical maximum FD*:             {:.4g} rad m^-2".format(
            np.sqrt(3.0) / dl2
        )
    )
    print(
        "*50% bandwdith depolarization threshold, for median channel width in Delta-lambda^2"
    )
    print(
        "* may not be reliable over very large fractional bandwidths or in data with "
    )
    print("differing channel widths or many frequency gaps.")
    # Explanation for below: This code find the local maxima in the positive half of the RMSF,
    # finds the highest amplitude one, and calls that the first sidelobe.
    try:
        x = np.diff(
            np.sign(
                np.diff(
                    np.abs(rmsf_results.RMSFcube[rmsf_results.RMSFcube.size // 2 :])
                )
            )
        )  # -2=local max, +2=local min
        y = (
            1 + np.where(x == -2)[0]
        )  # indices of peaks, +1 is because of offset from double differencing
        peaks = np.abs(rmsf_results.RMSFcube[rmsf_results.RMSFcube.size // 2 :])[y]
        print(
            "First sidelobe FD and amplitude:     {:.4g} rad m^-2".format(
                rmsf_results.phi2Arr[rmsf_results.phi2Arr.size // 2 :][
                    y[np.argmax(peaks)]
                ]
            )
        )
        print(
            "                                     {:.4g} % of peak".format(
                np.max(peaks) * 100
            )
        )
    except:
        pass

    # Plotting:
    plt.figure(figsize=(7, 7))
    plt.subplot(211)
    plt.axhline(0, color="k")
    if plotname == None:
        plt.title("Simulated RMSF")
    else:
        plt.title(plotname)
    plt.plot(
        rmsf_results.phi2Arr, np.real(rmsf_results.RMSFcube), "b-", label="Stokes Q"
    )
    plt.plot(
        rmsf_results.phi2Arr, np.imag(rmsf_results.RMSFcube), "r--", label="Stokes U"
    )
    plt.plot(
        rmsf_results.phi2Arr, np.abs(rmsf_results.RMSFcube), "k-", label="Amplitude"
    )
    plt.legend()
    plt.xlabel(r"Faraday depth (rad m$^{-2}$)")
    plt.ylabel("RMSF (unitless)")
    plt.subplot(212)
    ax = plt.gca()
    ax.axis([0, 1, 0, 1])
    ax.axis("off")
    ax.text(
        0.1,
        0.8,
        (
            "Theoretical (unweighted) FWHM:       {:.4g} rad m^-2\n"
            + "Measured FWHM:                       {:.4g} rad m^-2\n"
            + "Theoretical largest FD scale probed: {:.4g} rad m^-2\n"
            + "Theoretical maximum FD:              {:.4g} rad m^-2\n"
            + "\n\n\n"
            + "Lowest frequency/wavelength [GHz/cm]:  {:>7.4g}/{:.4g}\n"
            + "Highest frequency/wavelength [GHz/cm]: {:>7.4g}/{:.4g}\n"
            + "# of channels:                                {:.4g}\n"
        ).format(
            3.8 / (l2_max - l2_min),
            rmsf_results.fwhmRMSFArr,
            np.pi / l2_min,
            np.sqrt(3.0) / dl2,
            np.min(freq_array) / 1e9,
            speed_of_light.value / np.min(freq_array) * 100.0,
            np.max(freq_array) / 1e9,
            speed_of_light.value / np.max(freq_array) * 100.0,
            freq_array.size,
        ),
        family="monospace",
        horizontalalignment="left",
        verticalalignment="top",
    )

    try:
        ax.text(
            0.1,
            0.8,
            (
                "\n\n\n\n"
                + "First sidelobe FD and amplitude:     {:.4g} rad m^-2\n"
                + "                                     {:.4g} % of peak"
            ).format(
                rmsf_results.phi2Arr[rmsf_results.phi2Arr.size // 2 :][
                    y[np.argmax(peaks)]
                ],
                np.max(peaks) * 100,
            ),
            family="monospace",
            horizontalalignment="left",
            verticalalignment="top",
        )
    except:
        pass

    #    ax.text(0.,0.7,('Theoretical (unweighted) FWHM:      {:.4g} rad m^-2'.format(2*np.sqrt(3.0) / (l2_max-l2_min)))
    #    ax.text(0.,0.58,'Measured FWHM:                               {:.4g} rad m^-2'.format(fwhmRMSFArr))
    #    ax.text(0.,0.46,'Theoretical largest FD scale probed: {:.4g} rad m^-2'.format(np.pi/l2_min))
    #    ax.text(0.,0.34,'Theoretical maximum FD:                 {:.4g} rad m^-2'.format(np.sqrt(3.0)/dl2))
    #    ax.text(0.,0.22,'First sidelobe FD and amplitude:       {:.4g} rad m^-2'.format(phi2Arr[phi2Arr.size//2:][y[np.argmax(peaks)]]))
    #    ax.text(0.,0.1,'                                                           {:.4g} % of peak'.format(np.max(peaks)*100))

    if plotfile is not None:
        plt.savefig(plotfile, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
