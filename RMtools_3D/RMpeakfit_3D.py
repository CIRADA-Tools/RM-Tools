#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# NAME:     RMpeakfit_3D.py                                                   #
#                                                                             #
# PURPOSE:  Fit peak of RM spectra, for every pixel in 3D FDF cube.           #
#
# Initial version: Cameron Van Eck, Dec 2020
"""

import os
import sys

import astropy.io.fits as pf
import numpy as np
from astropy.constants import c as speed_of_light
from tqdm.auto import trange

from RMtools_3D.do_RMsynth_3D import readFitsCube, readFreqFile
from RMutils.util_misc import interp_images, remove_header_third_fourth_axis
from RMutils.util_RM import fits_make_lin_axis, measure_FDF_parms


def pixelwise_peak_fitting(
    FDF,
    phiArr,
    fwhmRMSF,
    lamSqArr_m2,
    lam0Sq,
    product_list,
    noiseArr=None,
    stokesIcube=None,
    weightType="uniform",
):
    """
    Performs the 1D FDF peak fitting used in RMsynth/RMclean_1D, pixelwise on
    all pixels in a 3D FDF cube.

    Inputs:
        FDF: FDF cube (3D array). This is assumed to be in astropy axis ordering
            (Phi, dec, ra)
        phiArr: (1D) array of phi values
        fwhmRMSF: 2D array of RMSF FWHM values
        lamSqArr_m2: 1D array of channel lambda^2 values.
        lam0Sq: scalar value for lambda^2_0, the reference wavelength squared.
        product_list: list containing the names of the fitting products to save.
        dFDF: 2D array of theoretical noise values. If not supplied, the
            peak fitting will default to using the measured noise.
    Outputs: dictionary of 2D maps, 1 per fit output
    """

    # FDF: output by synth3d or clean3d
    # phiArr: can be generated from FDF cube
    # fwhm: 2D map produced synth3D
    # dFDFth: not currently produced (default mode not to input noise!)
    #   If not present, measure_FDF_parms uses the corMAD noise.
    #
    # lamSqArr is only needed for computing errors in derotated angles
    #   This could be compressed to a map or single value from RMsynth?
    # lam0Sq is necessary for de-rotation

    map_size = FDF.shape[1:]

    # Create pixel location arrays:
    xarr, yarr = np.meshgrid(range(map_size[0]), range(map_size[1]))
    xarr = xarr.ravel()
    yarr = yarr.ravel()

    # Create empty maps:
    map_dict = {}
    for parameter in product_list:
        map_dict[parameter] = np.zeros(map_size)

    freqArr_Hz = speed_of_light.value / np.sqrt(lamSqArr_m2)
    freq0_Hz = speed_of_light.value / np.sqrt(lam0Sq)
    if stokesIcube is not None:
        idx = np.abs(freqArr_Hz - freq0_Hz).argmin()
        if freqArr_Hz[idx] < freq0_Hz:
            Ifreq0Arr = interp_images(
                stokesIcube[idx, :, :], stokesIcube[idx + 1, :, :], f=0.5
            )
        elif freqArr_Hz[idx] > freq0_Hz:
            Ifreq0Arr = interp_images(
                stokesIcube[idx - 1, :, :], stokesIcube[idx, :, :], f=0.5
            )
        else:
            Ifreq0Arr = stokesIcube[idx, :, :]
    else:
        Ifreq0Arr = np.ones(map_size)
        stokesIcube = np.ones((freqArr_Hz.size, map_size[0], map_size[1]))

    # compute weights if needed:
    if noiseArr is not None:
        if weightType == "variance":
            weightArr = 1.0 / np.power(noiseArr, 2.0)
            weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)
        elif weightType == "uniform":
            weightArr = np.ones(lamSqArr_m2.shape, dtype=np.float32)
            weightArr = np.where(np.isnan(noiseArr), 0.0, weightArr)
        else:
            raise Exception("Invalid weight type; must be 'uniform' or 'variance'")

        dFDF = Ifreq0Arr * np.sqrt(
            np.sum(weightArr**2 * np.nan_to_num(noiseArr) ** 2)
            / (np.sum(weightArr)) ** 2
        )

    else:
        weightArr = np.ones(lamSqArr_m2.shape, dtype=np.float32)
        dFDF = None

    # Run fitting pixel-wise:
    for i in trange(xarr.size):
        FDF_pix = FDF[:, xarr[i], yarr[i]]
        fwhmRMSF_pix = fwhmRMSF[xarr[i], yarr[i]]
        if type(dFDF) == type(None):
            dFDF_pix = None
        else:
            dFDF_pix = dFDF[xarr[i], yarr[i]]
        try:
            mDict = measure_FDF_parms(
                FDF_pix,
                phiArr,
                fwhmRMSF_pix,
                dFDF=dFDF_pix,
                lamSqArr_m2=lamSqArr_m2,
                lam0Sq=lam0Sq,
                snrDoBiasCorrect=5.0,
            )
            # Add keywords not included by the above function:
            mDict["lam0Sq_m2"] = lam0Sq
            mDict["freq0_Hz"] = freq0_Hz
            mDict["fwhmRMSF"] = fwhmRMSF_pix
            mDict["Ifreq0"] = Ifreq0Arr[xarr[i], yarr[i]]
            mDict["fracPol"] = mDict["ampPeakPIfit"] / mDict["Ifreq0"]
            mDict["min_freq"] = float(np.min(freqArr_Hz))
            mDict["max_freq"] = float(np.max(freqArr_Hz))
            mDict["N_channels"] = lamSqArr_m2.size
            mDict["median_channel_width"] = float(np.median(np.diff(freqArr_Hz)))
            if dFDF_pix is not None:
                mDict["dFDFth"] = dFDF_pix
            else:
                mDict["dFDFth"] = np.nan
            for parameter in product_list:
                map_dict[parameter][xarr[i], yarr[i]] = mDict[parameter]
        except:
            for parameter in product_list:
                map_dict[parameter][xarr[i], yarr[i]] = np.nan

    return map_dict


def delete_FITSheader_axis(fitsheader, axis_number):
    """Deletes FITS keywords associated with the specified axis."""
    axis_keywords = ["NAXIS", "CRVAL", "CRPIX", "CDELT", "CUNIT", "CTYPE"]
    axis_str = str(axis_number)
    for keyword in axis_keywords:
        try:
            del fitsheader[keyword + axis_str]
        except:
            pass


def save_maps(map_dict, prefix_path, FDFheader):
    """
    Saves the selected 2D maps of the fit output.
    Inputs:
        map_dict: a dictionary of 2D maps for the fitting outputs.
        prefix_path: the full or relative path to save the files, plus the file
            prefix to be given to the files.
    """
    # Set up generic FITS header
    product_header = FDFheader.copy()
    # Remove extra axes:
    product_header = remove_header_third_fourth_axis(product_header)

    product_header["HISTORY"] = (
        "Polarization peak maps created with RM-Tools RMpeakfit_3D"
    )

    # Set flux unit from FITS header if possible
    if "BUNIT" in FDFheader:
        flux_unit = FDFheader["BUNIT"]
    else:
        flux_unit = ""

    # Dictionary of units for peak fitting output parameters (for FITS headers)
    unit_dict = {
        "dFDFcorMAD": flux_unit,
        "phiPeakPIfit_rm2": "rad/m^2",
        "dPhiPeakPIfit_rm2": "rad/m^2",
        "ampPeakPIfit": flux_unit,
        "ampPeakPIfitEff": flux_unit,
        "dAmpPeakPIfit": flux_unit,
        "snrPIfit": "",
        "indxPeakPIfit": "",
        "peakFDFimagFit": flux_unit,
        "peakFDFrealFit": flux_unit,
        "polAngleFit_deg": "deg",
        "dPolAngleFit_deg": "deg",
        "polAngle0Fit_deg": "deg",
        "dPolAngle0Fit_deg": "deg",
        "Ifreq0": flux_unit,
        "polyCoeffs": "",
        "IfitStat": "",
        "IfitChiSqRed": "",
        "lam0Sq_m2": "m^2",
        "freq0_Hz": "Hz",
        "fwhmRMSF": "rad/m^2",
        "dQU": flux_unit,
        "dFDFth": flux_unit,
        "min_freq": "Hz",
        "max_freq": "Hz",
        "N_channels": "",
        "median_channel_width": "Hz",
        "fracPol": "",
        "sigmaAddQ": "",
        "dSigmaAddMinusQ": "",
        "dSigmaAddPlusQ": "",
        "sigmaAddU": "",
        "dSigmaAddMinusU": "",
        "dSigmaAddPlusU": "",
    }

    # Check that directory exists (in case it is requested to save maps to new subdirectory):
    if not os.path.exists(os.path.dirname(prefix_path)):
        os.mkdir(os.path.dirname(prefix_path))

    # per product, customize FITS header as needed and save file
    for product in map_dict.keys():
        if map_dict[product].dtype == np.float64:
            data = map_dict[product].astype(np.float32)
        else:
            data = map_dict[product]
        product_header["BUNIT"] = unit_dict[product]
        pf.writeto(
            prefix_path + product + ".fits", data, product_header, overwrite=True
        )


def find_axes(header):
    """Idenfities how many axes are present in a FITS file, and which is the
    Faraday depth axis. Necessary for bookkeeping on cube dimensionality,
    given that RM-clean only supports 3D cubes, but data may be 4D files."""
    Ndim = header["NAXIS"]
    FD_axis = Ndim
    # Check for FD axes:
    for i in range(1, Ndim + 1):
        try:
            if "FDEP" in header["CTYPE" + str(i)].upper():
                FD_axis = i
        except:
            pass  # The try statement is needed for if the FITS header does not
            # have CTYPE keywords.
    return Ndim, FD_axis


def read_files(FDF_filename, freq_filename):
    """
    Read in the files needed for peak fitting. Files assumed to be the standard
    outputs of RMsynth3D (and, optionally, RMclean3D). also, freq file.
    """

    HDUreal = pf.open(
        FDF_filename.replace("_tot", "_real").replace("_im", "_real"),
        "readonly",
        memmap=True,
    )
    head = HDUreal[0].header.copy()
    FDFreal = HDUreal[0].data

    HDUimag = pf.open(
        FDF_filename.replace("_tot", "_im").replace("_real", "_im"),
        "readonly",
        memmap=True,
    )
    FDFimag = HDUimag[0].data
    complex_cube = FDFreal + 1j * FDFimag

    # Identify Faraday depth axis (assumed to be last one if not explicitly found)
    Ndim, FD_axis = find_axes(head)

    # Move FD axis to first place in numpy order.
    if FD_axis != Ndim:
        complex_cube = np.moveaxis(complex_cube, Ndim - FD_axis, 0)

    # Remove degenerate axes to prevent problems with later steps.
    complex_cube = complex_cube.squeeze()

    lam0Sq = head["LAMSQ0"]

    freqArr_Hz = np.loadtxt(freq_filename, dtype=float)
    lambdaSqArr_m2 = np.power(speed_of_light.value / freqArr_Hz, 2.0)

    phiArr_radm2 = fits_make_lin_axis(head, axis=FD_axis - 1)

    HDUfwhm = pf.open(
        FDF_filename.replace("FDF_tot_dirty", "RMSF_FWHM")
        .replace("FDF_real_dirty", "RMSF_FWHM")
        .replace("FDF_im_dirty", "RMSF_FWHM")
        .replace("FDF_clean_tot", "RMSF_FWHM")
        .replace("FDF_clean_real", "RMSF_FWHM")
        .replace("FDF_clean_im", "RMSF_FWHM"),
        "readonly",
        memmap=True,
    )
    fwhmRMSF = HDUfwhm[0].data.squeeze()

    return complex_cube, phiArr_radm2, fwhmRMSF, lambdaSqArr_m2, lam0Sq, head


def main():
    """
    Start peak fitting from the command line.
    """

    import argparse

    descStr = """
    Perform pixel-wise fitting of the brightest peak in the FDF. This script
    performs the same fitting used in the RMsynth_1D for every pixel in a 3D
    FDF cube. The result is a set of 2D maps giving the fitting results (one
    map per fitting output). These are saved to a set of FITS files.
    """

    epilog_text = """
    The script saves 2D maps of the parameters from the peak fitting and
    FDF characterization function. The user can select how many products are
    saved:
    no flag: a curated list of potentially interesting parameters is saved
         -a: all outputs are saved.
         -p: only parameters that describe the peak are saved.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr,
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "FDF_filename",
        metavar="FDF_file.fits",
        nargs=1,
        help="An FDF cube from RMsynth3D or RMclean3D",
    )
    parser.add_argument(
        "freq_file",
        metavar="freq_file.txt",
        nargs=1,
        help="A text file containing the channel frequencies.",
    )
    parser.add_argument(
        "output_name",
        metavar="output_name",
        nargs=1,
        help="The path and base name of the output maps.",
    )
    parser.add_argument(
        "-a",
        dest="all_products",
        action="store_true",
        help="Save all fitting products.",
    )
    parser.add_argument(
        "-p", dest="peak_only", action="store_true", help="Save peak parameters only."
    )
    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="Verbose [False]."
    )
    parser.add_argument(
        "-i",
        dest="fitsI",
        default=None,
        help="FITS cube containing Stokes I model [None].",
    )
    parser.add_argument(
        "-n",
        dest="noiseFile",
        default=None,
        help="FITS file or cube containing noise values [None].",
    )
    parser.add_argument(
        "-w",
        dest="weightType",
        default="uniform",
        help="weighting ['uniform'] (all 1s) or 'variance' used in rmsynth3d, affects uncertainty estimation.",
    )

    args = parser.parse_args()

    # Check for incompatible options:
    if args.peak_only and args.all_products:
        print("Please select at most ONE of the -a and -p flags.")
        sys.exit()
    if not os.path.exists(args.FDF_filename[0]):
        print("Cannot find FDF file. Please check filename/path.")
        sys.exit()
    if not os.path.exists(args.freq_file[0]):
        print("Cannot find frequency file. Please check filename/path.")
        sys.exit()

    if args.output_name[0][0] != "/":
        args.output_name[0] = "./" + args.output_name[0]

    # Define default product lists:
    if args.peak_only:
        product_list = [
            "dFDFcorMAD",
            "phiPeakPIfit_rm2",
            "dPhiPeakPIfit_rm2",
            "ampPeakPIfitEff",
            "dAmpPeakPIfit",
            "snrPIfit",
            "polAngle0Fit_deg",
            "dPolAngle0Fit_deg",
        ]
    elif args.all_products:
        product_list = [
            "dFDFcorMAD",
            "phiPeakPIfit_rm2",
            "dPhiPeakPIfit_rm2",
            "ampPeakPIfit",
            "ampPeakPIfitEff",
            "dAmpPeakPIfit",
            "snrPIfit",
            "indxPeakPIfit",
            "peakFDFimagFit",
            "peakFDFrealFit",
            "polAngleFit_deg",
            "dPolAngleFit_deg",
            "polAngle0Fit_deg",
            "dPolAngle0Fit_deg",
            "Ifreq0",
            "dFDFth",
            "lam0Sq_m2",
            "freq0_Hz",
            "fwhmRMSF",
            "min_freq",
            "max_freq",
            "N_channels",
            "median_channel_width",
            "fracPol",
        ]
    else:  # Default option is a curated list of products I think are most useful.
        product_list = [
            "dFDFcorMAD",
            "phiPeakPIfit_rm2",
            "dPhiPeakPIfit_rm2",
            "ampPeakPIfitEff",
            "dAmpPeakPIfit",
            "snrPIfit",
            "peakFDFimagFit",
            "peakFDFrealFit",
            "Ifreq0",
            "lam0Sq_m2",
            "polAngle0Fit_deg",
            "dPolAngle0Fit_deg",
        ]

    # Read in files
    FDF, phiArr_radm2, fwhmRMSF, lambdaSqArr_m2, lam0Sq, header = read_files(
        args.FDF_filename[0], args.freq_file[0]
    )

    if args.fitsI is not None:
        dataI = readFitsCube(args.fitsI, args.verbose)[1]
    else:
        dataI = None
    if args.noiseFile is not None:
        rmsArr = readFreqFile(args.noiseFile, args.verbose)
    else:
        rmsArr = None

    if (
        FDF.ndim == 2
    ):  # When inputting chunks made from createchunks, it drops an axis and breaks things. Re-add to fix things.
        FDF = np.expand_dims(FDF, axis=1)
        fwhmRMSF = np.expand_dims(fwhmRMSF, axis=0)

    # Fit peaks
    map_dict = pixelwise_peak_fitting(
        FDF,
        phiArr_radm2,
        fwhmRMSF,
        lambdaSqArr_m2,
        lam0Sq,
        product_list,
        noiseArr=rmsArr,
        stokesIcube=dataI,
        weightType=args.weightType,
    )

    save_maps(map_dict, args.output_name[0], header)


if __name__ == "__main__":
    main()
