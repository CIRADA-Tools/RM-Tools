#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the Stokes I fitting in RMtools_3D"""

import logging
import os

import numpy as np
from astropy.io import fits

# rmtools_fitIcube
from RMtools_3D.do_fitIcube import make_model_I, open_datacube
from RMtools_3D.make_freq_file import get_freq_array

# import RMtools_3D
# print(f"Using version {RMtools_3D.__file__}")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_fake_StokesIcube(filename="random_Icube.fits"):
    # Create random data cube, 144 channels 102,102 pixels
    data = np.random.rand(144, 1, 102, 102).astype(np.float32)

    # Create FITS header
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 4
    header["NAXIS1"] = 102
    header["NAXIS2"] = 102
    header["NAXIS3"] = 1
    header["NAXIS4"] = 144
    header["WCSAXES"] = 4
    header["CRPIX1"] = -139869.0021857
    header["CRPIX2"] = -94562.00147332
    header["CRPIX3"] = 1.0
    header["CRPIX4"] = 1.0
    header["PC1_1"] = 0.7071067811865
    header["PC1_2"] = 0.7071067811865
    header["PC2_1"] = -0.7071067811865
    header["PC2_2"] = 0.7071067811865
    header["CDELT1"] = -0.0009710633743375
    header["CDELT2"] = 0.0009710633743375
    header["CDELT3"] = 1.0
    header["CDELT4"] = 1000000.0
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT4"] = "Hz"
    header["CTYPE1"] = "RA---HPX"
    header["CTYPE2"] = "DEC--HPX"
    header["CTYPE3"] = "STOKES"
    header["CTYPE4"] = "FREQ"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRVAL3"] = 1.0
    header["CRVAL4"] = 1295990740.741
    header["PV2_1"] = 4.0
    header["PV2_2"] = 3.0
    header["LONPOLE"] = 0.0
    header["LATPOLE"] = 90.0
    header["RESTFRQ"] = 1420405751.786
    header["RADESYS"] = "FK5"
    header["EQUINOX"] = 2000.0
    header["SPECSYS"] = "TOPOCENT"
    header["BMAJ"] = 0.005555555555556
    header["BMIN"] = 0.005555555555556
    header["BPA"] = 0.0
    header["BUNIT"] = "JY/BEAM"
    header["HISTORY"] = "RANDOM FITS FILE FOR TESTING"

    # Create PrimaryHDU object
    hdu = fits.PrimaryHDU(data=data, header=header)
    # Write the FITS file
    hdu.writeto(filename, overwrite=True)
    print(f"Random FITS cube created: {filename}")

    return filename


def cleanup(outDir, prefixOut, polyOrd):
    """
    Cleanup files that are made by make_model_I
    """
    os.system("rm random_Icube.fits")

    for i in range(np.abs(polyOrd) + 1):
        outname = os.path.join(outDir, prefixOut + "coeff" + str(i) + ".fits")
        os.system(f"rm {outname}")

        outname = os.path.join(outDir, prefixOut + "coeff" + str(i) + "err.fits")
        os.system(f"rm {outname}")

    MaskfitsFile = os.path.join(outDir, prefixOut + "mask.fits")
    os.system(f"rm {MaskfitsFile}")

    fitsModelFile = os.path.join(outDir, prefixOut + "model.i.fits")
    os.system(f"rm {fitsModelFile}")

    noisefile = os.path.join(outDir, prefixOut + "noise.dat")
    os.system(f"rm {noisefile}")

    outname = os.path.join(outDir, prefixOut + "reffreq.fits")
    os.system(f"rm {outname}")

    outname = os.path.join(outDir, prefixOut + "covariance.fits")
    os.system(f"rm {outname}")


def test_stokesIfit_with_without_verbose():
    """
    Testing RMtools_3D/do_fitIcube.py with and without verbose
    """
    I_filename = make_fake_StokesIcube()

    datacube, headI = open_datacube(fitsI=I_filename, verbose=False)
    # Deriving frequencies from the fits header.")
    freqArr_Hz = get_freq_array(I_filename)

    prefixOut = ""
    outDir = "./"
    polyOrd = 2

    logger.info("Running make_RMtools_3D.do_fitIcube.make_model_I with verbose=True")
    # Run polynomial fitting on the spectra with verbose=T
    make_model_I(
        datacube=datacube,
        header=headI,
        freqArr_Hz=freqArr_Hz,
        polyOrd=polyOrd,
        prefixOut=prefixOut,
        outDir=outDir,
        nBits=32,
        threshold=-3,
        apply_mask=False,
        num_cores=2,
        chunk_size=1,
        verbose=True,
        fit_function="log",
    )
    logger.info("Finished succesfully")

    logger.info("Running make_RMtools_3D.do_fitIcube.make_model_I with verbose=False")
    # Run polynomial fitting on the spectra with verbose=F
    make_model_I(
        datacube=datacube,
        header=headI,
        freqArr_Hz=freqArr_Hz,
        polyOrd=polyOrd,
        prefixOut=prefixOut,
        outDir=outDir,
        nBits=32,
        threshold=-3,
        apply_mask=False,
        num_cores=2,
        chunk_size=1,
        verbose=False,
        fit_function="log",
    )
    logger.info("Fitting finished succesfully")
    logger.info("Removing output files...")
    cleanup(outDir, prefixOut, polyOrd)

    logger.info("Stokes I fitting test finished succesfully")


if __name__ == "__main__":
    test_stokesIfit_with_without_verbose()
