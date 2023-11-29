# This script creates a frequency file from a FITS header. This is a helper
# script to make it easier to run RMsynth 1D or 3D. Run this first to generate
# the required frequency file. If you create a spectrum or cube from multiple
# FITS files, run it on the individual input files.
# This script assumes the FITS header has a FREQ axis, and that this axis
# accurately describes the frequency channels.

# version 1 by Boris Gbeasor, summer 2019
# ver2: modified by Cameron Van Eck


import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def save_freq_file():
    """
    Parses command line arguments, extracts FITS header, and saves it to a file.
    """
    # Parse the command line options
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "fits_file",
        metavar="Infile.fits",
        nargs=1,
        help="FITS cube with frequency axis.",
    )
    parser.add_argument(
        "freq_file", metavar="outfile.dat", help="Name of the freq file to write."
    )

    args = parser.parse_args()

    freq_array = get_freq_array(args.fits_file[0])

    np.savetxt(args.freq_file, freq_array, delimiter="")
    print("Saving the frequencies list to {}".format(args.freq_file))


def get_fits_header(filename):
    hduList = fits.open(filename)
    header = hduList[0].header
    hduList.close()
    return header


def get_freq_array(filename):
    header = fits.getheader(filename)
    wcs = WCS(header)
    spec_wcs = wcs.spectral
    nchan = spec_wcs.array_shape[0]
    freqs = spec_wcs.pixel_to_world_values(np.arange(nchan))

    return freqs


if __name__ == "__main__":
    save_freq_file()
