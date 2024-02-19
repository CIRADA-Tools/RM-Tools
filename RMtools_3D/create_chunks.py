#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:25:30 2019

This code will divide a FITS cube into individual chunks.
To minimize problems with how to divide the cube, it will convert the image
plane into a 1D list of spectra.
Then the file will divided into smaller files, with fewer pixels, in order to
be run through RM synthesis and CLEAN.
A separate routine will re-assemble the individual chunks into a combined file again.
This code attempts to minimize the memory profile: in principle it should never
need more memory than the size of a single chunk, and perhaps not even that much.

@author: cvaneck
May 2019
"""

import argparse
import os.path as path
from math import ceil, floor, log10

import astropy.io.fits as pf
import numpy as np
from tqdm.auto import trange


def main():
    """This function will divide a large FITS file or cube into smaller chunks.
    It does so in a memory efficient way that requires only a small RAM overhead
    (approximately 1 chunk worth?).
    """
    descStr = """
    Divide a FITS cube into small pieces for memory-efficient RM synthesis.
    Files will be created in running directory.
    WARNING: ONLY WORKS ON FIRST HDU, OTHERS WILL BE LOST."""

    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "infile", metavar="filename.fits", help="FITS cube containing data"
    )
    parser.add_argument(
        "Nperchunk", metavar="N_pixels", help="Number of pixels per chunk"
    )
    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="Verbose [False]."
    )
    parser.add_argument(
        "-p", dest="prefix", default=None, help="Prefix of output files [filename]"
    )

    args = parser.parse_args()

    Nperchunk = int(args.Nperchunk)

    if not path.exists(args.infile):
        raise Exception("Input file not found!")

    if args.prefix == None:
        prefix = path.splitext(args.infile)[0]
    else:
        prefix = args.prefix

    hdu = pf.open(args.infile, memmap=True)
    header = hdu[0].header
    data = np.transpose(hdu[0].data)

    x_image = header["NAXIS1"]
    y_image = header["NAXIS2"]
    Npix_image = x_image * y_image

    num_chunks = ceil(Npix_image / Nperchunk)
    digits = floor(log10(num_chunks)) + 1
    prntcode = ":0" + str(digits) + "d"

    if args.verbose:
        print(('Chunk name set to "{}.C{' + prntcode + '}.fits"').format(prefix, 0))
        print("File will be divided into {} chunks".format(num_chunks))

    base_idx_arr = np.array(range(Nperchunk))

    new_header = header.copy()
    new_header["NAXIS2"] = 1
    new_header["NAXIS1"] = Nperchunk
    new_header["OLDXDIM"] = x_image
    new_header["OLDYDIM"] = y_image

    # Run for all but last. Last chunk requires some finessing.
    for i in trange(num_chunks - 1, desc="Creating chunks"):
        idx = base_idx_arr + i * Nperchunk
        xarr = idx // y_image
        yarr = idx % y_image
        newdata = np.expand_dims(data[xarr, yarr], 1)
        filename = ("{}.C{" + prntcode + "}.fits").format(prefix, i)
        pf.writeto(filename, np.transpose(newdata), new_header, overwrite=True)

    i += 1
    idx = base_idx_arr + i * Nperchunk
    idx = idx[idx < Npix_image]
    xarr = idx // y_image
    yarr = idx % y_image
    newdata = np.expand_dims(data[xarr, yarr], 1)
    filename = ("{}.C{" + prntcode + "}.fits").format(prefix, i)
    pf.writeto(filename, np.transpose(newdata), new_header, overwrite=True)

    hdu.close()


if __name__ == "__main__":
    main()
