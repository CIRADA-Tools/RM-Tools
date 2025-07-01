#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:10:26 2019

This code reassembles chunks into larger files. This is useful for assembling
output files from 3D RM synthesis back into larger cubes.

@author: cvaneck
"""

import argparse
import re
from glob import glob
from math import ceil

import astropy.io.fits as pf
import numpy as np
from tqdm.auto import trange


def main():
    """This function will assemble a large FITS file or cube from smaller chunks."""

    descStr = """
    Assemble a FITS image/cube from small pieces. The new image will be created
    in the running directory.
    Supply one of the chunk files (other files will be identified by name pattern).
    Output name will follow the name of the input chunk, minus the '.C??.'
    """

    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "chunkname", metavar="chunk.fits", help="One of the chunks to be assembled"
    )
    parser.add_argument(
        "-f",
        dest="output",
        default=None,
        help="Specify output file name [basename of chunk]",
    )
    parser.add_argument(
        "-o",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing file? [False].",
    )

    args = parser.parse_args()

    if args.output == None:
        output_filename = ".".join(
            [
                x
                for x in args.chunkname.split(".")
                if not (x.startswith("C") and x[1:].isnumeric())
            ]
        )
    else:
        output_filename = args.output

    # Get all the chunk filenames. Missing chunks will break things!
    filename = re.search("\.C\d+\.", args.chunkname)
    chunkfiles = glob(
        args.chunkname[0 : filename.start()] + ".C*." + args.chunkname[filename.end() :]
    )
    chunkfiles.sort()

    old_header = pf.getheader(chunkfiles[0])
    x_dim = old_header["OLDXDIM"]
    y_dim = old_header["OLDYDIM"]
    Nperchunk = old_header["NAXIS1"]
    Npix_image = x_dim * y_dim
    num_chunks = ceil(Npix_image / Nperchunk)
    Ndim = old_header["NAXIS"]

    if (Ndim != 4) and (Ndim != 3) and (Ndim != 2):
        raise Exception(
            "Right now this code only supports FITS files with 2-4 dimensions!"
        )

    new_header = old_header.copy()
    del new_header["OLDXDIM"]
    del new_header["OLDYDIM"]
    new_header["NAXIS1"] = x_dim
    new_header["NAXIS2"] = y_dim

    # Create blank file:
    new_header.tofile(output_filename, overwrite=args.overwrite)

    # According to astropy, this is how to create a large file without needing it in memory:
    shape = tuple(
        new_header["NAXIS{0}".format(ii)] for ii in range(1, new_header["NAXIS"] + 1)
    )
    with open(output_filename, "rb+") as fobj:
        fobj.seek(
            len(new_header.tostring())
            + (np.prod(shape) * np.abs(new_header["BITPIX"] // 8))
            - 1
        )
        fobj.write(b"\0")

    if len(chunkfiles) != num_chunks:
        raise Exception("Number of chunk files found does not match expectations!")

    base_idx_arr = np.array(range(Nperchunk))

    large = pf.open(output_filename, mode="update", memmap=True)

    for i in trange(num_chunks - 1, desc="Assembling chunks"):
        file = chunkfiles[i]
        idx = base_idx_arr + i * Nperchunk
        xarr = idx // y_dim
        yarr = idx % y_dim

        chunk = pf.open(file, memmap=True)
        if Ndim == 4:
            large[0].data[:, :, yarr, xarr] = chunk[0].data[:, :, 0, :]
        elif Ndim == 3:
            large[0].data[:, yarr, xarr] = chunk[0].data[:, 0, :]
        elif Ndim == 2:
            large[0].data[yarr, xarr] = chunk[0].data

        #       large.flush()
        chunk.close()

    i += 1
    file = chunkfiles[i]
    idx = base_idx_arr + i * Nperchunk
    idx = idx[idx < Npix_image]
    xarr = idx // y_dim
    yarr = idx % y_dim
    chunk = pf.open(file, memmap=True)
    if Ndim == 4:
        large[0].data[:, :, yarr, xarr] = chunk[0].data[:, :, 0, :]
    elif Ndim == 3:
        large[0].data[:, yarr, xarr] = chunk[0].data[:, 0, :]
    elif Ndim == 2:
        large[0].data[yarr, xarr] = chunk[0].data
    large.flush()
    chunk.close()

    large.close()


if __name__ == "__main__":
    main()
