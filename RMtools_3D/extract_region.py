#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:44:28 2019

Extract subregion of a FITS file, with option to extract a plane.

There are many cutout tools like it, but this one is mine.


@author: cvaneck
May 2019
"""

import argparse
import os

import astropy.io.fits as pf
from astropy.wcs import WCS


def main():
    """This function will extract a region ('cutout) from an FITS file and save
    it to a new file. Command line options will allow the user to select the
    region in either pixel or sky coordinates.
    """

    descStr = """
    Cut out a region in a fits file, writing it to a new file.
    Selecting -1 for any coordinate parameter will cause it be set as the
    maximum/minimum value allowed.
    Default is for box to be defined in pixel coordinates, in the form
    xmin xmax ymin ymax.
    Pixel selection is inclusive: all corner pixels will be present in output.
    Pixel counting starts at 1 (FITS convention).
    Sky coordinates not guaranteed to give correct size box in if projection is
    highly nonlinear.
    If a third non-generate axis is present (as either axis 3 or 4), the
    -z flag will allow selection of subranges along this axis."""

    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "infile", metavar="infile.fits", help="FITS file containing data."
    )
    parser.add_argument("outfile", metavar="outfile.fits", help="Output fits file")
    parser.add_argument(
        "box",
        metavar="xmin xmax ymin ymax",
        nargs=4,
        type=float,
        help="Box dimensions (in pixels unless -s set)",
    )
    parser.add_argument(
        "-s",
        dest="sky",
        action="store_true",
        help="Box defined in sky coordinates (in decimal degrees if set, otherwise pixels).",
    )
    parser.add_argument(
        "-c",
        dest="center",
        action="store_true",
        help="If true, define box as x_center x_halfwidth y_center y_halfwidth",
    )
    parser.add_argument(
        "-z",
        dest="zlim",
        metavar="axis3",
        nargs=2,
        default=None,
        type=int,
        help="3rd axis limits (only pixel coords supported)",
    )
    parser.add_argument(
        "-o",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing file if present?",
    )

    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise Exception("Input file not found!")

    if os.path.exists(args.outfile) and not args.overwrite:
        raise Exception("Outfile file already exists! Add -o flag to overwrite.")

    if not args.center:
        box = args.box
    else:
        box = [
            args.box[0] - args.box[1],
            args.box[0] + args.box[1],
            args.box[2] - args.box[3],
            args.box[2] + args.box[3],
        ]

    if box[0] > box[1]:
        raise Exception("Box dimensions incorrect! x_max < x_min!")

    if box[2] > box[3]:
        raise Exception("Box dimensions incorrect! y_max < y_min!")

    hdu = pf.open(args.infile, memmap=True)
    header = hdu[0].header

    if args.sky:
        raise Exception("Not yet implemented. Soon!")
        csys = WCS(header, naxis=2)
        pixbox = [-1, -1, -1, -1]
        pix = csys.all_world2pix(box[0], (box[2] + box[3]) / 2, 1)
        pixbox[0] = float(pix[0])
        pix = csys.all_world2pix(box[1], (box[2] + box[3]) / 2, 1)
        pixbox[1] = float(pix[0])
        if pixbox[1] < pixbox[0]:
            a = pixbox[0]
            pixbox[0] = pixbox[1]
            pixbox[1] = a
        pix = csys.all_world2pix((box[0] + box[1]) / 2, box[2], 1)
        pixbox[2] = float(pix[1])
        pix = csys.all_world2pix((box[0] + box[1]) / 2, box[3], 1)
        pixbox[3] = float(pix[1])
        box = [round(x) for x in pixbox]
    else:
        box = [int(x) for x in box]

    if header["NAXIS"] == 3:
        cube_axis = 3
    if header["NAXIS"] == 4:
        if header["NAXIS3"] != 1:
            cube_axis = 3
        else:
            cube_axis = 4
    if header["NAXIS"] == 2:
        cube_axis = 2

    if args.zlim != None:
        zlim = args.zlim
    elif cube_axis > 2:
        zlim = [1, header["NAXIS" + str(cube_axis)]]

    if box[0] < 1:
        box[0] = 1
    if (box[1] == -1) or (box[1] > header["NAXIS1"]):
        box[1] = header["NAXIS1"]
    if box[2] < 1:
        box[2] = 1
    if (box[3] == -1) or (box[3] > header["NAXIS2"]):
        box[3] = header["NAXIS2"]

    # Extract sub-region:
    if header["NAXIS"] == 4:
        if cube_axis == 3:
            data = hdu[0].data[
                :, zlim[0] - 1 : zlim[1], box[2] - 1 : box[3], box[0] - 1 : box[1]
            ]
        if cube_axis == 4:
            data = hdu[0].data[
                zlim[0] - 1 : zlim[1], :, box[2] - 1 : box[3], box[0] - 1 : box[1]
            ]
    elif header["NAXIS"] == 3:
        data = hdu[0].data[
            zlim[0] - 1 : zlim[1], box[2] - 1 : box[3], box[0] - 1 : box[1]
        ]
    elif header["NAXIS"] == 2:
        data = hdu[0].data[box[2] - 1 : box[3] - 1, box[0] - 1 : box[1] - 1]
    else:
        raise Exception("Number of dimensions is some nonsupported value!")

    # Change header information:
    new_header = header.copy()
    new_header["NAXIS1"] = box[1] - box[0] + 1
    new_header["NAXIS2"] = box[3] - box[2] + 1
    if args.zlim != None:
        new_header["NAXIS" + str(cube_axis)] = zlim[1] - zlim[0] + 1
        new_header["CRPIX" + str(cube_axis)] = (
            header["CRPIX" + str(cube_axis)] - zlim[0] + 1
        )
    new_header["CRPIX1"] = header["CRPIX1"] - box[0] + 1
    new_header["CRPIX2"] = header["CRPIX2"] - box[2] + 1

    pf.writeto(args.outfile, data, new_header, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
