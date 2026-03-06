from astropy.io import fits
import numpy as np
import argparse


def save_freq_file():
    # Parse the command line options
    parser = argparse.ArgumentParser(
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fits_file", metavar="Stokes.fits", nargs=1,
                        help="FITS cube containing Stokes data.")
    parser.add_argument("-n", dest="namefile", default="drao_Q_small_sub1_freqHz_1.dat",
                        help="Name of the freq file")
    parser.add_argument("-g", dest="fits_header", action="store_true",
                        help="gets fits header.")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="get freq array.")
    parser.add_argument("-s", dest="save_array", action="store_true",
                        help="save array.")

    args = parser.parse_args()


    if args.fits_header:
        header = get_fits_header(args.fits_file[0])
        print(header)

    if args.save_array:
        freq_array = get_freq_array(args.fits_file[0])

        np.savetxt(args.namefile, freq_array, delimiter="")
        if args.verbose: print(">Saving the freq files")

def get_fits_header(filename):
    hduList = fits.open(filename)
    return hduList[0].header

def get_freq_array(filename):
    hduList = fits.open(filename)
    header = hduList[0].header
    for key, value in header.items():
        if type(value) is str:
            if value.lower() == "freq":
                n_axis = int(key[-1])

    iter_pixel= int(header["NAXIS{}".format(n_axis)]- header["CRPIX{}".format(n_axis)])
    freq_data = [header["CRVAL{}".format(n_axis)]]

    for i in range(0, iter_pixel):
        freq_data.append(freq_data[i] + header["CDELT{}".format(n_axis)])


    freq_data_array = np.asarray(freq_data)
    return freq_data_array



if __name__ == "__main__":
    save_freq_file()
