#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for unit tests.

Created on Fri Sep  6 15:03:12 2024
@author: cvaneck
"""

import shutil

import numpy as np
from astropy.constants import c as speed_of_light
from astropy.io import fits as pf
from scipy.ndimage import gaussian_filter


def Faraday_thin_complex_polarization(freq_array, RM, Polint, initial_angle):
    """freq_array = channel frequencies in Hz
    RM = source RM in rad m^-2
    Polint = polarized intensity in whatever units
    initial angle = pre-rotation polarization angle (in degrees)"""
    l2_array = (speed_of_light.value / freq_array) ** 2
    Q = Polint * np.cos(2 * (np.outer(l2_array, RM) + np.deg2rad(initial_angle)))
    U = Polint * np.sin(2 * (np.outer(l2_array, RM) + np.deg2rad(initial_angle)))
    return np.squeeze(np.transpose(Q + 1j * U))


def create_1D_data(freq_arr, TEST_PATH, ONED_PATH):
    RM = 200
    pol_angle_deg = 50
    StokesI_midband = 1
    fracpol = 0.7
    noise_amplitude = 0.1
    spectral_index = -0.7
    error_estimate = 1  # Size of assumed errors as multiple of actual error.
    if not ONED_PATH.exists():
        ONED_PATH.mkdir(parents=True)
    shutil.copy(TEST_PATH / "RMsynth1D_testdata.dat", ONED_PATH / "simsource.dat")
    with open(ONED_PATH / "sim_truth.txt", "w") as f:
        f.write("RM = {} rad/m^2\n".format(RM))
        f.write("Intrsinsic polarization angle = {} deg\n".format(pol_angle_deg))
        f.write("Fractional polarization = {} %\n".format(fracpol * 100.0))
        f.write("Stokes I = {} Jy/beam\n".format(StokesI_midband))
        f.write(
            "Reference frequency for I = {} GHz\n".format(np.median(freq_arr) / 1e9)
        )
        f.write("Spectral index = {}\n".format(spectral_index))
        f.write("Actual error per channel = {} Jy/beam\n".format(noise_amplitude))
        f.write(
            "Input assumed error = {} Jy/beam\n".format(
                noise_amplitude * error_estimate
            )
        )


def create_3D_data(freq_arr, THREED_PATH, N_side=100):
    src_RM = 200
    src_pol_angle_deg = 50
    src_flux = 2
    src_x = N_side // 4
    src_y = N_side // 4

    diffuse_RM = 50
    diffuse_pol_angle_deg = -10
    diffuse_flux = 1

    noise_amplitude = 0.1
    beam_size_pix = 20

    src_pol_spectrum = Faraday_thin_complex_polarization(
        freq_arr, src_RM, src_flux, src_pol_angle_deg
    )
    diffuse_pol_spectrum = Faraday_thin_complex_polarization(
        freq_arr, diffuse_RM, diffuse_flux, diffuse_pol_angle_deg
    )

    src_Q_cube = np.zeros((N_side, N_side, freq_arr.size))
    src_U_cube = np.zeros((N_side, N_side, freq_arr.size))
    src_Q_cube[src_x, src_y, :] = src_pol_spectrum.real
    src_U_cube[src_x, src_y, :] = src_pol_spectrum.imag

    src_Q_cube = gaussian_filter(
        src_Q_cube, (beam_size_pix / 2.35, beam_size_pix / 2.35, 0), mode="wrap"
    )
    src_U_cube = gaussian_filter(
        src_U_cube, (beam_size_pix / 2.35, beam_size_pix / 2.35, 0), mode="wrap"
    )
    scale_factor = (
        np.max(np.sqrt(src_Q_cube**2 + src_U_cube**2)) / src_flux
    )  # Renormalizing flux after convolution
    src_Q_cube = src_Q_cube / scale_factor
    src_U_cube = src_U_cube / scale_factor

    diffuse_Q_cube = np.tile(
        diffuse_pol_spectrum.real[np.newaxis, np.newaxis, :], (N_side, N_side, 1)
    )
    diffuse_U_cube = np.tile(
        diffuse_pol_spectrum.imag[np.newaxis, np.newaxis, :], (N_side, N_side, 1)
    )

    rng = np.random.default_rng(20200422)
    noise_Q_cube = rng.normal(scale=noise_amplitude, size=src_Q_cube.shape)
    noise_U_cube = rng.normal(scale=noise_amplitude, size=src_Q_cube.shape)
    noise_Q_cube = gaussian_filter(
        noise_Q_cube, (beam_size_pix / 2.35, beam_size_pix / 2.35, 0), mode="wrap"
    )
    noise_U_cube = gaussian_filter(
        noise_U_cube, (beam_size_pix / 2.35, beam_size_pix / 2.35, 0), mode="wrap"
    )
    scale_factor = (
        np.std(noise_Q_cube) / noise_amplitude
    )  # Renormalizing flux after convolution
    noise_Q_cube = noise_Q_cube / scale_factor
    noise_U_cube = noise_U_cube / scale_factor

    Q_cube = src_Q_cube + noise_Q_cube + diffuse_Q_cube
    U_cube = src_U_cube + noise_U_cube + diffuse_U_cube

    header = pf.Header()
    header["BITPIX"] = -32
    header["NAXIS"] = 3
    header["NAXIS1"] = N_side
    header["NAXIS2"] = N_side
    header["NAXIS3"] = freq_arr.size
    header["CTYPE1"] = "RA---SIN"
    header["CRVAL1"] = 90
    header["CDELT1"] = -1.0 / 3600.0
    header["CRPIX1"] = 1
    header["CUNIT1"] = "deg"

    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL2"] = 0
    header["CDELT2"] = 1.0 / 3600.0
    header["CRPIX2"] = 1
    header["CUNIT2"] = "deg"

    header["CTYPE3"] = "FREQ"
    header["CRVAL3"] = freq_arr[0]
    header["CDELT3"] = freq_arr[1] - freq_arr[0]
    header["CRPIX3"] = 1
    header["CUNIT3"] = "Hz"

    header["BUNIT"] = "Jy/beam"

    if not THREED_PATH.exists():
        THREED_PATH.mkdir(parents=True)

    pf.writeto(
        THREED_PATH / "Q_cube.fits", np.transpose(Q_cube), header=header, overwrite=True
    )
    pf.writeto(
        THREED_PATH / "U_cube.fits", np.transpose(U_cube), header=header, overwrite=True
    )
    with open(THREED_PATH / "freqHz.txt", "w") as f:
        for freq in freq_arr:
            f.write("{:}\n".format(freq))

    with open(THREED_PATH / "sim_truth.txt", "w") as f:
        f.write("Point source:\n")
        f.write("RM = {} rad/m^2\n".format(src_RM))
        f.write("Intrsinsic polarization angle = {} deg\n".format(src_pol_angle_deg))
        f.write("Polarized Flux = {} Jy/beam\n".format(src_flux))
        f.write("x position = {} pix\n".format(src_x))
        f.write("y position = {} pix\n".format(src_y))
        f.write("\n")
        f.write("Diffuse emission:\n")
        f.write("RM = {} rad/m^2\n".format(diffuse_RM))
        f.write(
            "Intrsinsic polarization angle = {} deg\n".format(diffuse_pol_angle_deg)
        )
        f.write("Polarized Flux = {} Jy/beam\n".format(diffuse_flux))
        f.write("\n")
        f.write("Other:\n")
        f.write("Actual error per channel = {} Jy/beam\n".format(noise_amplitude))
        f.write("Beam FWHM = {} pix\n".format(beam_size_pix))
