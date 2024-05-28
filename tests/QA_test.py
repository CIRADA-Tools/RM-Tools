#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA testing tools for RM-tools.
These tools are intended to produce test simulated data sets, and run them
through RM-tools. Automated tools will only be able to confirm that things ran,
but user inspection of the results will be needed to confirm that the expected
values are produced.

Random values are necessary to simulate noise, which is expected for different
parts of the code. I have forced the random seed to be the same each run,
in order to make the tests deterministic. This works for everything except
QU fitting, which uses random numbers internally that can't be controlled.

Created on Fri Oct 25 10:00:24 2019
@author: Cameron Van Eck
"""
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np
from astropy.constants import c as speed_of_light
from astropy.io import fits as pf
from scipy.ndimage import gaussian_filter

TEST_PATH = Path(__file__).parent.absolute()
ONED_PATH = TEST_PATH / "simdata" / "1D"
THREED_PATH = TEST_PATH / "simdata" / "3D"


def Faraday_thin_complex_polarization(freq_array, RM, Polint, initial_angle):
    """freq_array = channel frequencies in Hz
    RM = source RM in rad m^-2
    Polint = polarized intensity in whatever units
    initial angle = pre-rotation polarization angle (in degrees)"""
    l2_array = (speed_of_light.value / freq_array) ** 2
    Q = Polint * np.cos(2 * (np.outer(l2_array, RM) + np.deg2rad(initial_angle)))
    U = Polint * np.sin(2 * (np.outer(l2_array, RM) + np.deg2rad(initial_angle)))
    return np.squeeze(np.transpose(Q + 1j * U))


def create_1D_data(freq_arr):
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


def create_3D_data(freq_arr, N_side=100):
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


class test_RMtools(unittest.TestCase):
    def setUp(self):
        # Clean up old simulations to prevent interference with new runs.
        N_chan = 288
        self.freq_arr = np.linspace(800e6, 1088e6, num=N_chan)
        self.models = (1, 2, 3, 4, 5, 7, 11)
        self.sampler = "nestle"

    def test_a1_1D_synth_runs(self):
        create_1D_data(self.freq_arr)
        returncode = subprocess.call(
            f"rmsynth1d '{(ONED_PATH/'simsource.dat').as_posix()}' -l 600 -d 3 -S -i",
            shell=True,
        )
        self.assertEqual(returncode, 0, "RMsynth1D failed to run.")

    def test_a2_1D_synth_values(self):
        mDict = json.load(open(ONED_PATH / "simsource_RMsynth.json", "r"))
        refDict = json.load(open(TEST_PATH / "RMsynth1D_referencevalues.json", "r"))
        for key in mDict.keys():
            if (key == "polyCoefferr") or key == "polyCoeffs":
                ref_values = refDict[key].split(",")
                test_values = mDict[key].split(",")
                for ref, test in zip(ref_values, test_values):
                    self.assertAlmostEqual(
                        float(test),
                        float(ref),
                        places=3,
                        msg=f"Key {key} differs from expectation",
                    )
            elif type(mDict[key]) == str or refDict[key] == 0:
                self.assertEqual(
                    mDict[key], refDict[key], "{} differs from expectation.".format(key)
                )
            else:
                self.assertTrue(
                    np.abs((mDict[key] - refDict[key]) / refDict[key]) < 1e-3,
                    "{} differs from expectation.".format(key),
                )

    def test_c_3D_synth(self):
        create_3D_data(self.freq_arr)
        returncode = subprocess.call(
            f"rmsynth3d '{(THREED_PATH/'Q_cube.fits').as_posix()}' '{(THREED_PATH/'U_cube.fits').as_posix()}' '{(THREED_PATH/'freqHz.txt').as_posix()}' -l 300 -d 10",
            shell=True,
        )
        self.assertEqual(returncode, 0, "RMsynth3D failed to run.")
        header = pf.getheader(THREED_PATH / "FDF_tot_dirty.fits")
        self.assertEqual(header["NAXIS"], 3, "Wrong number of axes in output?")
        self.assertEqual(
            (header["NAXIS1"], header["NAXIS2"]),
            (100, 100),
            "Image plane has wrong dimensions!",
        )
        self.assertEqual(
            header["NAXIS3"], 61, "Number of output FD planes has changed."
        )

    def test_b1_1D_clean(self):
        if not (ONED_PATH / "simsource_RMsynth.dat").exists():
            self.skipTest("Could not test 1D clean; 1D synth failed first.")
        returncode = subprocess.call(
            f"rmclean1d '{(ONED_PATH/'simsource.dat').as_posix()}' -n 11 -S", shell=True
        )
        self.assertEqual(returncode, 0, "RMclean1D failed to run.")

    def test_b2_1D_clean_values(self):
        mDict = json.load(open(ONED_PATH / "simsource_RMclean.json", "r"))
        refDict = json.load(open(TEST_PATH / "RMclean1D_referencevalues.json", "r"))
        for key in mDict.keys():
            self.assertTrue(
                np.abs((mDict[key] - refDict[key]) / refDict[key]) < 1e-3,
                "{} differs from expectation.".format(key),
            )

    def test_d_3D_clean(self):
        if not (THREED_PATH / "FDF_tot_dirty.fits").exists():
            self.skipTest("Could not test 3D clean; 3D synth failed first.")
        returncode = subprocess.call(
            f"rmclean3d '{(THREED_PATH/'FDF_tot_dirty.fits').as_posix()}' '{(THREED_PATH/'RMSF_tot.fits').as_posix()}' -n 10",
            shell=True,
        )
        self.assertEqual(returncode, 0, "RMclean3D failed to run.")
        # what else?

    def test_e_1Dsynth_fromFITS(self):
        if not (THREED_PATH / "Q_cube.fits").exists():
            create_3D_data(self.freq_arr)
        returncode = subprocess.call(
            f"rmsynth1dFITS '{(THREED_PATH/'Q_cube.fits').as_posix()}' '{(THREED_PATH/'U_cube.fits').as_posix()}'  25 25 -l 600 -d 3 -S",
            shell=True,
        )
        self.assertEqual(returncode, 0, "RMsynth1D_fromFITS failed to run.")

    def test_f1_QUfitting(self):
        if not (ONED_PATH / "simsource.dat").exists():
            create_1D_data(self.freq_arr)

        local_models = Path("models_ns")
        if not local_models.exists():
            shutil.copytree(TEST_PATH / ".." / "RMtools_1D" / "models_ns", local_models)

        for model in self.models:
            returncode = subprocess.call(
                f"qufit simdata/1D/simsource.dat --sampler {self.sampler} -m {model}",
                shell=True,
            )

        self.assertEqual(returncode, 0, "QU fitting failed to run.")
        shutil.rmtree(local_models)

    def _test_f2_QUfit_values(self):
        # I have temporarily disabled this test because it causes a lot of problems
        # with values not being consistant across different runs.
        err_limit = 0.05  # 5%

        for model in self.models:
            mDict = json.load(
                open(ONED_PATH / f"simsource_m{model}_{self.sampler}.json", "r")
            )

            refDict = json.load(
                open(
                    TEST_PATH
                    / f"QUfit_referencevalues/simsource_m{model}_{self.sampler}.json",
                    "r",
                )
            )

            # The QU-fitting code has internal randomness that I can't control. So every run
            # will produce slightly different results. I want to assert that these differences
            # are below some limit.
            for key, val in refDict.items():
                if isinstance(val, str):
                    continue
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        if isinstance(v, str):
                            continue
                        self.assertTrue(
                            abs(mDict[key][i] - v) / abs(v) < err_limit,
                            f"values[{i}] of {key} of model {model} differs from expectation.",
                        )
                else:
                    self.assertTrue(
                        abs(mDict[key] - val) / abs(val) < err_limit,
                        f"{key} of model {model} differs from expectation.",
                    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if (TEST_PATH / "simdata").exists():
        shutil.rmtree("simdata")

    print("\nUnit tests running.")
    print("Test data inputs and outputs can be found in {}\n\n".format(os.getcwd()))

    unittest.TestLoader.sortTestMethodsUsing = None
    suite = unittest.TestLoader().loadTestsFromTestCase(test_RMtools)
    unittest.TextTestRunner(verbosity=2).run(suite)
