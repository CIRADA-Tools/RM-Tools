#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:10:36 2024

Testing tools for confirming the basic functions of the various helper tools.
In the past, changes to RM-Tools has broken some of these, because only the
core functionality has had tests to confirm function after code changes.

The initial version of these tests only really tests that they run, but not
that the outputs are correct or that the functions within have performed as
expected.

List of helper functions:
    rmtools_testdata1D
    rmtools_testdata3D
    rmtools_calcRMSF
    rmtools_freqfile
    rmtools_extractregion
    rmtools_createchunks
    rmtools_assemblechunks
    rmtools_fitIcube
    rmtools_3DIrescale
    rmtools_peakfitcube
    rmtools_bwpredict
    rmtools_bwdepol


@author: cvaneck
"""

import os
import shlex
import shutil
import subprocess
import unittest
from pathlib import Path

import astropy.io.fits as pf
import numpy as np

from RMutils.util_testing import create_1D_data, create_3D_data

TEST_PATH = Path(__file__).parent.absolute()
ONED_PATH = TEST_PATH / "simdata" / "1D"
THREED_PATH = TEST_PATH / "simdata" / "3D"


def create_3D_stokesI(freq_arr, N_side=100):
    from scipy.ndimage import gaussian_filter

    from RMutils.util_misc import powerlaw_poly5

    if not (TEST_PATH / "simdata/3D").exists():
        (TEST_PATH / "simdata/3D").mkdir(parents=True)

    I_cube = np.zeros((N_side, 2 * N_side, freq_arr.size))
    model_parms = [0, 0, 0, 0, -1, 10000]
    source_I = powerlaw_poly5(model_parms)(freq_arr / 940e6)
    I_cube[N_side // 4, N_side // 2, :] += source_I

    beam_sigma = 5
    new_I_cube = gaussian_filter(I_cube, (beam_sigma, beam_sigma, 0), mode="wrap")

    # Output for 3D RM-tools:
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

    header["BUNIT"] = "mJy/beam"
    pf.writeto(
        TEST_PATH / "simdata/3D/I_cube.fits",
        np.transpose(new_I_cube),
        header=header,
        overwrite=True,
    )


class test_RMtools(unittest.TestCase):
    def setUp(self):

        # Clean up directory if it exists (to keep test clean).
        if (TEST_PATH / "simdata").exists():
            shutil.rmtree(TEST_PATH / "simdata")

        (TEST_PATH / "simdata").mkdir(parents=True)
        (TEST_PATH / "simdata/1D").mkdir(parents=True)
        (TEST_PATH / "simdata/3D").mkdir(parents=True)

        N_chan = 288
        self.freq_arr = np.linspace(800e6, 1088e6, num=N_chan)

    def test_testdata3D(self):
        """Test that rmtools_testdata3D runs and outputs."""
        res = subprocess.run(
            [
                "rmtools_testdata3D",
                f"{TEST_PATH}/../RMtools_3D/catalogue.csv",
                "./simdata/3D",
            ]
        )
        self.assertEqual(res.returncode, 0, "testdata3D fails to run to completion.")

        self.assertTrue(
            (TEST_PATH / "simdata/3D/StokesU.fits").exists(),
            "testdata3D not outputting files as expected.",
        )

    def test_testdata1D(self):
        """Test that rmtools_testdata1D runs and outputs."""
        res = subprocess.run(
            [
                "rmtools_testdata1D",
                f"{TEST_PATH}/../RMtools_3D/catalogue.csv",
                "./simdata/1D",
            ]
        )
        self.assertEqual(res.returncode, 0, "testdata1D fails to run to completion.")

        self.assertTrue(
            (TEST_PATH / "simdata/1D/Source8.dat").exists(),
            "testdata1D not outputting files as expected.",
        )

    def test_calcRMSF(self):
        """Test that rmtools_calcRMSF runs as expected."""
        res = subprocess.run(
            shlex.split("rmtools_calcRMSF -f 800e6 1088e6 1e6 -s ./simdata/rmsf.png"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, "calcRMSF fails to run to completion.")

        output = res.stdout.splitlines()
        # Testing some output values; hardcoding expected values.
        self.assertEqual(
            output[1].split()[3],
            "59.04",
            "calcRMSF values have changed from expectations",
        )
        self.assertEqual(
            output[4].split()[3],
            "8093",
            "calcRMSF values have changed from expectations",
        )

    def test_freqfile(self):
        """Test that rmtools_freqfile runs as expected."""
        if not (TEST_PATH / "simdata/3D/Q_cube.fits").exists():
            create_3D_data(self.freq_arr, THREED_PATH)

        res = subprocess.run(
            shlex.split(
                f"rmtools_freqfile {TEST_PATH}/simdata/3D/Q_cube.fits {TEST_PATH}/simdata/freqfile.dat"
            )
        )
        self.assertEqual(res.returncode, 0, "freqfile fails to run to completion.")

        expected_array = np.loadtxt(f"{TEST_PATH}/simdata/3D/freqHz.txt")
        output_array = np.loadtxt(f"{TEST_PATH}/simdata/freqfile.dat")

        self.assertIsNone(
            np.testing.assert_array_almost_equal_nulp(
                expected_array.astype("float32"), output_array.astype("float32"), nulp=1
            )
        )

    def test_extractregion(self):
        """Test that rmtools_extractregion runs as expected."""
        if not (TEST_PATH / "simdata/3D/Q_cube.fits").exists():
            create_3D_data(self.freq_arr, THREED_PATH)

        res = subprocess.run(
            shlex.split(
                f"rmtools_extractregion {TEST_PATH}/simdata/3D/Q_cube.fits {TEST_PATH}/simdata/3D/Q_cutout.fits 30 50 40 60 -z 12 24"
            )
        )
        self.assertEqual(res.returncode, 0, "extractregion fails to run to completion.")

    def test_createchunks(self):
        """Test that rmtools_createchunks runs as expected."""
        if not (TEST_PATH / "simdata/3D/Q_cube.fits").exists():
            create_3D_data(self.freq_arr, THREED_PATH)

        res = subprocess.run(
            shlex.split(f"rmtools_createchunks {TEST_PATH}/simdata/3D/Q_cube.fits 2099")
        )
        self.assertEqual(res.returncode, 0, "createchunks fails to run to completion.")

        self.assertTrue(
            (TEST_PATH / "simdata/3D/Q_cube.C0.fits").exists(),
            "createchunks does not create first chunk.",
        )
        self.assertTrue(
            (TEST_PATH / "simdata/3D/Q_cube.C4.fits").exists(),
            "createchunks does not create final chunk.",
        )

    def test_assemblechunks(self):
        """Test that rmtools_assemblechunks runs as expected."""
        if not (TEST_PATH / "simdata/3D/Q_cube.fits").exists():
            create_3D_data(self.freq_arr, THREED_PATH)

        if not (TEST_PATH / "simdata/3D/Q_cube.C4.fits").exists():
            res = subprocess.run(
                shlex.split(
                    f"rmtools_createchunks {TEST_PATH}/simdata/3D/Q_cube.fits 2099"
                )
            )

        res = subprocess.run(
            shlex.split(
                f"rmtools_assemblechunks {TEST_PATH}/simdata/3D/Q_cube.C0.fits -f {TEST_PATH}/simdata/3D/Q_assembled.fits"
            )
        )
        self.assertEqual(
            res.returncode, 0, "assemblechunks fails to run to completion."
        )

        assembled_data = pf.open(f"{TEST_PATH}/simdata/3D/Q_assembled.fits")
        original_data = pf.open(f"{TEST_PATH}/simdata/3D/Q_cube.fits")

        self.assertIsNone(
            np.testing.assert_array_almost_equal_nulp(
                assembled_data[0].data, original_data[0].data, nulp=1
            ),
            "Pixel values not roundtripped identically.",
        )

        self.assertEqual(
            assembled_data[0].header.tostring(),
            original_data[0].header.tostring(),
            "Headers don't round trip identically.",
        )

    def test_fitIcube(self):
        """Test that rmtools_fitIcube runs as expected."""
        if not (TEST_PATH / "simdata/3D/I_cube.fits").exists():
            create_3D_stokesI(self.freq_arr)

        res = subprocess.run(
            shlex.split(f"rmtools_fitIcube {TEST_PATH}/simdata/3D/I_cube.fits")
        )
        self.assertEqual(res.returncode, 0, "fitIcube fails to run to completion.")

        self.assertTrue(
            (TEST_PATH / "simdata/3D/coeff0.fits").exists(), "fitIcube outputs missing."
        )
        self.assertTrue(
            (TEST_PATH / "simdata/3D/coeff0err.fits").exists(),
            "fitIcube outputs missing.",
        )
        self.assertTrue(
            (TEST_PATH / "simdata/3D/noise.dat").exists(), "fitIcube outputs missing."
        )
        self.assertTrue(
            (TEST_PATH / "simdata/3D/covariance.fits").exists(),
            "fitIcube outputs missing.",
        )

    def test_3DIrescale(self):
        """Test that rmtools_fitIcube runs as expected."""
        if not (TEST_PATH / "simdata/3D/I_cube.fits").exists():
            create_3D_stokesI(self.freq_arr)

        if not (TEST_PATH / "simdata/3D/covariance.fits").exists():
            res = subprocess.run(
                shlex.split(f"rmtools_fitIcube {TEST_PATH}/simdata/3D/I_cube.fits")
            )

        res = subprocess.run(
            shlex.split(
                f"rmtools_3DIrescale {TEST_PATH}/simdata/3D/covariance.fits -f 950e6 -o {TEST_PATH}/simdata/3D/rescale"
            )
        )
        self.assertEqual(res.returncode, 0, "3DIrescale fails to run to completion.")

    def test_peakfitcube(self):
        """Test that rmtools_peakfitcube runs as expected."""
        if not (TEST_PATH / "simdata/3D/Q_cube.fits").exists():
            create_3D_data(self.freq_arr, THREED_PATH)
        res = subprocess.run(
            shlex.split(
                f"rmsynth3d {TEST_PATH}/simdata/3D/Q_cube.fits {TEST_PATH}/simdata/3D/Q_cube.fits {TEST_PATH}/simdata/3D/freqHz.txt -s 2"
            )
        )

        res = subprocess.run(
            shlex.split(
                f"rmtools_peakfitcube {TEST_PATH}/simdata/3D/FDF_real_dirty.fits {TEST_PATH}/simdata/3D/freqHz.txt {TEST_PATH}/simdata/3D/peakfit -p"
            )
        )
        self.assertEqual(res.returncode, 0, "peakfitcube fails to run to completion.")

        self.assertTrue(
            (TEST_PATH / "simdata/3D/peakfitphiPeakPIfit_rm2.fits").exists(),
            "peakfitcube outputs missing.",
        )

    def test_bwpredict(self):
        """Test that rmtools_bwpredict runs as expected."""
        if not (TEST_PATH / "simdata/3D/Q_cube.fits").exists():
            create_3D_data(self.freq_arr, THREED_PATH)

        res = subprocess.run(
            shlex.split(
                f"rmtools_bwpredict {TEST_PATH}/simdata/3D/freqHz.txt -f {TEST_PATH}/simdata/test.png"
            )
        )
        self.assertEqual(res.returncode, 0, "bwpredict fails to run to completion.")

    def test_bwdepol(self):
        """Test that rmtools_bwdepol runs as expected."""
        if not (TEST_PATH / "simdata/1D/simsource.dat").exists():
            create_1D_data(self.freq_arr, TEST_PATH, ONED_PATH)

        res = subprocess.run(
            shlex.split(f"rmtools_bwdepol {TEST_PATH}/simdata/1D/simsource.dat -s 3 -S")
        )
        self.assertEqual(res.returncode, 0, "bwdepol fails to run to completion.")
        self.assertTrue(
            (TEST_PATH / "simdata/1D/simsource_RMsynth.dat").exists(),
            "bwdepol outputs missing.",
        )


if __name__ == "__main__":
    #    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    print("\nUnit tests running.")
    print("Test data inputs and outputs can be found in {}\n\n".format(os.getcwd()))

    unittest.TestLoader.sortTestMethodsUsing = None
    suite = unittest.TestLoader().loadTestsFromTestCase(test_RMtools)
    unittest.TextTestRunner(verbosity=2).run(suite)
