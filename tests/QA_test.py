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
in order to make the tests deterministic.

The (slower) QU-fitting tests have been moved to QUfitting_test.py.

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
from astropy.io import fits as pf

from RMutils.util_testing import create_1D_data, create_3D_data

TEST_PATH = Path(__file__).parent.absolute()
ONED_PATH = TEST_PATH / "simdata" / "1D"
THREED_PATH = TEST_PATH / "simdata" / "3D"


class test_RMtools(unittest.TestCase):
    def setUp(self):
        # Clean up old simulations to prevent interference with new runs.
        N_chan = 288
        self.freq_arr = np.linspace(800e6, 1088e6, num=N_chan)

    def test_a1_1D_synth_runs(self):
        create_1D_data(self.freq_arr, TEST_PATH, ONED_PATH)
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
        create_3D_data(self.freq_arr, THREED_PATH)
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
            create_3D_data(self.freq_arr, THREED_PATH)
        returncode = subprocess.call(
            f"rmsynth1dFITS '{(THREED_PATH/'Q_cube.fits').as_posix()}' '{(THREED_PATH/'U_cube.fits').as_posix()}'  25 25 -l 600 -d 3 -S",
            shell=True,
        )
        self.assertEqual(returncode, 0, "RMsynth1D_fromFITS failed to run.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if (TEST_PATH / "simdata").exists():
        shutil.rmtree("simdata")

    print("\nUnit tests running.")
    print("Test data inputs and outputs can be found in {}\n\n".format(os.getcwd()))

    unittest.TestLoader.sortTestMethodsUsing = None
    suite = unittest.TestLoader().loadTestsFromTestCase(test_RMtools)
    unittest.TextTestRunner(verbosity=2).run(suite)
