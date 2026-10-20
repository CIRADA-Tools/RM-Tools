#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QU-fitting QA tests for RM-tools.

These were split out of QA_test.py because QU-fitting is much slower than the
other tests, so it is useful to be able to run the RM-synthesis tests without it.

As in QA_test.py, the simulated data is created with a fixed random seed, but
QU fitting uses random numbers internally that can't be controlled, so the
fit results are not exactly reproducible between runs.

Originally part of QA_test.py, split off Oct 2026
@author: Cameron Van Eck
"""

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from RMutils.util_testing import create_1D_data

TEST_PATH = Path(__file__).parent.absolute()
ONED_PATH = TEST_PATH / "simdata" / "1D"


class test_QUfitting(unittest.TestCase):
    def setUp(self):
        N_chan = 288
        self.freq_arr = np.linspace(800e6, 1088e6, num=N_chan)
        self.models = (1, 2, 3, 4, 5, 7, 11)
        self.sampler = "nestle"

    def test_f1_QUfitting(self):
        if not (ONED_PATH / "simsource.dat").exists():
            create_1D_data(self.freq_arr, TEST_PATH, ONED_PATH)

        local_models = Path("models_ns")
        if not local_models.exists():
            shutil.copytree(TEST_PATH / ".." / "RMtools_1D" / "models_ns", local_models)

        failed_models = []
        for model in self.models:
            returncode = subprocess.call(
                f"qufit simdata/1D/simsource.dat --sampler {self.sampler} -m {model}",
                shell=True,
            )
            if returncode != 0:
                failed_models.append(model)

        self.assertEqual(
            failed_models, [], f"QU fitting failed to run for models: {failed_models}"
        )

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

    print("\nUnit tests running.")
    print("Test data inputs and outputs can be found in {}\n\n".format(os.getcwd()))

    unittest.TestLoader.sortTestMethodsUsing = None
    suite = unittest.TestLoader().loadTestsFromTestCase(test_QUfitting)
    unittest.TextTestRunner(verbosity=2).run(suite)
