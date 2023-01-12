"""Tests for importing modules."""

import unittest


class test_imports(unittest.TestCase):
    def test_imports(self):
        """Tests that package imports are working correctly."""
        # This is a bit of a weird test, but package imports
        # have not worked before.
        modules = [
            "RMtools_1D.calculate_RMSF",
            # 'RMtools_1D.do_QUfit_1D_mnest',
            "RMtools_1D.do_RMclean_1D",
            "RMtools_1D.do_RMsynth_1D_fromFITS",
            "RMtools_1D.do_RMsynth_1D",
            "RMtools_3D.make_freq_file",
            "RMtools_1D.mk_test_ascii_data",
            "RMtools_3D.assemble_chunks",
            "RMtools_3D.create_chunks",
            "RMtools_3D.do_fitIcube",
            "RMtools_3D.do_RMclean_3D",
            "RMtools_3D.do_RMsynth_3D",
            "RMtools_3D.extract_region",
            "RMtools_3D.mk_test_cube_data",
            "RMutils.corner",
            "RMutils.mpfit",
            "RMutils.nestle",
            "RMutils.normalize",
            "RMutils.util_FITS",
            "RMutils.util_misc",
            "RMutils.util_plotFITS",
            "RMutils.util_plotTk",
            "RMutils.util_rec",
            "RMutils.util_RM",
        ]
        for module in modules:
            __import__(module)


if __name__ == "__main__":
    unittest.main()
