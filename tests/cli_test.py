"""Tests for CLI."""

import subprocess
import unittest


class test_cli(unittest.TestCase):
    def test_cli_rmsynth1d(self):
        """Tests that the CLI `rmsynth1d` runs."""
        res = subprocess.run(["rmsynth1d", "--help"])
        self.assertEqual(res.returncode, 0)

    def test_cli_rmsynth3d(self):
        """Tests that the CLI `rmsynth3d` runs."""
        res = subprocess.run(["rmsynth3d", "--help"])
        self.assertEqual(res.returncode, 0)

    def test_cli_rmclean1d(self):
        """Tests that the CLI `rmclean1d` runs."""
        res = subprocess.run(["rmclean1d", "--help"])
        self.assertEqual(res.returncode, 0)

    def test_cli_rmclean3d(self):
        """Tests that the CLI `rmclean3d` runs."""
        res = subprocess.run(["rmclean3d", "--help"])
        self.assertEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main()
