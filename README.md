![Tests](https://github.com/CIRADA-Tools/RM-tools/actions/workflows/python-package.yml/badge.svg) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/CIRADA-Tools/RM-Tools/master.svg)](https://results.pre-commit.ci/latest/github/CIRADA-Tools/RM-Tools/master)

# RM-Tools

RM-synthesis, RM-clean and QU-fitting on polarised radio spectra

 Python scripts to perform RM-synthesis, RM-clean and QU-fitting on
 polarised radio spectra.


 Initial version by Cormac R. Purcell
 Currently hosted by CIRADA and maintained by Cameron Van Eck

## Installation / Usage
Installation, usage instructions and detailed algorithm information can be found in the [wiki](https://github.com/CIRADA-Tools/RM-Tools/wiki).

## Structure:
- RMtools_1D  ... Toolkit to produce Faraday spectra of single pixels.
- RMtools_3D  ... Toolkit to produce Faraday depth cubes.
- RMutils     ... Utilities for interacting with polarized data and Faraday depth

![RM-Tools component diagram](https://github.com/CIRADA-Tools/RM-Tools/wiki/diagram.png)

Five terminal commands are added to invoke the main tools:
- `rmsynth1d`
- `rmclean1d`
- `rmsynth3d`
- `rmclean3d`
- `qufit`

Use these commands with a -h flag to get information on the usage of each. Full documentation is on the [wiki](https://github.com/CIRADA-Tools/RM-Tools/wiki).

The following terminal commands are available to access the [additional tools](https://github.com/CIRADA-Tools/RM-Tools/wiki/Tools):
- `rmtools_freqfile`
- `rmtools_calcRMSF`
- `rmtools_testdata1D`
- `rmtools_createchunks`
- `rmtools_assemblechunks`
- `rmtools_fitIcube`
- `rmtools_peakfitcube`
- `rmtools_testdata3D`
- `rmtools_extractregion`


## Citing
If you use this package in a publication, please cite the [ASCL entry](https://ui.adsabs.harvard.edu/abs/2020ascl.soft05003P/abstract) for the time being. A paper with a full description of the package is being prepared but is not available yet.

More information on the Canadian Initiative for Radio Astronomy Data Analysis (CIRADA) can be found at cirada.ca.

RM-Tools is open source under an MIT License.

## Contributing
Contributions are welcome. Questions, bug reports, and feature requests can be posted to the GitHub issues page or sent to Cameron Van Eck, cameron.vaneck (at) anu.edu.au.

The development dependencies can be installed via `pip` from PyPI:
```bash
pip install "RM-Tools[dev]"
```
or for a local clone:
```bash
cd RM-Tools
pip install ".[dev]"
```

Code formatting and style is handled by `black` and `isort`, with tests run by `pytest`. A `pre-commit` hook is available to handle the autoformatting. After installing the `dev` dependencies, you can install the hooks by running:
```bash
cd RM-Tools
pre-commit install
```
