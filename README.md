# RM

RM-synthesis, RM-clean and QU-fitting on polarised radio spectra

 Python scripts to perform RM-synthesis, RM-clean and QU-fitting on
 polarised radio spectra.


 Initial version by Cormac R. Purcell  
 Currently maintained by CIRADA


Installation, usage instructions and detailed algorithm information can be found in the [wiki](https://github.com/CIRADA-Tools/RM-Tools/wiki).

Structure:  
RMtools_1D  ... Toolkit to produce Faraday spectra of single pixels.  
RMtools_3D  ... Toolkit to produce Faraday depth cubes.  
RMutils     ... Utilities for interacting with polarized data and Faraday depth 

This will make the following modules importable in Python: RMtools_1D, RMtools_3D, RMutil

Five terminal commands are added to invoke the main tools:  
rmsynth1d  
rmclean1d  
rmsynth3d  
rmclean3d  
qufit

Use these commands with a -h flag to get information on the usage of each. Full documentation is on the [wiki](https://github.com/CIRADA-Tools/RM-Tools/wiki).

If you use this package in a publication, please include a footnote with a link to the Github repository.  
A paper with a full description of the package, as well as an ASCL code reference, are being prepared but are not available yet.

Questions, bug reports, and feature requests can be posted to the GitHub issues page or sent to Cameron Van Eck, cameron.van.eck (at) dunlap.utoronto.ca.

More information on the Canadian Initiative for Radio Astronomy Data Analysis (CIRADA) can be found at cirada.org.

