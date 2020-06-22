# Python scripts to perform RM-synthesis & RM-clean on polarised radio
# cubes. The spectral cubes should be in FITS format with NAXIS=3 or 4
# and a separate vector of frequencies should be provided in an ASCII
# file called freqs_Hz.dat.
#
# See comments in the individual files for more details.
# Instructions below show how generate artificial data and run the scripts.
#
# by Cormac R. Purcell
#
# Note: Print detailed help by running each script with a "-h" flag.
#
# EXAMPLES:

# Create a test dataset from parameters in a catalogue file.
./mk_test_cube_data.py catalogue.csv

# Optional: Create a model Stokes I cube by fiting a polynomial to
# each spectrum. Also save an ASCII noise spectrum measured from the residual.
./do_fitIcube.py data/StokesI.fits data/freqs_Hz.dat -b 50

# Perform RM-synthesis on the Stokes Q & U FITS files
./do_RMsynth_3D.py data/StokesQ.fits data/StokesU.fits data/freqs_Hz.dat
./do_RMsynth_3D.py data/StokesQ.fits data/StokesU.fits data/freqs_Hz.dat -n data/Inoise.dat -o "WithNoise_"
./do_RMsynth_3D.py data/StokesQ.fits data/StokesU.fits data/freqs_Hz.dat -i data/Imodel.fits -o "WithI_"

# Perform RM-clean on the dirty FDF
./do_RMclean_3D.py -c 0.12 data/FDF_tot_dirty.fits data/RMSF_tot.fits

