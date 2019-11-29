#-----------------------------------------------------------------------------#
#
# Python scripts to perform RM-synthesis, RM-clean and QU-fitting on
# polarised radio spectra. The spectra should be in ASCII format as
# columns in a space-delimited file of format:
#    [freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy]
# or
#    [freq_Hz, Q_Jy, U_Jy, dQ_Jy, dU_Jy]
#
# If I data is not present then the RM-synthesis code runs on the raw Q and U
# values, which may be fractional or absolute.
#
# If the I data is absent for the QU-fitting script then the code
# assumes that the Q and U values are fractional.
#
# Use the '-h' flag to show detailed help and usage information.
#
# by Cormac R. Purcell
#
#-----------------------------------------------------------------------------#
# EXAMPLES:

# Create a test dataset from parameters in a catalogue file.
./mk_test_ascii_data.py cats/catalogue.csv

# Perform RM-synthesis on the sources
./do_RMsynth_1D.py data/Source1.dat -p -S
./do_RMsynth_1D.py data/Source2.dat -p -S
./do_RMsynth_1D.py data/Source3.dat -p -S
./do_RMsynth_1D.py data/Source4.dat -p -S
./do_RMsynth_1D.py data/Source5.dat -p -S

# Perform RM-clean on the sources
./do_RMclean_1D.py data/Source1.dat -p
./do_RMclean_1D.py data/Source2.dat -p
./do_RMclean_1D.py data/Source3.dat -p
./do_RMclean_1D.py data/Source4.dat -p
./do_RMclean_1D.py data/Source5.dat -p

# Perform QU-fitting on the sources using the PyMultiNest sampler
./do_QUfit_1D_mnest.py data/Source1.dat -m 1 -p
./do_QUfit_1D_mnest.py data/Source2.dat -m 3 -p
./do_QUfit_1D_mnest.py data/Source3.dat -m 3 -p

