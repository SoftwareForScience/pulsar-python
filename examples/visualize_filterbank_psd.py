"""
    Example of plotting a Power Spectral Density plot, using filterbank data
"""
# pylint: disable-all
import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from plot import npsd
from filterbank.header import read_header
from filterbank.filterbank import Filterbank

# Instatiate the filterbank reader and point to the filterbank file
fb = Filterbank(filename='examples/pspm32.fil')

# read the data in the filterbank file
_, samples = fb.select_data()

# Convert 2D Array to 1D Array with complex numbers
samples = samples[0] + (samples[1] * 1j)

# Read the header of the filterbank file
header = read_header('examples/pspm32.fil')

# Calculate the center frequency with the data in the header
center_freq = header[b'fch1'] + float(header[b'nchans']) * header[b'foff'] / 2.0

# Get the powerlevels and the frequencies
freqs, power_levels = npsd(samples, sampling_frequency=80, center_frequency=center_freq)

# Plot the PSD
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (dB)')
plt.plot(freqs, power_levels)
plt.show()
