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

from plot import psd
from filterbank.header import read_header
from filterbank.filterbank import Filterbank

# Instatiate the filterbank reader and point to the filterbank file
fb = Filterbank(filename='examples/pspm32.fil')

# read filterbank at once
fb.read_filterbank()

# read the data in the filterbank file
_, samples = fb.select_data()

# print(samples[1].shape)

# Convert 2D Array to 1D Array
samples = samples.reshape((samples.shape[0]*samples.shape[1],))

# Read the header of the filterbank file
header = read_header('examples/pspm32.fil')

# Calculate the center frequency with the data in the header
center_freq = header[b'fch1'] + float(header[b'nchans']) * header[b'foff'] / 2.0

# Get the powerlevels and the frequencies
freqs, power_levels = psd(samples, 80, center_freq, nfft=None)

# Plot the PSD
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (arbitrary units)')
plt.plot(freqs, power_levels)
plt.show()
