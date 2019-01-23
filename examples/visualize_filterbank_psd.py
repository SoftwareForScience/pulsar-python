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

from plot import opsd
from filterbank.header import read_header
from filterbank.filterbank import Filterbank

# Instatiate the filterbank reader and point to the filterbank file
fb = Filterbank(filename='./pspm32.fil', read_all=True)

# read the data in the filterbank file
f, samples = fb.select_data()

# Assign the center frequency with the data in the header
center_freq = fb.header[b'center_freq']

print(samples.shape)
# Get the powerlevels and the frequencies
print(samples[0])
power_levels, freqs, _ = opsd(samples[0], nfft=128, sample_rate=80, sides='twosided')

# Plot the PSD
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (arbitrary units)')
plt.plot(f, power_levels)
plt.show()
