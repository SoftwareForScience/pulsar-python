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
fb = Filterbank(filename='examples/pspm32bit.fil', as_stream=False)

# read the data in the filterbank file
f, samples = fb.select_data()

# Read the header of the filterbank file
header = read_header('examples/pspm32bit.fil')

# Calculate the center frequency with the data in the header
center_freq = header[b'fch1'] + float(header[b'nchans']) * header[b'foff'] / 2.0

print(samples.shape)
# Get the powerlevels and the frequencies
print(samples[0])
power_levels, freqs, _ = opsd(samples[0], nfft=128, sample_rate=80, sides='twosided')
# freqs, power_levels = psd(samples[0], 80, center_freq, nfft=128)

# Plot the PSD
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (arbitrary units)')
plt.plot(f, power_levels)
plt.show()
