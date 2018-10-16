"""
    Example of plotting a Power Spectral Density plot, using filterbank data
"""
# pylint: disable-all
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from filterbank.filterbank import Filterbank
import matplotlib.pyplot as plt
import numpy as np
from plot import psd
from filterbank.header import read_header

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

print(center_freq)
# Get the powerlevels and the frequencies
PXX, freqs = psd(samples, nfft=1024, sample_rate=80,
                    scale_by_freq=True, sides='twosided')

# Calculate the powerlevel dB's
power_levels =  10 * np.log10(PXX/(80))

# Add the center frequency to the frequencies so they match the actual frequencies
freqs = freqs + center_freq

# Plot the PSD
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (dB)')
plt.plot(freqs, power_levels)
plt.show()