"""
    Example of plotting a Power Spectral Density plot, using RTLSDR data
"""
# pylint: disable-all
import os
import sys
import inspect

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

# Install rtlsdr package from pyrtlsdr
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
from plot import opsd

# Initiate RtlSdr
sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.4e6
sdr.center_freq = 102.2e6

# Read samples
samples = sdr.read_samples(1024)

# Close RTLSDR device connection
sdr.close()

print(samples[1:100])
# Number of samples equals the length of samples
sample_length = samples.shape[0]

# Get the powerlevels and the frequencies
PXX, freqs, _ = opsd(samples, nfft=1024, sample_rate=sdr.sample_rate/1e6,
                 scale_by_freq=True, sides='twosided')

# Calculate the powerlevel dB's
power_levels = 10 * np.log10(PXX/(sdr.sample_rate/1e6))

# Add the center frequency to the frequencies so it matches the actual frequencies
freqs = freqs + sdr.center_freq/1e6

# Plot the PSD
plt.plot(freqs, power_levels)
plt.show(block=True)
