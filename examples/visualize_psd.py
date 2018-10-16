"""
    Example of plotting a Power Spectral Density plot, using RTLSDR data
"""
import os
import sys
import inspect
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
from plot import psd

# Initiate RtlSdr
SDR = RtlSdr()

# configure device
SDR.sample_rate = 2.4e6
SDR.center_freq = 102.2e6

# Read samples
SAMPLES = SDR.read_samples(1024)

# Close RTLSDR device connection
SDR.close()

# Number of samples equals the length of samples
N = SAMPLES.shape[0]

# T equals N/Fs
T = N/SDR.sample_rate

# Get the powerlevels and the frequencies
PXX, freqs, _ = psd(SAMPLES, NFFT=1024, Fs=SDR.sample_rate/1e6,
                    scale_by_freq=True, sides='twosided')

# Calculate the powerlevel dB's
POWER_LEVELS = 10*np.log10(PXX/(SDR.sample_rate/1e6))

# Add the center frequency to the frequencies so it matches the actual frequencies
freqs = freqs + SDR.center_freq/1e6

# Plot the PSD
plt.plot(freqs, POWER_LEVELS)
plt.show()
