import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
from fourier import fourier
from plot import psd

# Initiate RtlSdr
sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.4e6
sdr.center_freq = 102.2e6

# Read samples
samples = sdr.read_samples(1024)

# Close RTLSDR device connection
sdr.close()

# Number of samples equals the length of samples
N = samples.shape[0]

# T equals N/Fs 
T = N/sdr.sample_rate

# Get the powerlevels and the frequencies
Pxx, freqs, _ = psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, scale_by_freq=True, sides='twosided')

# Calculate the powerlevel dB's 
power_levels = 10*np.log10(Pxx/(sdr.sample_rate/1e6))

# Add the center frequency to the frequencies so it matches the actual frequencies
freqs = freqs + sdr.center_freq/1e6

# Plot the PSD
plt.plot(freqs, power_levels)
plt.show()
