import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import matplotlib.pyplot as plt
import numpy as np
from fourier import fourier
from plot import psd
from data_import import _8bit, _16bit, _32bit

start_freq = 433e6
stop_freq = 434e6
center_freq = start_freq + ((stop_freq - start_freq) / 2)

print(center_freq)
samples = _32bit(start_freq, stop_freq)
sample_rate = 2.4e6

print(samples)

samples = samples[1]
# Number of samples equals the length of samples
N = samples.shape[0]

# T equals N/Fs 
T = N/sample_rate

# Get the powerlevels and the frequencies
Pxx, freqs, _ = psd(samples, NFFT=1024, Fs=sample_rate/1e6, scale_by_freq=True, sides='twosided')

# Calculate the powerlevel dB's 
power_levels = 10*np.log10(Pxx/(sample_rate/1e6))

# Add the center frequency to the frequencies so it matches the actual frequencies
freqs = freqs + center_freq/1e6

# Plot the PSD
plt.plot(freqs, power_levels)
plt.show()
