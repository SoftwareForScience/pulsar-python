"""
    Example using matplotlib to plot raw RTLSDR data
"""
# pylint: disable-all
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np

# Initiate RtlSdr
SDR = RtlSdr()

# configure device
SDR.sample_rate = 2.4e6
SDR.center_freq = 102.2e6

# Read samples
SAMPLES = SDR.read_samples(256*1024)

# Close RTLSDR device connection
SDR.close()

# Number of samples equals the length of samples
N = SAMPLES.shape[0]

# T equals N/Fs
T = N/SDR.sample_rate

# Define the x axis
X = np.linspace(0.0, T, N)

# Generate the plot
FIG, AX = plt.subplots()
AX.plot(X, SAMPLES)
plt.show()
