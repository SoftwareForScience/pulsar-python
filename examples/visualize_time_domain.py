"""
    Example using matplotlib to plot raw RTLSDR data
"""
# pylint: disable-all
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np

# Initiate RtlSdr
sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.4e6
sdr.center_freq = 102.2e6

# Read samples
samples = sdr.read_samples(256*1024)

# Close RTLSDR device connection
sdr.close()

# Number of samples equals the length of samples
sample_length = samples.shape[0]

# sample size equals sample length / sample rate
sample_size = sample_length/sdr.sample_rate

# Define the x axis
intervals = np.linspace(0.0, sample_size, sample_length)

# Generate the plot
fig, ax = plt.subplots()
ax.plot(intervals, samples)
plt.show()
