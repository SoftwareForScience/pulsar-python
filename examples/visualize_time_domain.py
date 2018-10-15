from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np

# Initiate RtlSdr
sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.4e6
sdr.center_freq = 102.2e6
# sdr.gain = 0

# Read samples
samples = sdr.read_samples(256*1024)

# Close RTLSDR device connection
sdr.close()

# Number of samples equals the length of samples
N = samples.shape[0]

# T equals N/Fs 
T = N/sdr.sample_rate

# Define the x axis
x = np.linspace(0.0, T, N)

# Generate the plot
fig, ax = plt.subplots()
ax.plot(x, samples)
plt.show()
