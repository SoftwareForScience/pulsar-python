'''
    Example that dedisperses filterbank data and plots it together with a timeseries plot
'''
# pylint: disable-all
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import matplotlib.pyplot as plt

import filterbank.filterbank as fb
import dedisperse as dedisperse
from plot.static_waterfall import waterfall_plot

from clipping import clipping

# Read filterbank data,

# Standard file
special_pspm = fb.Filterbank(filename = "./my_special_pspm.fil")
highest_x=10
max_delay=10

special_pspm.read_filterbank()

frequencies, samples = special_pspm.select_data()

# Plot the original data
plt.subplot(2,1,1)
data, extent = waterfall_plot(samples, frequencies)

img = plt.imshow(data.T,
                 aspect='auto',
                 origin='lower',
                 rasterized=True,
                 interpolation='nearest',
                 extent=extent,
                 cmap='cubehelix')

plt.colorbar()

time_series = []

for sample_set in samples:
            summed_sample = np.sum(sample_set)
            time_series.append(summed_sample)

plt.subplot(2,1,2)
plt.plot(time_series)
plt.show()

# Dedisperse the samples
samples = dedisperse.dedisperse(samples, highest_x, max_delay)

# Plot the dedispersed data
plt.subplot(2,1,1)
data, extent = waterfall_plot(samples, frequencies)

img = plt.imshow(data.T,
                 aspect='auto',
                 origin='lower',
                 rasterized=True,
                 interpolation='nearest',
                 extent=extent,
                 cmap='cubehelix')

time_series = []

for sample_set in samples:
            summed_sample = np.sum(sample_set)
            time_series.append(summed_sample)

plt.subplot(2,1,2)
plt.plot(time_series)
plt.show()
