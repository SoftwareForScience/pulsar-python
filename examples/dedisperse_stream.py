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

from timeseries.timeseries import Timeseries

from clipping import clipping

# Read filterbank data
special_pspm = fb.Filterbank(filename = "../data/my_uber_pspm.fil")

special_pspm.read_filterbank()

frequencies, samples = special_pspm.select_data()


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

#clipped_samples = clipping(frequencies, samples)
samples = dedisperse.dedisperse(samples)
#samples = dedisperse.find_lowest_pulsar(samples)
#samples = dedisperse.estimate_dm(samples)
#samples = dedisperse.find_initial_line(samples)

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
