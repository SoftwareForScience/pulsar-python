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
import dedisperse.dedisperse as dedisperse
from plot.static_waterfall import waterfall_plot

from timeseries.timeseries import Timeseries

def print_possible_dm(data):
        '''
        This method is for testing purposes and should probably be removed later on
        '''
        first_sample = 400

        for i, data_point in enumerate(data[first_sample]):
                if(data_point > 10):
                        print("first sample ", i, " - ", data_point)
                        break

        highest_difference = 0

        for s, samples in enumerate(data):
                for i, data_point in enumerate(samples):
                        if(i > highest_difference):
                                
                                if(data_point > 10):
                                        print(s, i, " - ", data_point)                                        
                                        highest_difference = i
                                break

        print(highest_difference)


# Read filterbank data
special_pspm = fb.Filterbank(filename = "../data/my_special_pspm.fil")

special_pspm.read_filterbank()

frequencies, samples = special_pspm.select_data()

# Use this if you have your own file with a clear pulsar signal, this method assumes all signals other than the pulsar are lower than 10
#print_possible_dm(dispersed_samples)

# Dispersion Measure
DM = 230

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

samples = dedisperse(samples, DM)

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