"""
    Example of clipping RFI from filterbank data
"""
# pylint: disable-all
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import matplotlib.pyplot as plt
from timeseries.timeseries import Timeseries
from filterbank.filterbank import Filterbank
from plot.static_waterfall import waterfall_plot
from clipping.clipping import clipping

# init filterbank object
fil = Filterbank(filename='./pspm32.fil', read_all=True)
# retrieve channels and samples from filterbank
freqs, samples = fil.select_data()

print(freqs.shape, samples.shape)

# visualize data before clipping
time_series = Timeseries()

time_series.from_filterbank(fil)

plt.subplot(2,1,1)
plt.plot(time_series.timeseries)

# perform clipping on the filterbank data
new_freqs, new_samples = clipping(freqs, samples)

print(new_freqs.shape, new_samples.shape)

# draw time series after clipping
time_series = Timeseries(new_samples)

plt.subplot(2,1,2)
plt.plot(time_series.timeseries)
plt.show()
# draw waterfall plots
data, extent = waterfall_plot(samples, freqs)

plt.subplot(211)

img = plt.imshow(data.T,
                 aspect='auto',
                 origin='lower',
                 rasterized=True,
                 interpolation='nearest',
                 extent=extent,
                 cmap='cubehelix')


new_data, new_extent = waterfall_plot(new_samples, new_freqs)

plt.subplot(212)

new_img = plt.imshow(new_data.T,
                 aspect='auto',
                 origin='lower',
                 rasterized=True,
                 interpolation='nearest',
                 extent=new_extent,
                 cmap='cubehelix')

plt.show()
