import filterbank.filterbank as filterbank
import matplotlib.pyplot as plt
from harmonic_summing import harmonic_summing as harmsum
from timeseries.timeseries import Timeseries
import pandas as pd
import numpy as np
from plot import waterfall_plot
from clipping import clipping

fb = filterbank.Filterbank(filename='../pspm32.fil', read_all=True)
print(fb.get_header())
# Get time between values of sample
header = fb.get_header()
tsamp = header[b'tsamp']
fch1 = header[b'fch1']
nifs = header[b'nifs']
foff = header[b'foff']
nchans = header[b'nchans']
# observation time (default)
tobs = 10

nsamples = tobs / tsamp
freqs, fil_data = fb.select_data()
print("----------something-------------")
N = nifs * nchans * nsamples
print('N = ' + str(N))
print(freqs)
print(fil_data[1])
skyfreq = fch1 + fil_data[1] * foff
# print('skyfreq:')
# print(skyfreq)
# Set absolute values of Fil_data, since we don't know what negative values mean
fil_dataframe = pd.DataFrame(fil_data, columns=freqs).abs()
most_pwrful_freq2 = fil_dataframe.sum().idxmax(axis=0, skipna=True)
columnsData2 = fil_dataframe.loc[:, most_pwrful_freq2]
fundamental2 = columnsData2.to_numpy()
print("--------------fundamental FROM FILTERBANK-----------------")
print(fundamental2)
fil_data2 = fil_dataframe.to_numpy()
harmsum1 = harmsum.apply_harmonic_summing(frequencies=freqs, fil_data=fil_data2, precision=0.001,
                                          num_lo_up_harmonics=(5, 5))
freqs1, samples1 = harmsum1
fil_dataframe2 = pd.DataFrame(samples1, columns=freqs1)

most_pwrful_freq = fil_dataframe2.sum().idxmax(axis=0, skipna=True)
print(most_pwrful_freq2)

# Get Sample of the fundamental
columnsData = fil_dataframe2.loc[:, most_pwrful_freq]
fundamental = columnsData.to_numpy()
print("--------------fundamental FROM HARMONIC SUMMING-----------------")
print(fundamental)
ax = plt.gca()
columnsData.plot(kind='line', x='Sample', y=most_pwrful_freq, ax=ax)
plt.xlabel('time (us)', fontsize=12)
plt.ylabel('Intensity', fontsize=12)
plt.show()


# Plot fundamental2
# time_series = Timeseries(fundamental2)
# plt.subplot(2,1,2)
# plt.plot(time_series.timeseries)
# plt.xlabel('time (us)', fontsize=12)
# plt.ylabel('Intensity', fontsize=12)
# plt.show()

# Plot fundamental
# time_series1 = Timeseries(fundamental)
# plt.subplot(2,1,2)
# plt.plot(time_series1.timeseries)
# plt.xlabel('time', fontsize=12)
# plt.ylabel('Intensity', fontsize=12)
# plt.show()
def get_fundamental(frequencies, filterbank_data):
    fil_dataframe = pd.DataFrame(filterbank_data, columns=frequencies)
    most_pwrful_freq = fil_dataframe.sum().idxmax(axis=0, skipna=True)
    columnsData = fil_dataframe2.loc[:, most_pwrful_freq]
    fundamental = columnsData.to_numpy()
    ax = plt.gca()
    columnsData.plot(kind='line', x='Sample', y=most_pwrful_freq, ax=ax)
    plt.xlabel('time (us)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.show()
    return fundamental


def search_periodicity(fundamental):
    fundamental2 = np.copy(fundamental)
    fundamental2[::-1].sort()
    max_values = []
    for i in range(2):
        max_values.append(fundamental2[i])

    x = np.where(fundamental == max_values[0])
    y = np.where(fundamental == max_values[1])
    time = tsamp * x - tsamp * y
    return time


search_periodicity(fundamental)
