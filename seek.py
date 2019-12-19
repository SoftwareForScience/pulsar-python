from clipping import clipping
from dedisperse import dedisperse
from dedisperse.dedisperse import find_estimation_intensity

from timeseries.timeseries import Timeseries
import filterbank.filterbank as filterbank
import matplotlib.pyplot as plt
from harmonic_summing import harmonic_summing as harmsum
from timeseries.timeseries import Timeseries
from pandas import DataFrame
import pandas as pd
size = 512
fb = filterbank.Filterbank(filename='./pspm32.fil', read_all=True)
# get filertbank data + frequency labels
print(fb.get_header())

freqs, fil_data = fb.select_data()

# Transform filterbank data into dataframe
fil_dataframe = pd.DataFrame(fil_data, columns=freqs).abs()
fd = fil_dataframe.max(axis=0).max()
print(fd)
print("<---------------------------------------------------->")
fil_data2 = fil_dataframe.to_numpy()
print(fil_data2)
print("<---------------------------------------------------->")

# Apply Clipping
new_freqs, new_samples = clipping(freqs, fil_data2)
print("<----------------------Samples from Clipping------------------------------>")
print(new_samples)
# print(new_freqs)
print("<---------------------------------------------------->")

# Apply Dedisperse TODO doesn't Work dedisperse method
# deds = dedisperse(new_samples, 235)
# print(deds)
# average = find_estimation_intensity(new_samples,5)
# print("Average" + str(average))
# print("<---------------------------------------------------->")

# Apply harmonic summing on fb data
harmsum1 = harmsum.apply_harmonic_summing(frequencies=freqs, fil_data=fil_data2, precision=0.001, num_lo_up_harmonics=(5, 5))
# print(harmsum)
freqs2, samples2 = harmsum1

# Apply Harmonic summing after clipping TODO obtaining same result doing harmonic summing and from clipping
harmsum2 = harmsum.apply_harmonic_summing(frequencies=new_freqs, fil_data=new_samples, precision=0.001,
                                          num_lo_up_harmonics=(5, 5))
freqs3, samples3 = harmsum2

print("<---------------------------------------------------->")
print(freqs3)
print("<---------------------------------------------------->")
print(freqs2)
print("<--------------------Original Samples-------------------------------->")
print(fil_data2)
print("<-------------------Samples after Clipping--------------------------------->")
print(new_samples)
print("<--------------------------Samples after Clipping and harmonic-------------------------->")
print(samples3)
print("<-------------------Samples after harmonic--------------------------------->")
print(samples2)
print("<---------------------------------------------------->")


time_series2 = Timeseries(samples2)

print(time_series2.get())
print("<---------------------------------------------------->")
time_series3 = Timeseries(fil_data2)
print(time_series3.get())
print("<---------------------------------------------------->")

fil_dataframe2 = pd.DataFrame(fil_data2, columns=freqs2)
most_pwrful_freq = fil_dataframe2.sum().idxmax(axis=0, skipna=True)
list_num = [x for x in range(len(fil_data2))]
print(list_num)
fil_dataframe2['Sample'] = list_num

print("most pwrful freq:" + str(most_pwrful_freq))
print("<---------------------------------------------------->")
columnsData = fil_dataframe2.loc[ : , most_pwrful_freq ]
print(columnsData)
print("<---------------------------------------------------->")
print(fil_dataframe2)
print("<---------------------------------------------------->")


# ax = plt.gca()
# fil_dataframe2.plot(kind='line',x='Sample',y=most_pwrful_freq,ax=ax)
# plt.show()

# Same thing but with the samples of the harmonic summing

fil_dataframe3 = pd.DataFrame(samples2, columns=freqs2)
most_pwrful_freq2 = fil_dataframe3.sum().idxmax(axis=0, skipna=True)
list_num = [x for x in range(len(fil_dataframe3))]
print(list_num)
fil_dataframe3['Sample'] = list_num

print("most pwrful freq:" + str(most_pwrful_freq2))
print("<---------------------------------------------------->")
columnsData = fil_dataframe3.loc[ : , most_pwrful_freq ]
print(columnsData)
print("<---------------------------------------------------->")
print(fil_dataframe3)
print("<---------------------------------------------------->")

signal = columnsData.to_numpy()
time_series4 = Timeseries(signal)

# plot filterbank data without changes
# plt.subplot(2,1,2)
# plt.plot(time_series3.timeseries)
# plt.show()

# plot filterbank data after harmonic summing
# plt.subplot(2,1,2)
# plt.plot(time_series2.timeseries)
# plt.show()

# plot signal with most intensity isolated from filterbank data after
# harmonic summing
ax = plt.gca()
fil_dataframe3.plot(kind='line',x='Sample',y=most_pwrful_freq,ax=ax)
plt.show()

# plot timeseries of signal with most intensity isolated from filterbank data after
# harmonic summing
plt.subplot(2,1,2)
plt.plot(time_series4.timeseries)
plt.show()
# plt.subplot(2,1,2)
# plt.plot(time_series.timeseries)
# plt.show()


