import filterbank.filterbank as filterbank
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from harmonic_summing import harmonic_summing as harmsum
from plot import waterfall, waterfall_plot
import pylab as pyl

fb = filterbank.Filterbank(filename='./pspm32.fil', read_all=True)
# get filertbank data + frequency labels

freqs, fil_data = fb.select_data()
harmsum = harmsum.apply_harmonic_summing(frequencies=freqs, fil_data=fil_data, precision=0.001, num_lo_up_harmonics=(5, 5))
print(harmsum)

freqs, fil_data = harmsum


# Create DataFrame with the fil_data changed and the freqs
fil_dataframe = pd.DataFrame(fil_data, columns=freqs)

list_num = [x for x in range(len(fil_data))]
print(list_num)
fil_dataframe['Sample'] = list_num

# Get amplitudes of freq
ampl = fil_dataframe.sum(axis=0, skipna=True)
print(ampl.size)
inten = []

# Store in inten the amplitudes for each freq
# -1 since dataframe has 'Sample' column wich we don't need here
for i in range(ampl.size - 1):
    inten.append(ampl[i])
print(inten)

# Create datatFrame with freq and their amplitudes
ampl_freq = {'frequencies': freqs, 'Intensity': inten}
a = DataFrame(ampl_freq, columns=['frequencies', 'Intensity'])
print(a)

# Plot the dataFrame
a.plot(x='frequencies', y='Intensity', kind='line')
plt.show()




# Waterfall
data, extent = waterfall_plot(fil_data, freqs)
img = plt.imshow(data.T,
                 aspect='auto',
                 origin='lower',
                 rasterized=True,
                 interpolation='nearest',
                 extent=extent,
                 cmap='cubehelix')
plt.show()


