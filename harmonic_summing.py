# !!!!WORK IN PROGRESS!!!!
# TODO: implement algorithm for higher harmonics
# TODO: 1. make final addition and subtraction dependent on difference between perfect and recorded harmonics.
# TODO: 2. make sure that every recorded frequency close to perfect harmonic frequency are affected.
# TODO: something with the output, maybe a plot or even another filterbank file.
# TODO: ask if power is being read correctly. (spoiler: probably not)
import filterbank.filterbank as filterbank
import pandas as pd
import numpy as np

# uncomment following options to print dataframes in their entirety
#pd.set_option('display.max_rows', 1000)
#pd.set_option('display.max_columns', 1000)

#set number of harmonics to look for
numUpperHarmonics = 5
numLowerHarmonics = 3

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

fb = filterbank.Filterbank(filename='./pspm32.fil', read_all=True)
# get filertbank data + frequency labels
fil_data = fb.select_data()

# subtract 226 FOR TESTING ONLY
frequencies = np.array(fil_data[0])
frequencies -= 426
frequencies = frequencies.round(3)

# create dataframe with each channel as column (frequencies as column labels)
fil_dataframe = pd.DataFrame(fil_data[1], columns=frequencies)

print(fil_dataframe)

# find the overall most powerful frequency
most_pwrful_freq = fil_dataframe.sum().idxmax(axis=0, skipna=True)
print('Most powerful frequency = ', str(most_pwrful_freq))
print('Sum amplitude of mpf = ', fil_dataframe.sum().max())

# create empty array which we'll soon use for the lower harmonics
low_harms = np.empty([numLowerHarmonics])

# calculate the perfect lower harmonic frequencies
multiplier = 1
for i in range(len(low_harms)):
    multiplier = multiplier / 2
    low_harms[i] = most_pwrful_freq * multiplier

print(low_harms)

# find the closest recorded frequencies and replace perfect harmonics with found freqs
for i in range(len(low_harms)):
    low_harms[i] = find_nearest(frequencies, low_harms[i])

print(low_harms)

# add each sample of the recorded harmonics to the fundamental frequency's samples. set harmonics' samples to zero.
for i in range(len(low_harms)):
    fil_dataframe[most_pwrful_freq] = fil_dataframe[most_pwrful_freq] + fil_dataframe[low_harms[i]]
    fil_dataframe[low_harms[i]] = 0

print(fil_dataframe)







