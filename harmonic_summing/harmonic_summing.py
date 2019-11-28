# !!!!WORK IN PROGRESS!!!!
# TODO: implement algorithm for higher harmonics
# TODO: 1. make final addition and subtraction dependent on difference between perfect and recorded harmonics.
# TODO: 2. make sure that every recorded frequency close to perfect harmonic frequency are affected.
# TODO: something with the output, maybe a plot or even another filterbank file.
# TODO: ask if power is being read correctly with negative values. (spoiler: probably not)
# TODO: check if frequency.round(3) is allowed and if not, use a trick to make sure DataFrame doesn't truncate column
#       names, like converting them to strings.
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# uncomment following options to print dataframes in their entirety
# pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_columns', 1000)

# set number of harmonics to look for
numUpperHarmonics = 5
numLowerHarmonics = 3


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def apply_harmonic_summing(fil_data):
    print(type(fil_data))
    # subtract 226 FOR TESTING ONLY
    frequencies = np.array(fil_data[0])
    frequencies -= 426
    frequencies = frequencies.round(3)

    # create dataframe with each channel as column (frequencies as column labels)
    fil_dataframe = pd.DataFrame(fil_data[1], columns=frequencies).abs()

    print(fil_dataframe)

    # find the overall most powerful frequency
    most_pwrful_freq = fil_dataframe.sum().idxmax(axis=0, skipna=True)
    print('Most powerful frequency = ', str(most_pwrful_freq))
    print('Sum amplitude of mpf = ', fil_dataframe.sum().max())

    # create empty array which we'll soon use for the lower harmonics
    low_perfect_harms = np.empty([numLowerHarmonics])
    high_perfect_harms = np.empty([numUpperHarmonics])

    low_harms = np.empty([numLowerHarmonics])
    high_harms = np.empty([numUpperHarmonics])

    # calculate the perfect lower harmonic frequencies
    multiplier = 1
    for i in range(len(low_perfect_harms)):
        multiplier = multiplier / 2
        low_perfect_harms[i] = most_pwrful_freq * multiplier

    print(low_perfect_harms)

    # calculate the perfect upper harmonic frequencies
    for i in range(len(high_perfect_harms)):
        high_perfect_harms[i] = most_pwrful_freq + (most_pwrful_freq * (i + 1))

    print(high_perfect_harms)

    # find the closest recorded frequencies and replace perfect harmonics with found freqs
    # add each sample of the recorded harmonics to the fundamental frequency's samples. set harmonics' samples to zero.
    for i in range(len(low_perfect_harms)):
        low_harms[i] = find_nearest(frequencies, low_perfect_harms[i])
        if abs(low_harms[i] - low_perfect_harms[i]) < (low_perfect_harms[i] * 0.001):
            fil_dataframe[most_pwrful_freq] = fil_dataframe[most_pwrful_freq] + fil_dataframe[low_harms[i]]
            # fil_dataframe[low_harms[i]] = 0
        else:
            print('no harmonic close to ' + str(low_perfect_harms[i]))

    # find the closest recorded frequencies and replace perfect harmonics with found freqs
    # add each sample of the recorded harmonics to the fundamental frequency's samples. set harmonics' samples to zero.
    for i in range(len(high_perfect_harms)):
        high_harms[i] = find_nearest(frequencies, high_perfect_harms[i])
        if abs(high_harms[i] - high_perfect_harms[i]) < (high_perfect_harms[i] * 0.001):
            fil_dataframe[most_pwrful_freq] = fil_dataframe[most_pwrful_freq] + fil_dataframe[high_harms[i]]
            # fil_dataframe[high_harms[i]] = 0
        else:
            print('no harmonic close to ' + str(high_perfect_harms[i]))

    print(low_harms)
    print(fil_dataframe)
    list_num = [x for x in range(len(fil_data[1]))]
    # print(list_num)
    fil_dataframe['Sample'] = list_num


    # print(fil_dataframe)
    # Getting the columns of the dataframe, creating new dataframe and ploting to confirm values
    # print(list(fil_dataframe.columns.values.tolist()))
    # print(fil_dataframe[most_pwrful_freq].tolist())
    # s1 = fil_dataframe[most_pwrful_freq].tolist()
    # print(s1)
    # data_fund = {str(most_pwrful_freq): s1,
    #              'Sample': list_num}
    # df = DataFrame(data_fund, columns=[str(most_pwrful_freq), 'Sample'])
    # df.plot(x='Sample', y=str(most_pwrful_freq), kind='scatter')
    # plt.show()

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
    ampl_freq = {'frequencies': frequencies, 'Intensity' : inten}
    a = DataFrame(ampl_freq, columns=['frequencies', 'Intensity'])
    print(a)
    # Plot the dataFrame
    a.plot(x='frequencies', y='Intensity', kind='line')
    plt.show()





    # print(low_perfect_harms)
    # for i in low_harms:
    #     fil_dataframe.plot(x='Sample', y=i, kind='line')
    #     plt.show()
    # fil_dataframe.plot(x= 'Sample', y=most_pwrful_freq, kind='line')
    # plt.show()




