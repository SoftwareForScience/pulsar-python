
import pandas as pd
import numpy as np

# uncomment following options to print dataframes in their entirety
# pd. set_option('display.max_rows', 10000)
# pd.set_option('display.max_columns', 10000)

# set number of harmonics to look for


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def match_and_add_harmonics(fil_dataframe, frequencies, most_pwrful_freq, perfect_harms, precision):
    """
    this function is used by apply_harmonic_summing().
    It finds the nearest recorded harmonics in the data frame to the to
    the calculated 'perfect' harmonics (calculated in apply_harmonic_summing())
    and adds the intensity to the supposed funamental component.
    :param fil_dataframe: data frame of the filterbank data.
    :param frequencies: frequency labels of the filterbank data.
    :param most_pwrful_freq: overall most powerful frequency component.
    :param perfect_harms: calculated perfect harmonics.
    :param precision: a decimal number e.g. 0.001 which determines the threshold for 'finding'
        the nearest recorded frequency or concluding that the frequency is not present in the
        filterbank data.
    """
    for i in range(len(perfect_harms)):
        harms = np.empty([len(perfect_harms)])
        harms[i] = find_nearest(frequencies, perfect_harms[i])
        if abs(harms[i] - perfect_harms[i]) < (perfect_harms[i] * precision):
            fil_dataframe[most_pwrful_freq] = fil_dataframe[most_pwrful_freq] + fil_dataframe[harms[i]]
            fil_dataframe[harms[i]] = 0
        else:
            print('no harmonic close to ' + str(perfect_harms[i]))


def apply_harmonic_summing(frequencies, fil_data, precision, num_lo_up_harmonics):
    """
    Top level function of this file.
    Converts imported filterbank (fourier transformed) data to pandas DataFrame,
    detects the most intense frequency component, calculates upper and lower harmonic frequencies,
    finds recorded frequencies which are nearest to the calculated harmonic frequencies (if within
    range specified by 'precision'),
    adds the sample values of the matching frequency components to the fundamental and zeros matching
    components' values out.
    :param frequencies: frequency labels from filterbank data.
    :param fil_data: data frame of the filterbank data.
    :param precision: a decimal number e.g. 0.001 which determines the threshold for 'finding'
        the nearest recorded frequency or concluding that the frequency is not present in the
        filterbank data.
    :param num_lo_up_harmonics: tuple of 2 integers specifying the number of lower and upper
        harmonics to find and add to the fundamental.
    :return: a tuple of processed filterbank data, same format as output of filterbank.read_all()
    """

    num_lower_harmonics, num_upper_harmonics = num_lo_up_harmonics

    # subtract 226 FOR TESTING ONLY
    frequencies -= 426
    frequencies = frequencies.round(3)

    # create dataframe with each channel as column (frequencies as column labels)
    fil_dataframe = pd.DataFrame(fil_data, columns=frequencies).abs()

    # print(fil_dataframe)

    # find the overall most powerful frequency
    most_pwrful_freq = fil_dataframe.sum().idxmax(axis=0, skipna=True)
    print('Most powerful frequency = ', str(most_pwrful_freq))
    print('Sum amplitude of mpf = ', fil_dataframe.sum().max())

    # create empty array which we'll soon use for the lower harmonics
    low_perfect_harms = np.empty([num_lower_harmonics])
    high_perfect_harms = np.empty([num_upper_harmonics])

    # calculate the perfect lower harmonic frequencies
    multiplier = 1
    for i in range(len(low_perfect_harms)):
        multiplier = multiplier / 2
        low_perfect_harms[i] = most_pwrful_freq * multiplier

    # print(low_perfect_harms)

    # calculate the perfect upper harmonic frequencies
    for i in range(len(high_perfect_harms)):
        high_perfect_harms[i] = most_pwrful_freq + (most_pwrful_freq * (i + 1))

    # print(high_perfect_harms)

    # find the closest recorded frequencies to the calculated harmonics
    # add each sample of the recorded harmonics to the fundamental frequency's samples. set harmonics' samples to zero.
    # for high harmonics
    match_and_add_harmonics(fil_dataframe, frequencies, most_pwrful_freq, high_perfect_harms, precision)
    # for low harmonics
    match_and_add_harmonics(fil_dataframe, frequencies, most_pwrful_freq, low_perfect_harms, precision)

    return frequencies, fil_dataframe.to_numpy()
