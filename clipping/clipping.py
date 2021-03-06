"""
    Utilities for performing clipping on Filterbank data
"""

import numpy as np


def clipping(channels, samples):
    """
        Perform clipping on samples
    """
    n_samples = 2000
    # remove all rows(samples) with noise
    samples = filter_samples(samples)
    # remove all columns(channels) with noise, select first max n samples
    bad_channels = filter_channels(samples[:n_samples])
    # remove bad channels from samples
    channels = np.delete(channels, bad_channels)
    samples = np.delete(samples, bad_channels, axis=1)
    # remove all individual cells with noise
    samples = filter_indv_channels(samples)
    return channels, samples


def filter_samples(samples, factor=11):
    """
        Calulate mean power of all frequencies per time sample
        and remove samples with significantly high power
    """
    new_samples = list()
    # calculate mean intensity per sample
    avg_sample = np.sum(samples)/len(samples)
    # remove samples with significant high power
    for sample in samples:
        if np.sum(sample) <= (avg_sample * factor):
            new_samples.append(sample)
    return np.array(new_samples)


def filter_channels(samples, factor=9):
    """
        Calculate mean power of all time samples per frequency
        and remove frequencies with significantly high power
    """
    bad_channels = list()
    # calculate the mean power per channel
    avg_power_chan = samples.mean(axis=0)
    # calculate the standard deviation per channel
    sd_power_chan = samples.std(axis=0)
    # calculate the mean power for all channels
    avg_power = sum(avg_power_chan)/len(avg_power_chan)
    # find channels with significant high power
    for i, (avg_channel, sd_channel) in enumerate(zip(avg_power_chan, sd_power_chan)):
        if avg_channel >= (avg_power * factor) or sd_channel >= (avg_power * factor):
            bad_channels.append(i)
    return bad_channels


def filter_indv_channels(samples, factor=9):
    """
        Calculate mean power per frequency
        and remove samples with significantly high power
    """
    new_samples = np.zeros((len(samples), len(samples[0])))
    # calculate the mean power for each sample per channel
    avg_power_chan = samples.mean(axis=0)
    # calculate the median power for each sample per channel
    med_power_chan = np.median(samples, axis=0)
    # iterator over columns(channels)
    for i, channel in enumerate(samples.T):
        # replace the intensity with the median of the channel,
        # if the intensity is higher than the average of that channel
        channel[channel > (avg_power_chan[i] * factor)] = med_power_chan[i]
        new_samples[:, i] = channel
    return new_samples
