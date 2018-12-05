"""
    Utilities for performing clipping on Filterbank data
"""

import numpy as np


def clipping(channels, samples):
    """
        Perform clipping on samples
    """
    samples = filter_samples(samples)
    # select first max 2000 samples
    samples = filter_channels(samples[:2000])


def filter_samples(samples):
    """
        Calulate mean power of all frequencies per time sample
        and remove samples with significantly high power
    """
    factor = 1.3
    new_samples = []
    # calculate mean intensity per sample
    avg_sample = np.sum(samples)/len(samples)
    # remove samples with significant high power
    for sample in samples:
        if np.sum(sample) <= (avg_sample * factor):
            new_samples.append(sample)
    return new_samples
    
def filter_channels(samples):
    """
        Calculate mean power of all time samples per frequency
        and remove frequencies with significantly high power
    """
    factor = 1.3
    bad_channels = []
    # calculate the average power per channel
    avg_power_per_chan = [sum(sample) for sample in zip(*samples)]
    # calculate the average power for all channels
    avg_power = sum(avg_power_per_chan)/len(avg_power_per_chan)
    # find channels with significant high power
    for i in range(len(avg_power_per_chan)):
        if avg_power_per_chan[i] >= (avg_power * factor):
            bad_channels.append(i)
    # remove bad channels from samples
    return np.delete(samples, bad_channels)