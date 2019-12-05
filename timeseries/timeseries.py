"""
    Time series module for all time_series related operations.
    Kept as global as possible to ensure compatibility.
    along the line. Shall be used for more than just 'filterbank' operations.
"""

import numpy as np


class Timeseries:
    """
        Time series class handles all timeseries related operations.
        Every operation returns the timeseries itself again.
        Can be initialized using a normal python array or,
         filterbank file by calling from_filterbank()
    """
    timeseries = []

    def __init__(self, timeseries_arr=None):
        self.timeseries = np.array(timeseries_arr)

    def downsample(self, sample_scale):
        """
            Decimated/Downsamples the timeseries by a given scale 'sample_scale'
        """
        downsampled_ts = self.timeseries[0: self.timeseries.size: sample_scale]
        self.timeseries = downsampled_ts

        return self.timeseries


    def get(self):
        """
           Returns the current timeseries object which,
            could have been manipulated using functions below.
        """
        return self.timeseries


    def from_filterbank(self, filterbank_object):
        """
               Initializes timeseries object from filterbank object,
               while on the fly converting all 'combined' channels into
               one summed intensity plot.
        """
        time_series = []

        _, samples = filterbank_object.select_data()

        for sample_set in samples:
            summed_sample = np.sum(sample_set)
            time_series.append(summed_sample)

        self.timeseries = time_series

        return Timeseries(time_series)

    def from_filterbank_data(self, filterbank_data):
        """
               Initializes timeseries object from filterbank data,
               while on the fly converting all 'combined' channels into
               one summed intensity plot.
        """
        time_series = []

        for sample_set in filterbank_data:
            summed_sample = np.sum(sample_set)
            time_series.append(summed_sample)

        self.timeseries = time_series

        return Timeseries(time_series)
