"""
    test_timeseries.py file for testing the timeseries module.
"""

import math
import unittest

import numpy as np

from filterbank.filterbank import Filterbank
from timeseries.timeseries import Timeseries


class TestTimeseries(unittest.TestCase):
    """
           Testclass for testing the timeseries module.
    """

    timeseries_array = [-3.4771993, 28.411894, 1.5027914, 12.363131, -9.0473995,
                        13.312918, -8.387265, 2.095158, 9.921608, -1.1714606,
                        10.393141, -7.7038407, 24.99097, -15.776686, 0.53236485,
                        -11.958438, -6.911227, 8.95864, 11.76904, -2.0692701,
                        15.08512, -5.0073714, 0.35948455, -9.356973, -5.992894,
                        4.0259366, -7.884355, 6.3973246, 3.8018064, 13.149004,
                        8.768967, 5.0174375, 27.876976, -11.685825, 4.2599483,
                        20.358032, 9.604366, -12.078389, 4.7057323, -11.94255,
                        -17.29993, -2.19723, -9.520994, 2.9300866, 0.26252317,
                        -18.206154, -22.393536, -3.3634825, 2.3057923, 18.17247,
                        3.1966555, -5.1954975, 0.55871594, -1.2240806, -8.97003]

    def test_get_timeseries(self):
        """
            Tests the standard initialization of the timeseries object from
            a normal numpy array, also tests the retrieval functionality to retrieve
            the timeseries as an numpy array.
        :return:
        """

        timeseries_array = np.array(self.timeseries_array)
        get_timeseries = Timeseries(timeseries_array).get()

        self.assertEqual(timeseries_array.all(), get_timeseries.all())

    def test_from_filterbank(self):
        """
            Function that tests the from_filterbank functionaility,
            does it's own conversion to a timeseries from a filterbank file
            and cross-validates this with the modules implementation.
        :return:
        """
        filter_bank = Filterbank("./pspm32.fil")
        filter_bank.read_filterbank()

        timeseries = Timeseries().from_filterbank(filter_bank)
        _, samples = filter_bank.select_data()

        timeseries_check = []

        for sample_set in samples:
            summed_sample = np.sum(sample_set)
            timeseries_check.append(summed_sample)

        self.assertEqual(timeseries.get().all(), np.array(timeseries_check).all())

    def test_downsample_length(self):
        """
            Function to test the downsample functionality based up on the
            array length. The test array shall be 3 times smaller (based on downsample_rate_q)
            than the input array.
        :return:
        """
        downsample_rate_q = 3

        # Downsample by 3 so array should be 3 times smaller.

        downsamples_arr = Timeseries(self.timeseries_array).downsample(downsample_rate_q)

        self.assertEqual(len(downsamples_arr),
                         math.ceil(len(self.timeseries_array) / downsample_rate_q))
