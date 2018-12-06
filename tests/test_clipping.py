"""
    clipping.py unit tests
"""

import unittest
import numpy as np
from .context import clipping # pylint: disable-msg=E0611

class TestClipping(unittest.TestCase):
    """
        Class for testing clipping.py
    """

    fil_chans = list(range(430, 438))

    fil_vector = [[1, 1, 1, 1, 5, 1, 1, 1],
                  [1, 1, 1, 1, 5, 1, 1, 1],
                  [5, 5, 5, 5, 5, 5, 5, 5],
                  [1, 1, 1, 1, 5, 1, 1, 1],
                  [1, 1, 1, 1, 5, 1, 1, 1],
                  [1, 1, 1, 1, 5, 1, 1, 1]]

    def test_filter_samples(self):
        """
            When filtering samples, all time samples
            with a relative high power should be removed
        """
        result = clipping.filter_samples(self.fil_vector)
        expect = self.fil_vector
        del expect[2]
        self.assertListEqual(result, expect)
    
    def test_filter_channels(self):
        """
            When filtering samples, all channels
            with a relative high power should be removed
        """
        bad_channels = [4]
        result_channels, result_samples = clipping.filter_channels(self.fil_chans, self.fil_vector)
        expect_channels = np.delete(self.fil_chans, bad_channels)
        expect_samples = np.delete(self.fil_vector, bad_channels, axis = 1)
        self.assertEqual(result_channels.all(), expect_channels.all())
        self.assertEqual(result_samples.all(), expect_samples.all())
