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
    # indices of bad channels and samples
    bad_channels = [4]
    bad_samples = [2]

    fil_chans = list(range(430, 438))

    fil_vector = [[1, 1, 1, 1, 5, 1, 1, 1],
                  [1, 1, 1, 1, 5, 1, 1, 1],
                  [5, 5, 5, 5, 5, 5, 5, 5],
                  [1, 1, 1, 1, 5, 1, 1, 1],
                  [1, 1, 5, 1, 5, 1, 1, 1],
                  [1, 1, 1, 1, 5, 1, 1, 1]]

    def test_clipping(self):
        """
            When running all clipping methods,
            expect all noise to be removed
        """
        result_chans, result_samples = clipping.clipping(self.fil_chans, self.fil_vector)
        expect_chans = np.delete(self.fil_chans, self.bad_channels)
        expect_samples = np.full((len(self.fil_vector)-len(self.bad_samples),
                                  len(self.fil_vector[0])), 1)
        self.assertEqual(result_chans.all(), expect_chans.all())
        self.assertEqual(result_samples.all(), expect_samples.all())

    def test_filter_samples(self):
        """
            When filtering samples, all time samples
            with a relative high power should be removed
        """
        result = clipping.filter_samples(self.fil_vector)
        expect = self.fil_vector.copy()
        expect = np.delete(expect, self.bad_samples)
        self.assertEqual(result.all(), expect.all())

    def test_filter_channels(self):
        """
            When filtering samples, all channels
            with a relative high power should be removed
        """
        result_channels = np.array(clipping.filter_channels(np.array(self.fil_vector)))
        expect_channels = np.delete(self.fil_chans, self.bad_channels)
        self.assertEqual(result_channels.all(), expect_channels.all())

    def test_filter_indv_channels(self):
        """
            When filtering samples, all samples
            with a relative high power should be removed
        """
        result = clipping.filter_indv_channels(np.array(self.fil_vector))
        new_vector = np.full((len(self.fil_vector), len(self.fil_vector[0])), 1)
        new_column = [5] * len(self.fil_vector)
        new_vector[:, 5] = new_column
        self.assertEqual(result.all(), new_vector.all())

if __name__ == '__main__':
    unittest.main()
