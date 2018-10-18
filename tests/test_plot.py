"""
    plot.py unit tests
"""
import unittest
import numpy as np

from context import plot # pylint: disable-msg=E0611

class TestPlot(unittest.TestCase):
    """
        Testclass for testing the plotting functions
    """

    def __init__(self):
        np.random.seed(0)
        n = 1000

        self.sig_rand = np.random.standard_normal(n) + 100.
        self.sig_ones = np.ones(n)

    def test_window_hanning_rand(self):
        targ = np.hanning(len(self.sig_rand)) * self.sig_rand
        res = plot.window_hanning(self.sig_rand)

        self.assertAlmostEquals(targ.all(), res.all())

    def test_window_hanning_ones(self):
        targ = np.hanning(len(self.sig_ones))
        res = plot.window_hanning(self.sig_ones)

        self.assertAlmostEquals(targ.all(), res.all(), atol=1e-06)

