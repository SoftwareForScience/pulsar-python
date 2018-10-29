"""
    plot.py unit tests
"""
import unittest
import numpy as np

from context import plot # pylint: disable-msg=E0611

class TestWindowHanning(unittest.TestCase):
    """
        Testclass for testing the plotting functions
    """
    np.random.seed(0)
    n = 1000

    sig_rand = np.random.standard_normal(n) + 100.
    sig_ones = np.ones(n)

    Fs = 100

    def test_window_hanning_rand(self):
        targ = np.hanning(len(self.sig_rand)) * self.sig_rand
        res = plot.window_hanning(self.sig_rand)

        self.assertAlmostEqual(targ.all(), res.all())


    def test_window_hanning_ones(self):
        targ = np.hanning(len(self.sig_ones))
        res = plot.window_hanning(self.sig_ones)

        self.assertAlmostEqual(targ.all(), res.all())
        
if __name__ == '__main__':
    unittest.main()
