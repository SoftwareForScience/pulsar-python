"""
    fourier.py test
"""

import unittest
import numpy as np

from context import fourier

def fft1(test_input):
    """
        A simple fft implementation. 
        This is the same function with which numpy tests their FFT funciton.
    """
    L = len(test_input)
    phase = -2j*np.pi*(np.arange(L)/float(L))
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(test_input*np.exp(phase), axis=1)

class TestFft(unittest.TestCase):
    """
        Testclass for testing fourier transforms
    """

    def test_dft(self):
        """
            Test for DFT function
        """
        test_input = np.random.random(32) + 1j*np.random.random(32)
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(fft1(test_input).all(), fourier.DFT_slow(test_input).all())


    def test_fft(self):
        """
            Test for FFT_vectorized function
        """
        test_input = np.random.random(32) + 1j*np.random.random(32)
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(fft1(test_input).all(), fourier.FFT_vectorized(test_input).all())


    def test_wrong_array_size(self):
        """
            Tests for correct input for the FFT_vectorized function. 
            Fails if the input is not a power of 2.
        """
        test_input = np.random.random(30) + 1j*np.random.random(30)
        with self.assertRaises(ValueError):
            fourier.FFT_vectorized(test_input)


    def test_fftfreq(self):
        """
            Test for fftfreq function
        """
        test_input = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(9*fourier.fftfreq(9).all(), test_input.all())
        self.assertAlmostEqual(9*np.pi*fourier.fftfreq(9, np.pi).all(), test_input.all())
        test_input = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(10*fourier.fftfreq(10).all(), test_input.all())
        self.assertAlmostEqual(10*np.pi*fourier.fftfreq(10, np.pi).all(), test_input.all())


if __name__ == '__main__':
    unittest.main()
