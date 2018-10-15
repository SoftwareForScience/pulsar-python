import unittest
import numpy as np

from .context import fourier

def fft1(x):
    L = len(x)
    phase = -2j*np.pi*(np.arange(L)/float(L))
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x*np.exp(phase), axis=1)

class TestFft(unittest.TestCase):
    """
        Testclass for testing fourier transforms
    """

    def test_dft(self):
        x = np.random.random(30) + 1j*np.random.random(30)
        self.assertAlmostEqual(fft1(x), fourier.DFT_slow(x))

    def test_fft(self):
        x = np.random.random(30) + 1j*np.random.random(30)
        self.assertAlmostEqual(fft1(x), fourier.FFT_vectorized(x))

if __name__ == '__main__':
    unittest.main()
