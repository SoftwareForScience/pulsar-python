import unittest
import numpy as np

from context import fourier

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
        x = np.random.random(32) + 1j*np.random.random(32)
        x = np.asarray(x)
        self.assertAlmostEqual(fft1(x).all(), fourier.DFT_slow(x).all())


    def test_fft(self):
        x = np.random.random(32) + 1j*np.random.random(32)
        x = np.asarray(x)
        self.assertAlmostEqual(fft1(x).all(), fourier.FFT_vectorized(x).all())


    def test_wrong_array_size(self):
        x = np.random.random(30) + 1j*np.random.random(30)
        with self.assertRaises(ValueError):
            fourier.FFT_vectorized(x)

    def test_fftfreq(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        x = np.asarray(x)
        self.assertAlmostEqual(9*fourier.fftfreq(9).all(), x.all())
        self.assertAlmostEqual(9*np.pi*fourier.fftfreq(9, np.pi).all(), x.all())
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        x = np.asarray(x)
        self.assertAlmostEqual(10*fourier.fftfreq(10).all(), x.all())
        self.assertAlmostEqual(10*np.pi*fourier.fftfreq(10, np.pi).all(), x.all())


if __name__ == '__main__':
    unittest.main()
