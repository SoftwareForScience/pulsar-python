"""
    fourier.py test
"""

import unittest
import numpy as np

from .context import fourier # pylint: disable-msg=E0611

def fft_simple(test_input):
    """
        A simple fft implementation.
        This is the same function with which numpy tests their FFT function.
    """
    input_length = len(test_input)
    phase = -2j * np.pi * (np.arange(input_length) / float(input_length))
    phase = np.arange(input_length).reshape(-1, 1) * phase
    return np.sum(test_input * np.exp(phase), axis=1)

class TestFft(unittest.TestCase):
    """
        Testclass for testing fourier transforms
    """

    def test_dft(self):
        """
            Test for DFT function
        """
        test_input = np.random.random(32) + 1j * np.random.random(32)
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(fft_simple(test_input).all(), fourier.dft_slow(test_input).all())


    def test_fft(self):
        """
            Test for FFT_vectorized function
        """
        test_input = np.random.random(32) + 1j * np.random.random(32)
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(fft_simple(test_input).all(),
                               fourier.fft_vectorized(test_input).all())


    def test_fft_buildup(self):
        """
            Test for FFT_vectorized function with recursive build-up
        """
        test_input = np.random.random(1024)
        self.assertAlmostEqual(fft_simple(test_input).all(),
                               fourier.fft_vectorized(test_input).all())


    def test_fft_wrong_array_size(self):
        """
            Tests for correct input for the FFT_vectorized function.
            Fails if the input is not a power of 2.
        """
        test_input = np.random.random(30) + 1j * np.random.random(30)
        with self.assertRaises(ValueError):
            fourier.fft_vectorized(test_input)


    def test_ifft_wrong_array_size(self):
        """
            Tests for correct input for the ifft function.
            Fails if the input is not a power of 2.
        """
        test_input = np.random.random(30) + 1j * np.random.random(30)
        with self.assertRaises(ValueError):
            fourier.ifft(test_input)


    def test_fft_smaller_shape(self):
        """
            Test for FFT_vectorized function
            using a smaller nfft
        """
        test_input = np.random.random(32) + 1j * np.random.random(32)
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(fft_simple(test_input).all(),
                               fourier.fft_vectorized(test_input, 1).all())


    def test_fft_larger_shape(self):
        """
            Test for FFT_vectorized function
            using a larger nfft
        """
        test_input = np.random.random(32) + 1j * np.random.random(32)
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(fft_simple(test_input).all(),
                               fourier.fft_vectorized(test_input, 256).all())


    def test_fft_freq(self):
        """
            Test for fftfreq function
        """
        test_input = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(9 * fourier.fft_freq(9).all(), test_input.all())
        self.assertAlmostEqual(9 * np.pi*fourier.fft_freq(9, np.pi).all(), test_input.all())
        test_input = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        test_input = np.asarray(test_input)
        self.assertAlmostEqual(10 * fourier.fft_freq(10).all(), test_input.all())
        self.assertAlmostEqual(10 * np.pi*fourier.fft_freq(10, np.pi).all(), test_input.all())


    def test_fft_matrix(self):
        """
            Test for fft matrix function
        """
        test_input = np.random.random(32) + 1j * np.random.random(32)
        test_input = np.asarray([test_input, test_input])
        self.assertAlmostEqual(fft_simple(test_input[0]).all(),
                               fourier.fft_matrix(test_input)[0].all())
        self.assertAlmostEqual(fft_simple(test_input[1]).all(),
                               fourier.fft_matrix(test_input)[1].all())


    def test_fft_shift_no_axis(self):
        """
            Test for fftshift function
            without the axes parameter
        """
        test_input = range(1, 10)
        test_output = [6, 7, 8, 9, 1, 2, 3, 4, 5]
        self.assertSequenceEqual(fourier.fft_shift(test_input).tolist(), test_output)


    def test_fft_shift_one_axes(self):
        """
            Test for fftshift function
            with one axes as parameter
        """
        test_input = range(1, 10)
        test_output = [6, 7, 8, 9, 1, 2, 3, 4, 5]
        self.assertSequenceEqual(fourier.fft_shift(test_input, 0).tolist(), test_output)


    def test_fft_shift_multi_axes(self):
        """
            Test for fftshift function
            with multiple axes as parameter
        """
        test_input = range(1, 10)
        test_output = [2, 3, 4, 5, 6, 7, 8, 9, 1]
        self.assertSequenceEqual(fourier.fft_shift(test_input, [0, 0]).tolist(), test_output)


    def test_fft_and_ifft(self):
        """
            Test the fft and ifft
            and expect the same outcome and input
        """
        test_input = np.random.random(32) + 1j * np.random.random(32)
        test_input = np.asarray(test_input)
        self.assertEqual(test_input.all(), fourier.fft_vectorized(fourier.ifft(test_input)).all())


if __name__ == '__main__':
    unittest.main()
