"""
    benchmarking all functions, except for plots
"""

import time
import unittest

from .context import clipping, dedisperse, fourier, filterbank, header


class TestBenchmark(unittest.TestCase):
    """
        Testclass for benchmarking all methods
    """

    functions = clipping.clipping, dedisperse.dedisperse, fourier.dft_slow, fourier.fft_freq, fourier.fft_matrix, fourier.fft_shift, fourier.fft_vectorized, filterbank.Filterbank, filterbank.read_header

    times = {f.__name__: [] for  f in functions}

    # def test_benchmark(self):
    #     for func in enumerate(self.functions):
    #         t0 = time.time()
    #         # print(func.__name__)
    #         t1 = time.time()
    #         # print(t1-t0)
    #         # self.times[func.__name__].append((t1-t0) * 1000)


    def test_benchmark2(self):
        for _, func in enumerate(self.functions):
            t0 = time.time()
            func()
            t1 = time.time()
            self.times[func.__name__].append((t1 - t0) * 1000)


if __name__ == '__main__':
    unittest.main()

