"""
    Pipeline for all modules
"""
import os
import sys
import inspect
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
import filterbank.filterbank as filterbank
import clipping
import dedisperse
import timeseries.timeseries as timeseries
import fourier
import time

class Pipeline():
    """
        The Pipeline combines the functionality of all modules
        in the library.
    """

    def __init__(self, as_stream=False, filename=None, DM=230, scale=3):
        """
            Initialize Pipeline object

            Args:
                as_stream, read the filterbank data as stream
        """
        if as_stream:
            self.read_stream(filename, DM, scale)
        else:
            self.read_static(filename, DM, scale)


    def read_stream(self, filename, DM, scale):
        """
            Read the filterbank data as stream
        """


    
    def read_static(self, filename, DM, scale):
        """
            Read the filterbank data at once
        """
        stopwatch = dict.fromkeys(['time_read', 'time_select', 'time_clipping', 'time_dedisp', 
                                   'time_t_series', 'time_downsample', 'time_fft_vect', 'time_dft'
                                   'time_ifft', 'time_fft_freq'])
        time_start = time.time()
        # init filterbank
        fil = filterbank.Filterbank(filename, read_all=True)
        stopwatch['time_read'] = time.time() - time_start
        # select data
        freqs, fil_data = fil.select_data()
        stopwatch['time_select'] = time.time() - stopwatch['time_read']
        # clipping
        _, _ = clipping.clipping(freqs, fil_data)
        stopwatch['time_clipping'] = time.time() - stopwatch['time_select']
        # dedisperse
        fil_data = dedisperse.dedisperse(fil_data, DM)
        stopwatch['time_dedisp'] = time.time() - stopwatch['time_clipping']
        # timeseries
        time_series = timeseries.Timeseries(fil_data)
        stopwatch['time_t_series'] = time.time() - stopwatch['time_dedisp']
        # downsample
        time_series = time_series.downsample(scale)
        stopwatch['time_downsample'] = time.time()- stopwatch['time_t_series']
        # fft vect
        fourier.fft_vectorized(time_series)
        stopwatch['time_fft_vect'] = time.time() - stopwatch['time_downsample']
        # dft
        fourier.dft_slow(time_series)
        stopwatch['time_dft'] = time.time() - stopwatch['time_fft_vect']
        # ifft
        fourier.ifft(time_series)
        stopwatch['time_ifft'] = time.time() - stopwatch['time_dft']
        # fft freq
        fourier.fft_freq(10)
        stopwatch['time_fft_freq'] = time.time() - stopwatch['time_ifft']
        print(stopwatch)

