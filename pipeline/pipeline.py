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
import datetime
import timeit

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
        stopwatch = dict.fromkeys(['time_read', 'time_select', 'time_clipping', 'time_dedisp', 
                                   'time_t_series', 'time_downsample', 'time_fft_vect', 'time_dft'
                                   'time_ifft', 'time_fft_freq'])
        # init filterbank as stream
        fil = filterbank.Filterbank(filename)

        fil.select_data()
        

    
    def read_static(self, filename, DM, scale):
        """
            Read the filterbank data at once
        """
        stopwatch = dict.fromkeys(['time_read', 'time_select', 'time_clipping', 'time_dedisp', 
                                   'time_t_series', 'time_downsample', 'time_fft_vect', 'time_dft'
                                   'time_ifft', 'time_fft_freq'])
        time_start = datetime.datetime.now()

        a = datetime.datetime.now()
        b = datetime.datetime.now()

        time_took = b - a

        print(time_took.seconds)
        # init filterbank
        fil = filterbank.Filterbank(filename, read_all=True)
        stopwatch['time_read'] = time_start - datetime.datetime.now()
        # select data
        freqs, fil_data = fil.select_data()
        time_select = datetime.datetime.now() - stopwatch['time_read']
        stopwatch['time_read'] = time_select.microsecond
        # clipping
        _, _ = clipping.clipping(freqs, fil_data)
        time_clipping = datetime.datetime.now() - time_select
        stopwatch['time_clipping'] = time_clipping.microseconds
        # stopwatch['time_clipping'] = stopwatch['time_clipping'].microseconds
        # dedisperse
        fil_data = dedisperse.dedisperse(fil_data, DM)
        time_dedisp = datetime.datetime.now() - time_clipping
        stopwatch['time_dedisp'] = time_dedisp.microseconds
        # timeseries
        time_series = timeseries.Timeseries(fil_data)
        time_t_series = datetime.datetime.now() - time_dedisp
        stopwatch['time_t_series'] = time_t_series.microseconds
        # downsample
        time_series = time_series.downsample(scale)
        time_downsample = datetime.datetime.now() - time_t_series
        stopwatch['time_downsample'] = time_downsample.microseconds
        # fft vect
        fourier.fft_vectorized(time_series)
        time_fft_vect = datetime.datetime.now() - time_downsample
        stopwatch['time_fft_vect'] = time_fft_vect.microseconds
        # dft
        fourier.dft_slow(time_series)
        time_dft = datetime.datetime.now() - time_fft_vect
        stopwatch['time_dft'] = time_dft.microseconds
        # ifft
        fourier.ifft(time_series)
        time_ifft = datetime.datetime.now() - time_dft
        stopwatch['time_ifft'] = time_ifft.microseconds
        # fft freq
        fourier.fft_freq(10)
        time_fft_freq = datetime.datetime.now() - time_ifft
        stopwatch['time_fft_freq'] = time_fft_freq.microseconds
        print(stopwatch)

