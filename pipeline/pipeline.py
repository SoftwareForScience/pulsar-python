"""
    Pipeline for running all the modules in order
"""
# pylint: disable=wrong-import-position
import os
import sys
import inspect

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
from timeit import default_timer as timer
import filterbank.filterbank
import timeseries.timeseries
import clipping
import dedisperse
import fourier


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=no-self-use

class Pipeline:
    """
        The Pipeline combines the functionality of all modules
        in the library.
    """

    def __init__(self, filename=None, as_stream=False, DM=230, scale=3, n=None, size=None):
        """
            Initialize Pipeline object

            Args:
                as_stream, read the filterbank data as stream
        """
        if as_stream:
            if n:
                result = self.read_n_rows(n, filename, DM, scale)
                file = open("n_rows_filterbank.txt", "a+")
            else:
                result = self.read_rows(filename)
                file = open("rows_filterbank.txt", "a+")
        else:
            result = self.read_static(filename, DM, scale, size)
            file = open("static_filterbank.txt", "a+")
        file.write(str(result) + ",")
        file.close()

    def read_rows(self, filename):
        """
            Read the filterbank data as stream
            and measure the time
        """
        # init filterbank as stream
        fil = filterbank.Filterbank(filename)
        time_start = timer()
        while True:
            fil_data = fil.next_row()
            if isinstance(fil_data, bool):
                break
        time_stop = timer() - time_start
        return time_stop

    def read_n_rows(self, n, filename, DM, scale):
        """
            Read the filterbank data as stream
            and measure the time
        """
        fil = filterbank.Filterbank(filename)
        stopwatch_list = list()
        while True:
            stopwatch = dict.fromkeys(['time_read', 'time_select', 'time_clipping', 'time_dedisp',
                                       'time_t_series', 'time_downsample', 'time_fft_vect',
                                       'time_dft', 'time_ifft', 'time_fft_freq'])
            time_start = timer()
            fil_data = fil.next_n_rows(n)
            # break if EOF
            if isinstance(fil_data, bool):
                break
            stopwatch['time_read'] = timer() - time_start
            # run methods
            stopwatch = self.measure_methods(stopwatch, fil_data, fil.freqs, DM, scale)
            stopwatch_list.append(stopwatch)
        return stopwatch_list

    def read_static(self, filename, DM, scale, size):
        """
            Read the filterbank data at once
            and measure the time per function/class
        """
        stopwatch = dict.fromkeys(['time_read', 'time_select', 'time_clipping', 'time_dedisp',
                                   'time_t_series', 'time_downsample', 'time_fft_vect', 'time_dft',
                                   'time_ifft', 'time_fft_freq'])
        time_start = timer()
        # init filterbank
        fil = filterbank.Filterbank(filename, read_all=True, time_range=(0, size))
        stopwatch['time_read'] = timer() - time_start
        # select data
        time_select = timer()
        freqs, fil_data = fil.select_data()
        stopwatch['time_select'] = timer() - time_select
        # run methods
        stopwatch = self.measure_methods(stopwatch, fil_data, freqs, DM, scale)
        return stopwatch

    def measure_methods(self, stopwatch, fil_data, freqs, DM, scale):
        """
            Run and time all methods/modules
        """
        # clipping

        time_clipping = timer()
        _, _ = clipping.clipping(freqs, fil_data)
        stopwatch['time_clipping'] = timer() - time_clipping
        # dedisperse
        time_dedisp = timer()
        fil_data = dedisperse.dedisperse(fil_data, disperion_measure=DM)
        stopwatch['time_dedisp'] = timer() - time_dedisp
        # timeseries
        time_t_series = timer()
        time_series = timeseries.Timeseries(fil_data)
        stopwatch['time_t_series'] = timer() - time_t_series
        # downsample
        time_downsamp = timer()
        time_series = time_series.downsample(scale)
        stopwatch['time_downsample'] = timer() - time_downsamp
        # fft vect
        time_fft_vect = timer()
        fourier.fft_vectorized(time_series)
        stopwatch['time_fft_vect'] = timer() - time_fft_vect
        # dft
        time_dft = timer()
        fourier.dft_slow(time_series)
        stopwatch['time_dft'] = timer() - time_dft
        # ifft
        time_ifft = timer()
        fourier.ifft(time_series)
        stopwatch['time_ifft'] = timer() - time_ifft
        # fft freq
        time_fft_freq = timer()
        fourier.fft_freq(10)
        stopwatch['time_fft_freq'] = timer() - time_fft_freq
        return stopwatch
