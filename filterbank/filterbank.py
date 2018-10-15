"""
    Utilities for reading data from filterbank file
"""

import os
import numpy as np
from filterbank.header import read_header, len_header


class Filterbank:
    """
        Processing .fil files
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, filename, freq_range=None, time_range=None):
        """
            Initialize Filterbank object

            Args:
                freq_range, tuple of freq_start and freq_stop in MHz
                time_range, tuple of time_start and time_stop
        """
        self.freqs = None
        self.timestamps = None
        if os.path.isfile(filename):
            self.filename = filename
            self.header = read_header(filename)
            self.idx_data = len_header(filename)
            self.n_bytes = int(self.header[b'nbits'] / 8)
            self.n_chans = self.header[b'nchans']
            self.n_ifs = self.header[b'nifs']
            self.read_filterbank(freq_range, time_range)
        else:
            raise FileNotFoundError(filename)

    def read_filterbank(self, freq_range=None, time_range=None):
        """
            Read filterbank file to 3d numpy array
        """
        fil = open(self.filename, 'rb')
        fil.seek(self.idx_data)

        ii_start, n_ints = self.setup_time(time_range)
        # search for start of data
        fil.seek(int(ii_start * self.n_bytes * self.n_ifs * self.n_chans), 1)

        i_0, i_1 = self.setup_chans(freq_range)

        n_chans_selected = self.freqs.shape[0]

        # decide appropriate datatype
        if self.n_bytes == 4:
            dd_type = b'float32'
        elif self.n_bytes == 2:
            dd_type = b'uint16'
        elif self.n_bytes == 1:
            dd_type = b'uint8'
        # create numpy array with 3d shape
        self.data = np.zeros((n_ints, self.n_ifs, n_chans_selected), dtype=dd_type)

        for i_i in range(n_ints):
            for j_j in range(self.n_ifs):
                fil.seek(self.n_bytes * i_0, 1)
                # add to matrix
                self.data[i_i, j_j] = np.fromfile(fil, count=n_chans_selected, dtype=dd_type)
                # search for start of next chunk
                fil.seek(self.n_bytes * (self.n_chans - i_1), 1)

    def setup_freqs(self, freq_range=None):
        """
            Calculate the frequency range
        """
        f_delt = self.header[b'foff']
        f_0 = self.header[b'fch1']

        i_start, i_stop = 0, self.n_chans

        if freq_range:
            if freq_range[0]:
                i_start = int((freq_range[0] - f_0) / f_delt)
            if freq_range[1]:
                i_stop = int((freq_range[1] - f_0) / f_delt)

        chan_start_idx = np.int(i_start)
        chan_stop_idx = np.int(i_stop)

        if i_start < i_stop:
            i_vals = np.arange(chan_start_idx, chan_stop_idx)
        else:
            i_vals = np.arange(chan_stop_idx, chan_start_idx)

        self.freqs = f_delt * i_vals + f_0

        if chan_stop_idx < chan_start_idx:
            chan_stop_idx, chan_start_idx = chan_start_idx, chan_stop_idx

        return chan_start_idx, chan_stop_idx

    def setup_time(self, time_range=None):
        """
            Calculate the time range
        """
        t_delt = self.header[b'tsamp']
        t_0 = self.header[b'tstart']

        n_bytes_data = os.path.getsize(self.filename) - self.idx_data
        n_ints_data = int(n_bytes_data / (self.n_bytes * self.n_chans * self.n_ifs))

        ii_start, ii_stop = 0, n_ints_data

        if time_range:
            if time_range[0]:
                ii_start = time_range[0]
            if time_range[1]:
                ii_stop = time_range[1]

        n_ints = ii_stop - ii_start

        self.timestamps = np.arange(0, n_ints) * t_delt / 24. / 60. / 60. + t_0

        return ii_start, n_ints

    def setup_chans(self, freq_range=None):
        """
            Calculate the channel range
        """
        chan_start_idx, chan_stop_idx = self.setup_freqs(freq_range)

        i_0 = np.min((chan_start_idx, chan_stop_idx))
        i_1 = np.max((chan_start_idx, chan_stop_idx))

        return i_0, i_1

    def select_data(self, freq_start=None, freq_stop=None, time_start=None, time_stop=None):
        """
            Select a range of data from the filterbank file
        """
        # if no frequency range is specified, select all frequencies
        if freq_start is None:
            freq_start = self.freqs[0]
        if freq_stop is None:
            freq_stop = self.freqs[-1]

        i_0 = np.argmin(np.abs(self.freqs - freq_start))
        i_1 = np.argmin(np.abs(self.freqs - freq_stop))

        if i_0 < i_1:
            freq_data = self.freqs[i_0:i_1 + 1]
            fil_data = np.squeeze(self.data[time_start:time_stop, ..., i_0:i_1 + 1])
        else:
            freq_data = self.freqs[i_1:i_0 + 1]
            fil_data = np.squeeze(self.data[time_start:time_stop, ..., i_1:i_0 + 1])

        return freq_data, fil_data
