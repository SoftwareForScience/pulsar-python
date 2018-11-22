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
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        # iterator for stream
        self.stream_iter = 0
        self.data, self.freqs, self.n_chans_selected = None, None, None
        self.filename = filename
        self.header = read_header(filename)
        self.idx_data = len_header(filename)
        self.n_bytes = int(self.header[b'nbits'] / 8)
        self.n_chans = self.header[b'nchans']
        self.n_ifs = self.header[b'nifs']
        # decide appropriate datatype
        if self.n_bytes == 4:
            self.dd_type = b'float32'
        elif self.n_bytes == 2:
            self.dd_type = b'uint16'
        elif self.n_bytes == 1:
            self.dd_type = b'uint8'
        # open filterbank file
        self.fil = open(self.filename, 'rb')
        # skip the header bytes
        self.fil.seek(self.idx_data)
        # find possible time range
        self.ii_start, self.n_samples = self.setup_time(time_range)
        # search for start of data
        self.fil.seek(int(self.ii_start * self.n_bytes * self.n_ifs * self.n_chans), 1)
        # find possible channels
        self.i_0, self.i_1 = self.setup_chans(freq_range)


    def read_filterbank(self):
        """
            Read filterbank file and transform to tuple of 3 matrices:
            including the sample amount, number of intermediate channels
            and the amount of selected frequencies/channels
        """
        # set shape of data
        self.data = np.zeros((self.n_samples, self.n_ifs, self.n_chans_selected),
                             dtype=self.dd_type)
        # read for each time sample the intensity per frequency
        for i_i in range(self.n_samples):
            for j_j in range(self.n_ifs):
                self.fil.seek(self.n_bytes * self.i_0, 1)
                # add to matrix
                self.data[i_i, j_j] = np.fromfile(self.fil, count=self.n_chans_selected,
                                                  dtype=self.dd_type)
                # skip bytes till start of next chunk
                self.fil.seek(self.n_bytes * (self.n_chans - self.i_1), 1)
        # release file resources
        self.fil.close()


    def next_row(self):
        """
            Read filterbank file per row

            returns True if EOF
        """
        if self.stream_iter < (self.n_samples * self.n_ifs):
            self.stream_iter += 1
            # skip bytes
            self.fil.seek(self.n_bytes * self.i_0, 1)
            # read row of data
            data = np.fromfile(self.fil, count=self.n_chans_selected, dtype=self.dd_type)
            # skip bytes till start of next chunk
            self.fil.seek(self.n_bytes * (self.n_chans - self.i_1), 1)
        else:
            data = True
            self.fil.close()
        return data


    def next_n_rows(self, n_rows):
        """
            Read filterbank per n rows

            returns True if EOF
        """
        if self.stream_iter < (self.n_samples * self.n_ifs):
            # more rows requested than available
            if self.stream_iter + n_rows >= self.n_samples * self.n_ifs:
                n_rows = self.n_samples * self.n_ifs - self.stream_iter
            self.stream_iter += n_rows
            # init array of n rows
            data = np.zeros((n_rows, self.n_chans_selected), dtype=self.dd_type)
            for row in range(n_rows):
                # skip bytes
                self.fil.seek(self.n_bytes * self.i_0, 1)
                # read row of data
                data[row] = np.fromfile(self.fil, count=self.n_chans_selected, dtype=self.dd_type)
                # skip bytes till start of next chunk
                self.fil.seek(self.n_bytes * (self.n_chans - self.i_1), 1)
        else:
            data = True
            self.fil.close()
        
        print(data)

        return data


    def setup_freqs(self, freq_range=None):
        """
            Calculate the frequency range
        """
        f_delt = self.header[b'foff']
        f_0 = self.header[b'fch1']
        i_start, i_stop = 0, self.n_chans
        # frequency range is specified
        if freq_range:
            if freq_range[0]:
                i_start = int((freq_range[0] - f_0) / f_delt)
            if freq_range[1]:
                i_stop = int((freq_range[1] - f_0) / f_delt)
        chan_start_idx = np.int(i_start)
        chan_stop_idx = np.int(i_stop)
        # create evenly spaced interval for frequencies
        if i_start < i_stop:
            i_vals = np.arange(chan_start_idx, chan_stop_idx)
        else:
            i_vals = np.arange(chan_stop_idx, chan_start_idx)
        # calculate all possible frequencies
        self.freqs = f_delt * i_vals + f_0
        # amount of channels
        self.n_chans_selected = self.freqs.shape[0]
        # change channel order if reversed
        if chan_stop_idx < chan_start_idx:
            chan_stop_idx, chan_start_idx = chan_start_idx, chan_stop_idx
        return chan_start_idx, chan_stop_idx


    def setup_time(self, time_range=None):
        """
            Calculate the time range
        """
        t_delt = self.header[b'tsamp']
        t_0 = self.header[b'tstart']
        # calculate amount of bytes in file without header
        n_bytes_data = os.path.getsize(self.filename) - self.idx_data
        # calculate sample size
        ii_start, ii_stop = 0, int(n_bytes_data / (self.n_bytes * self.n_chans * self.n_ifs))
        # time range is specified
        if time_range:
            if time_range[0]:
                ii_start = time_range[0]
            if time_range[1]:
                ii_stop = time_range[1]
        n_samples = ii_stop - ii_start
        # calculate all possible times
        self.timestamps = np.arange(0, n_samples) * t_delt / 24. / 60. / 60. + t_0
        return ii_start, n_samples


    def setup_chans(self, freq_range=None):
        """
            Calculate the channel range
        """
        chan_start_idx, chan_stop_idx = self.setup_freqs(freq_range)
        # set lowest value
        i_0 = np.min((chan_start_idx, chan_stop_idx))
        # set highest value
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
        # give index of minimum value
        i_0 = np.argmin(np.abs(self.freqs - freq_start))
        i_1 = np.argmin(np.abs(self.freqs - freq_stop))
        # reverse data if frequencies are reversed
        if i_0 < i_1:
            freq_data = self.freqs[i_0:i_1 + 1]
            fil_data = np.squeeze(self.data[time_start:time_stop, ..., i_0:i_1 + 1])
        else:
            freq_data = self.freqs[i_1:i_0 + 1]
            fil_data = np.squeeze(self.data[time_start:time_stop, ..., i_1:i_0 + 1])
        return freq_data, fil_data
