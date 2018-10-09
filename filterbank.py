"""Object which enables reading of filterbank files. """

import os
import numpy as np
import matplotlib.pyplot as plt

class Filterbank():
    """
        Processing .fil files
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable-msg=too-many-arguments
    # pylint: disable-msg=too-many-locals

    def __init__(self, filename, f_start, f_stop,
                 t_start, t_stop):
        """
            Initialize Filterbank object
        """

        self.f_0 = 433.698
        self.f_delt = -0.062

        self.t_samp = 8e-05

        # initialize cause pylint
        self.freqs = 10
        self.timestamps = float(10)

        self.read_filterbank(filename, f_start, f_stop, t_start, t_stop)


    def read_filterbank(self, filename, f_start, f_stop, t_start, t_stop):
        """
            Reading filterbank files to 2d numpy array (?)
        """

        # read header, not added yet

        self.n_chans = 128

        i_start, i_stop, chan_start_idx, chan_stop_idx = self.setup_freqs(f_start, f_stop)

        print(i_start)
        print(i_stop)

        n_bits = 8
        n_bytes = int(n_bits / 8)
        n_chans_selected = self.freqs.shape[0]
        n_ifs = 1

        idx_data = 128

        fil = open(filename, 'rb')
        fil.seek(idx_data)

        filesize = os.path.getsize(filename)
        n_bytes_data = filesize - idx_data

        self.n_ints_data = n_bytes_data / (n_bytes * self.n_chans * n_ifs)

        ii_start, ii_stop, n_ints = self.setup_time(t_start, t_stop)

        print(ii_stop)

        fil.seek(int(ii_start * n_bits * n_ifs * self.n_chans / 8), 1)

        i_0 = np.min((chan_start_idx, chan_stop_idx))
        i_1 = np.max((chan_start_idx, chan_stop_idx))

        if n_bits == 2:
            dd_type = b'unint8'
            n_chans_selected = int(n_chans_selected/4)
        elif n_bytes == 4:
            dd_type = b'float32'
        elif n_bytes == 2:
            dd_type = b'uint16'
        elif n_bytes == 1:
            dd_type = b'uint8'

        if n_bits == 2:
            self.data = np.zeros((n_ints, n_ifs, n_chans_selected * 4), dtype=dd_type)
        else:
            self.data = np.zeros((n_ints, n_ifs, n_chans_selected), dtype=dd_type)

        for i_i in range(n_ints):
            for j_j in range(n_ifs):
                fil.seek(n_bytes * i_0, 1)

                d_d = np.fromfile(fil, count=n_chans_selected, dtype=dd_type)

                if n_bits == 2:
                    d_d = unpack_2to8(d_d)

                self.data[i_i, j_j] = d_d

                fil.seek(n_bytes * (self.n_chans - i_1), 1)

        print(d_d)

    def setup_freqs(self, f_start, f_stop):
        """
            Calculate the frequency axis
        """

        i_start, i_stop = 0, self.n_chans

        if f_start:
            i_start = int((f_start - self.f_0) / self.f_delt)
        if f_stop:
            i_stop = int((f_stop - self.f_0) / self.f_delt)

        chan_start_idx = np.int(i_start)
        chan_stop_idx = np.int(i_stop)

        if i_start < i_stop:
            i_vals = np.arange(chan_start_idx, chan_stop_idx)
        else:
            i_vals = np.arange(chan_stop_idx, chan_start_idx)

        self.freqs = self.f_delt * i_vals + self.f_0

        if chan_stop_idx < chan_start_idx:
            chan_stop_idx, chan_start_idx = chan_start_idx, chan_stop_idx

        return i_start, i_stop, chan_start_idx, chan_stop_idx

    def setup_time(self, t_start, t_stop):
        """
            Calculate the time axis
        """

        ii_start, ii_stop = 0, self.n_ints_data

        if t_start:
            ii_start = t_start
        if t_stop:
            ii_stop = t_stop

        n_ints = ii_stop - ii_start

        self.timestamps = np.arange(0, n_ints) * self.t_samp / 24./60./60. + t_start

        return ii_start, ii_stop, n_ints

    def get_data(self, f_start, f_stop, t_start, t_stop):
        """
            Extract a piece of data from the filterbank file
        """

        i_0 = np.argmin(np.abs(self.freqs - f_start))
        i_1 = np.argmin(np.abs(self.freqs - f_stop))

        if i_0 < i_1:
            freq_data = self.freqs[i_0:i_1 + 1]
            fil_data = np.squeeze(self.data[t_start:t_stop, ..., i_0:i_1 + 1])
        else:
            freq_data = self.freqs[i_1:i_0 + 1]
            fil_data = np.squeeze(self.data[t_start:t_stop, ..., i_1:i_0 + 1])

        return freq_data, fil_data

    def plot_data(self):
        """
            Method for plotting the data and its shape
        """
        print(self.data.shape)
        plt.plot(self.data[1])
        plt.show()


def unpack_2to8(data):
    """
        Unpack bits to bytes
    """
    tmp = data.astype(np.uint32)
    tmp = (tmp | (tmp << 12)) & 0xF000F
    tmp = (tmp | (tmp << 6))  & 0x3030303
    tmp = tmp.byteswap()
    return tmp.view('uint8')
