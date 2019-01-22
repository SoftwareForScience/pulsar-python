''' Module for creating filterbank files with a fake signal
'''
import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from filterbank.header import read_header
from filterbank.filterbank import Filterbank

HEADER_KEYWORD_TYPES = {
    b'telescope_id': b'<l',
    b'machine_id': b'<l',
    b'data_type': b'<l',
    b'barycentric': b'<l',
    b'pulsarcentric': b'<l',
    b'nbits': b'<l',
    b'nsamples': b'<l',
    b'nchans': b'<l',
    b'nifs': b'<l',
    b'nbeams': b'<l',
    b'ibeam': b'<l',
    b'rawdatafile': b'str',
    b'source_name': b'str',
    b'az_start': b'<d',
    b'za_start': b'<d',
    b'tstart': b'<d',
    b'tsamp': b'<d',
    b'fch1': b'<d',
    b'foff': b'<d',
    b'refdm': b'<d',
    b'period': b'<d',
    b'src_raj': b'angle',
    b'src_dej': b'angle',
}

# pylint: disable=E1121

def write_data(n_bits, new_file, data, header):
    '''
        Write data to a filterbank file
    '''
    n_bytes = int(n_bits/8)
    with open(new_file, 'wb') as new_file:
        new_file.write(header)
        if n_bytes == 1:
            np.int8(data.ravel()).tofile(new_file)
        elif n_bytes == 2:
            np.int16(data.ravel()).tofile(new_file)
        elif n_bytes == 4:
            np.float32(data.ravel()).tofile(new_file)

def generate_header_string(header):
    '''
        Generate the header for the filterbank file
    '''
    header_string = b''
    header_string += to_keyword(b'HEADER_START')
    # add header dictionary keys and values to string
    for keyword in header.keys():
        header_string += to_keyword(keyword, header[keyword])
    
    header_string += to_keyword(b'HEADER_END')
    return header_string


def to_keyword(keyword, value=None):
    ''' Transform keyword to a serialied string
    '''
    keyword = bytes(keyword)
    # value of attribute has not been specified
    if not value:
        return np.int32(len(keyword)).tostring() + keyword
    
    dtype = HEADER_KEYWORD_TYPES[keyword]

    dtype_to_type = {b'<l'  : np.int32,
                    b'str' : str,
                    b'<d'  : np.float64}
    
    value_dtype = dtype_to_type[dtype]

    if value_dtype is str:
        return np.int32(len(keyword)).tostring() + keyword + np.int32(len(value)).tostring() + value
    else:
        return np.int32(len(keyword)).tostring() + keyword + value_dtype(value).tostring()

def generate_signal():
    '''
        Generate a fake signal
    '''
    pi = 3.14
    # number of points
    npts = 20000
    # interval between times
    dt = 0.1
    # period of time signal
    period = 0.5

    # generate times
    times = np.linspace(0, (npts-1)*dt, npts)

    # generate signal
    fake_signal = np.sin(2*pi*times/period)

    # generate noise
    random_noise = np.random.normal(-20, 20, npts)

    # combine noise and signal
    new_signal = random_noise + fake_signal

    return times, new_signal


new_header = {
    b'source_name': b'P: 80.0000 ms, DM: 200.000',
    b'machine_id': 10,
    b'telescope_id': 4,
    b'data_type': 1,
    b'fch1': 400,
    b'foff': -0.062,
    b'nchans': 128,
    b'tstart': 6000.0,
    b'tsamp': 8e-05,
    b'nifs': 1,
    b'nbits': 8
}


# Run the code
times, new_signal = generate_signal()

# print(times)
plt.plot(new_signal)
plt.show()

# header_string = generate_header_string(new_header)

# write_data(8, './examples/pspm.fil', new_signal, header_string)

# writ_header = read_header('./examples/pspm.fil')

# print(writ_header)

# fil = Filterbank('./examples/pspm.fil', read_all=True)
# data = fil.select_data()
# print(data[1].shape)
