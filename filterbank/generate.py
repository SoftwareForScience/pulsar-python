"""
    Functions for creating a filterbank file
    with a fake signal and header
"""
import numpy as np

from .header import HEADER_KEYWORD_TYPES

# pylint: disable=E1121

PI = 3.14

def generate_file(filename, header, noise_level=20, t_obs=2, n_pts=10):
    """
        Combines functionality of all functions
    """
    n_bytes = header[b'nbits']/8
    header_string = generate_header(header)
    signal_data = generate_signal(header, noise_level, t_obs, n_pts)
    write_data(filename, signal_data, n_bytes, header_string)


def generate_signal(header, noise_level, t_obs, n_pts):
    """
        Create a signal using the header values

        Args:
            noise_level, the max amplitude of the generated noise
            period, period of the signal
            t_obs, observation time in s
            n_pts, intervals between samples
    """
    n_samples = int(t_obs/header[b'tsamp'])
    # create an empty vector for the signals
    signal_data = np.zeros((n_samples, header[b'nchans']))
    # create array with size equal to the num of channels
    sample = np.linspace(0, n_pts, header[b'nchans'])
    # create a signal for each sample
    for i in range(n_samples):
        signal = np.sin(2*PI*sample/header[b'period'])
        noise = np.random.normal(0, noise_level, header[b'nchans'])
        signal_data[i] = signal + noise
    return signal_data


def generate_header(header):
    """
        Creates a header string
    """
    # create start of header
    header_string = keyword_to_string(b'HEADER_START')
    # add header dictionary keys and values to string
    for keyword in header.keys():
        header_string += keyword_to_string(keyword, header[keyword])
    # append end of header
    header_string += keyword_to_string(b'HEADER_END')
    return header_string


def keyword_to_string(keyword, value=None):
    """
        Converts a keyword and value to a serialized string
    """
    keyword = bytes(keyword)
    # value of attribute has not been specified
    if value is None:
        return np.int32(len(keyword)).tostring() + keyword
    # select datatype from keywords dictionary
    dtype = HEADER_KEYWORD_TYPES[keyword]
    # dictionary for transforming datatype to numpy type
    dtype_to_type = {b'str' : str,
                     b'<l'  : np.int32,
                     b'<d'  : np.float64}
    # select numpy type accordingly
    value_dtype = dtype_to_type[dtype]
    keyword_string = np.int32(len(keyword)).tostring() + keyword
    # cast value to correct numpy type
    if value_dtype is str:
        keyword_string += np.int32(len(value)).tostring() + value
    else:
        keyword_string += value_dtype(value).tostring()
    return keyword_string


def write_data(filename, fil_data, n_bytes, header_str):
    """
        Write the generated signal and header to filterbank file
    """
    # open file and write as bytes
    with open(filename, 'wb') as new_file:
        new_file.write(header_str)
        if n_bytes == 1:
            np.int8(fil_data.ravel()).tofile(new_file)
        elif n_bytes == 2:
            np.int16(fil_data.ravel()).tofile(new_file)
        elif n_bytes == 4:
            np.float32(fil_data.ravel()).tofile(new_file)
