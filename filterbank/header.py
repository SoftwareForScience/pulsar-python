"""
    Utilities for reading header of filterbank file
"""

from struct import unpack
import numpy as np

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

def read_header(filename):
    """
        Read Filterbank header and return a dictionary of key-value pairs
    """
    with open(filename, 'rb') as file_header:
        header_dict = {}

        keyword, value = read_next_header_keyword(file_header)

        try:
            assert keyword == b'HEADER_START'
        except AssertionError:
            raise RuntimeError("Not a valid Filterbank file.")

        while True:
            keyword, value = read_next_header_keyword(file_header)
            if keyword == b'HEADER_END':
                break
            else:
                header_dict[keyword] = value

    return header_dict

def read_next_header_keyword(file_header):
    """
        Read key-value pair from header
    """
    n_bytes = np.fromstring(file_header.read(4), dtype='uint32')[0]

    if n_bytes > 255:
        n_bytes = 16

    keyword = file_header.read(n_bytes)

    if b'HEADER_START' in keyword or b'HEADER_END' in keyword:
        return keyword, 0

    dtype = HEADER_KEYWORD_TYPES[keyword]

    if dtype == b'<l':
        val = unpack(dtype, file_header.read(4))[0]
    if dtype == b'<d':
        val = unpack(dtype, file_header.read(8))[0]
    if dtype == b'str':
        str_len = np.fromstring(file_header.read(4), dtype='int32')[0]
        val = file_header.read(str_len)
    if dtype == b'angle':
        val = unpack('<d', file_header.read(8))[0]
        val = fil_double_to_angle(val)

    return keyword, val

def fil_double_to_angle(angle):
    """
        Reads a little-endian double in ddmmss.s (or hhmmss.s) format and then
        converts to Float degrees (or hours).
    """
    negative = (angle < 0.0)
    angle = np.abs(angle)

    data_matrix = np.floor((angle / 10000))
    angle -= 10000 * data_matrix
    time_minutes = np.floor((angle / 100))
    time_seconds = angle - 100 * time_minutes
    data_matrix += time_minutes / 60.0 + time_seconds / 3600.0

    if negative:
        data_matrix *= -1

    return data_matrix

def len_header(filename):
    """
        Return the length of the header in bytes
    """
    chunk_size = 512

    with open(filename, 'rb') as file_:
        header_sub_count = 0
        eoh_found = False
        while not eoh_found:
            header_sub = file_.read(chunk_size)
            header_sub_count += 1
            if b'HEADER_END' in header_sub:
                idx_end = header_sub.index(b'HEADER_END') + len(b'HEADER_END')
                eoh_found = True
                break
        idx_end = int((header_sub_count - 1) * chunk_size + idx_end)
    return idx_end
    