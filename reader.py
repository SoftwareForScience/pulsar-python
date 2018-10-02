from struct import unpack
import numpy as np

header_keyword_types = {
    b'telescope_id' : b'<l',
    b'machine_id'   : b'<l',
    b'data_type'    : b'<l',
    b'barycentric'  : b'<l',
    b'pulsarcentric': b'<l',
    b'nbits'        : b'<l',
    b'nsamples'     : b'<l',
    b'nchans'       : b'<l',
    b'nifs'         : b'<l',
    b'nbeams'       : b'<l',
    b'ibeam'        : b'<l',
    b'rawdatafile'  : b'str',
    b'source_name'  : b'str',
    b'az_start'     : b'<d',
    b'za_start'     : b'<d',
    b'tstart'       : b'<d',
    b'tsamp'        : b'<d',
    b'fch1'         : b'<d',
    b'foff'         : b'<d',
    b'refdm'        : b'<d',
    b'period'       : b'<d',
    b'src_raj'      : b'angle',
    b'src_dej'      : b'angle',
    }

def read_header(filename, return_idxs=False):
    """ Read blimpy header and return a Python dictionary of key:value pairs
    Args:
        filename (str): name of file to open
    Optional args:
        return_idxs (bool): Default False. If true, returns the file offset indexes
                            for values
    returns
    """
    with open(filename, 'rb') as fh:
        header_dict = {}
        header_idxs = {}

        # Check this is a blimpy file
        keyword, value, idx = read_next_header_keyword(fh)

        try:
            assert keyword == b'HEADER_START'
        except AssertionError:
            raise RuntimeError("Not a valid blimpy file.")

        while True:
            keyword, value, idx = read_next_header_keyword(fh)
            if keyword == b'HEADER_END':
                break
            else:
                header_dict[keyword] = value
                header_idxs[keyword] = idx

    if return_idxs:
        return header_idxs
    else:
        return header_dict

def read_next_header_keyword(fh):
    """
    Args:
        fh (file): file handler
    Returns:
    """
    n_bytes = np.fromstring(fh.read(4), dtype='uint32')[0]
    #print n_bytes

    if n_bytes > 255:
        n_bytes = 16

    keyword = fh.read(n_bytes)

    #print keyword

    if keyword == b'HEADER_START' or keyword == b'HEADER_END':
        return keyword, 0, fh.tell()
    else:
        dtype = header_keyword_types[keyword]
        #print dtype
        idx = fh.tell()
        if dtype == b'<l':
            val = unpack(dtype, fh.read(4))[0]
        if dtype == b'<d':
            val = unpack(dtype, fh.read(8))[0]
        if dtype == b'str':
            str_len = np.fromstring(fh.read(4), dtype='int32')[0]
            val = fh.read(str_len)
        if dtype == b'angle':
            val = unpack('<d', fh.read(8))[0]
            val = fil_double_to_angle(val)
        return keyword, val, idx

def fil_double_to_angle(angle):
      """ Reads a little-endian double in ddmmss.s (or hhmmss.s) format and then
      converts to Float degrees (or hours).  This is primarily used to read
      src_raj and src_dej header values. """

      negative = (angle < 0.0)
      angle = np.abs(angle)

      dd = np.floor((angle / 10000))
      angle -= 10000 * dd
      mm = np.floor((angle / 100))
      ss = angle - 100 * mm
      dd += mm/60.0 + ss/3600.0

      if negative:
          dd *= -1

      return dd


value = read_header("../test-data/pspm_small.fil", True)
print(value)