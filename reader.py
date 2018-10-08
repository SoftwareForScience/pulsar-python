import os
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt


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

    if n_bytes > 255:
        n_bytes = 16

    keyword = fh.read(n_bytes)

    if keyword == b'HEADER_START' or keyword == b'HEADER_END':
        return keyword, 0, fh.tell()
    else:
        dtype = header_keyword_types[keyword]
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

def len_header(filename):
    """ Return the length of the blimpy header, in bytes
    Args:
        filename (str): name of file to open
    Returns:
        idx_end (int): length of header, in bytes
    """
    with open(filename, 'rb') as f:
        header_sub_count = 0
        eoh_found = False
        while not eoh_found:
            header_sub = f.read(512)
            header_sub_count += 1
            if b'HEADER_END' in header_sub:
                idx_end = header_sub.index(b'HEADER_END') + len(b'HEADER_END')
                eoh_found = True
                break
        idx_end = (header_sub_count - 1) * 512 + idx_end
    return idx_end


def read_string(filename, header_len, stdout=False):
    filfile = open(filename, 'rb')
    strlen = unpack('i', filfile.read(4))[0]
    strval = filfile.read(strlen)
    if stdout:
        print("  string = '%s'"%strval)
    return strval


def read_test(filename, header_len):
    # f = open(filename, 'rb')
    # f.seek(header_len, 1)
    # data = f.read(128 * 4 * 4)

    with open(filename, 'rb') as bin_file:
        bin_file.seek(header_len, 1)
        # data = np.fromfile(bin_file, dtype=b'float32')
        # data = np.fromfile(bin_file)
        str_bytes = bin_file.read(4)
        data = unpack(b'<d', str_bytes)
        # for i in range(0, 4 * 4):
        # little endian?
        # data = unpack(b'<f', bin_file.read(4))[0]
        return data
    # values = unpack('d', data)[0]
    # np_data = np.fromfile(f, dtype='float32').reshape(15676, 128)
    # data = np_data.reshape(128, 15679).transpose()
    # int_values = [x for x in data]
    # return data

def read_advanced(filename, header_len, t_start, t_stop):
    # Load binary data 
    f = open(filename, 'rb')
    f.seek(header_len)
    
    # Variables from header
    n_bytes = 2
    n_chans = 128
    n_chans_selected = 1
    n_ifs = 1
    # n_bytes  = self.header['nbits'] / 8
    # n_chans = self.header['nchans']
    # n_chans_selected = self.freqs.shape[0]
    # n_ifs   = self.header['nifs']

    f_delt = -0.062
    f0 = 433.968

    # tstart = 50000.0
    # tsamp = 80
    # only read first integration of large file (for now, later more flexible)
    filesize = os.path.getsize(filename)
    n_bytes_data = filesize - header_len
    n_ints_in_file = n_bytes_data / (n_bytes * n_chans * n_ifs)
    
    # now check to see how many integrations requested
    ii_start, ii_stop = 0, n_ints_in_file
    if t_start:
        ii_start = t_start
    if t_stop:
        ii_stop = t_stop
    n_ints = int(ii_stop - ii_start)

    #
    chan_start_idx, chan_stop_idx = 0, n_chans
    if f_delt < 0:
        chan_start_idx, chan_stop_idx = n_chans, 0
        

    freqs = np.arange(0, n_chans, 1, dtype='float64') * f_delt + f0
    
    if chan_start_idx > chan_stop_idx:
        freqs    = freqs[chan_stop_idx:chan_start_idx]
        freqs    = freqs[::-1]
    else:
        freqs    = freqs[chan_start_idx:chan_stop_idx]
    
    
    # Set up indexes used in file read (taken out of loop for speed)
    i0 = np.min((chan_start_idx, chan_stop_idx))
    i1 = np.max((chan_start_idx, chan_stop_idx)) 

    data = np.zeros((n_ints, n_ifs, n_chans_selected), dtype='float32')
    
    for ii in range(n_ints):
        """d = f.read(n_bytes * n_chans * n_ifs)  
        """
        
        for jj in range(n_ifs):
            f.seek(n_bytes * i0, 1) # 1 = from current location
            d = f.read(n_bytes * n_chans_selected)          
            f.seek(n_bytes * (n_chans - i1), 1)
        
            if n_bytes == 4:
                dd = np.fromstring(d, dtype='float32')
            elif n_bytes == 2:
                dd = np.fromstring(d, dtype='int16')
            elif n_bytes == 1:
                dd = np.fromstring(d, dtype='int8')
            
            # Reverse array if frequency axis is flipped
            if f_delt < 0:
                dd = dd[::-1]
            
            data[ii, jj] = dd        

    return data


# Reading bytes
def reading_bytes(filename, header_len):
    # read the filterbank file
    fh = open(filename, 'rb')
    # skip the header
    fh.seek(header_len, 1)

    # create a string of bytes
    str_bytes = fh.read(128 * 4)
    # str_bytes = b'\\xb7'

    print(str_bytes)
    print(len(str_bytes))

    # bytes to actual value [uint8/16/32, float16/32]
    values = np.fromstring(str_bytes, dtype='int32')
    # values = unpack(b'<f', str_bytes)[0]

    return values


filename = '../test-data/pspm2.fil'
length = len_header(filename)


values = reading_bytes(filename, length)
# values = read_string(filename, length)
# values = read_test(filename, length)
# values = read_advanced(filename, length, 9000.0, 10000.0)
print(values)
plt.plot(values)
plt.show()


# value = read_test(length, filename)
# value = read_header(filename)
# print(value[b'nbits'])
# print(value)
# plt.plot(value)
# plt.show()
