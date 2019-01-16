"""
    Generation of mocking data
"""
import numpy as np

def generate():
    """
        Generate mock filterbank data
    """
    n_chans = 128
    # n_bits = 4
    tstart = 50000.0
    # t_samp = 80.0e-6
    f_ch1 = 433.968
    f_off = -0.062
    n_ifs = 1
    period = 3.1415927
    # d_m = 30
    nsblk = 512

    i_c = n_ifs*n_chans
    arraysize = n_chans*n_ifs*nsblk

    fblock = np.empty(arraysize, dtype=object)

    for x_val in range(0, nsblk):
        for i in range(0, n_ifs):
            for y_val in range(0, n_chans):
                fblock[x_val*i_c+i*n_chans+y_val] = f_ch1 + y_val * f_off

    file_str = "HEADER_START"
    file_str += "P:" + str(period) + " ms"
    file_str += "DM: 30"
    file_str += "data_type: 1"
    file_str += "machine_id: 10"
    file_str += "telescope_id: 1"
    file_str += "nchans:" + str(n_chans)
    file_str += "tstart:" + str(tstart)
    file_str += "fch1:" + str(f_ch1)
    file_str += "HEADER_END"
    file_str += "SIGNAL_START"
    file_str += ''.join(map(str, fblock))
    file_str += "SIGNAL_END"

    file_name = open("../pspm.fil", "w")

    file_name.write(str(file_str.encode()))

    file_name.close()

generate()
