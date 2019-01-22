"""
    Example for generating a fake filterbank file
"""
# pylint: disable-all
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from filterbank.filterbank import Filterbank
from filterbank.generate import generate_file


filename = './examples/pspm.fil'

header = {
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

# generate a fake filterbank file
generate_file(filename, header)

# read fake filterbank file
fil = Filterbank(filename, read_all=True)

data = fil.select_data()

print(data)