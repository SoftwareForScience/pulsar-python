import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from rtlsdr import RtlSdr
from filterbank.header import read_header
from filterbank.filterbank import Filterbank
from plot import Waterfall

fb = Filterbank(filename='examples/pspm32.fil')

# read the data in the filterbank file
freqs, samples = fb.select_data()

sdr = RtlSdr()
wf = Waterfall.Waterfall(sdr)

# some defaults
sdr.rs = 2.4e6
sdr.fc = 100e6
sdr.gain = 10

wf.start()

# # cleanup
sdr.close()