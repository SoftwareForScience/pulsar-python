# pylint: disable-all
import os,sys,inspect
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0,PARENT_DIR)
import matplotlib.animation as animation
from filterbank.header import read_header
from filterbank.filterbank import Filterbank
from plot import waterfall
import pylab as pyl
from plot.plot import next_power_of_2


fb = Filterbank(filename='examples/pspm32bit.fil')

# read the data in the filterbank file
freqs, samples = fb.select_data()
print(freqs.shape)
print(samples.shape)

# Read the header of the filterbank file
header = read_header('examples/pspm32bit.fil')

# Calculate the center frequency with the data in the header
center_freq = header[b'fch1'] + float(header[b'nchans']) * header[b'foff'] / 2.0

wf = waterfall.Waterfall(samples=samples[0:next_power_of_2(8000)], freqs=freqs, fig=pyl.figure(), center_freq=center_freq)
img = wf.get_raw_image()

pyl.show(img)

