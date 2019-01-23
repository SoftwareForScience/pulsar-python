# pylint: disable-all
# Disabled all PyLint checking for examples files since this is not required.
import os,sys,inspect
from filterbank.filterbank import Filterbank
from plot import waterfall
import pylab as pyl

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0,PARENT_DIR)


filter_bank = Filterbank(filename='./pspm32.fil')

# Calculate the center frequency with the data in the header
center_freq = filter_bank.header[b'center_freq']

wf = waterfall.Waterfall(filter_bank=filter_bank, fig=pyl.figure(), center_freq=center_freq, t_obs=0.01, mode="discrete")
img = wf.get_raw_image()

pyl.show(img)