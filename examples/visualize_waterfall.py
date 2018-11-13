import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import matplotlib.animation as animation
from matplotlib.mlab import psd
import pylab as pyl
import numpy as np
import sys
from rtlsdr import RtlSdr
from plot import Waterfall

sdr = RtlSdr()
wf = Waterfall.Waterfall(sdr)

# some defaults
sdr.rs = 2.4e6
sdr.fc = 100e6
sdr.gain = 10

wf.start()

# # cleanup
sdr.close()