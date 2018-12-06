import os
import sys
import inspect

# Volkswagened this example as it is just an example for educational purposes, no production code.
# pylint: disable-all
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from timeseries.timeseries import Timeseries
from filterbank.filterbank import Filterbank
import matplotlib.pyplot as plt


fb = Filterbank("./pspm32.fil")
fb.read_filterbank()

time_series = Timeseries().from_filterbank(fb)

decimated_tv = time_series.downsample(3)

plt.plot(decimated_tv)
plt.show()
