import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import numpy as np
import matplotlib.pyplot as plt
from filterbank.header import read_header
from filterbank.filterbank import Filterbank


# fil = Filterbank('./pspm_test.fil', read_all=True)

# print(fil.header)

# data = fil.select_data()


samples = np.linspace(0, 0.5, 50)

print(samples)
