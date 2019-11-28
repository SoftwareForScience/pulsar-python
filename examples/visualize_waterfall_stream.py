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


fb = Filterbank(filename='./pspm32.fil')

wf = waterfall.Waterfall(filter_bank=fb, fig=pyl.figure(), mode="stream")

fig, update, frames, repeat = wf.animated_plotter()

ani = animation.FuncAnimation(fig, update, frames=frames,repeat=repeat)
pyl.show(block=True)
