# pylint: disable-all
import os,sys,inspect

import waterfall as waterfall

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0,PARENT_DIR)
from plot import waterfall
import pylab as pyl

class CreateAsyncWaterfallPlot:

    async def create_waterfall(self, fb):
        '''
        creates an asynchronous waterfall object which needs a filterbank object where it initializes a plot and returns
        this object

        @param fb: A filterbank file for the plot
        @type fb: Filterbank
        @return: a waterplot object
        @rtype: Waterfall
        '''

        wf = waterfall.Waterfall(filter_bank=fb, fig=pyl.figure(), mode='stream', sync=False)
        await wf._init()
        wf.init_plot()
        return wf