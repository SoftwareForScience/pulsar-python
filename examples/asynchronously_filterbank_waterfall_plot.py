# pylint: disable-all
import os,sys,inspect

import waterfall as waterfall

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0,PARENT_DIR)
import matplotlib.animation as animation
from filterbank.async_filterbank import AsyncFilterbank
from plot import waterfall
import pylab as pyl
import matplotlib.pyplot as plt
import asyncio
from filterbank.create_async_filterbank import CreateAsyncFilterbank
from plot.create_async_waterfall_plot import CreateAsyncWaterfallPlot

async def main():
    '''
    creates streaming waterfall plot by getting a filterbank file as an input.

    @return: shows streaming plot. You can use "ipython" (search it on Google or go to:
    https://ipython.org/install.html) to get an animated plot
    '''

    afb = CreateAsyncFilterbank()
    awf = CreateAsyncWaterfallPlot()

    # creates fb2 first and then fb1, because of sleep delay in create_filterbank method
    fb1, fb2 = await asyncio.gather(afb.create_filterbank("./pspm32.fil", "2"), afb.create_filterbank("./pspm32.fil"))
    wf1, wf2 = await asyncio.gather(awf.create_waterfall(fb1), awf.create_waterfall(fb2))

    fig, update, frames, repeat = wf1.animated_plotter()
    ani = animation.FuncAnimation(fig, update, frames=frames,repeat=repeat)

    fig2, update2, frames2, repeat2 = wf2.animated_plotter()
    ani2 = animation.FuncAnimation(fig2, update2, frames=frames2, repeat=repeat2)
    plt.show(block=True)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
