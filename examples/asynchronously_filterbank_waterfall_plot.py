# pylint: disable-all
import os,sys,inspect

import waterfall as waterfall

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0,PARENT_DIR)
import matplotlib.animation as animation
from filterbank.header import read_header
from filterbank.filterbank import Filterbank
from plot import waterfall
import pylab as pyl
import matplotlib.pyplot as plt
from plot.plot import next_power_of_2
import asyncio
import time

async def create_waterfall(fb):
    wf = waterfall.Waterfall(filter_bank=fb, fig=pyl.figure(), mode='stream')
    await wf._init()
    wf.init_plot()
    return wf

async def create_filterbank(test_async=None):
    filterbank = Filterbank(filename='./pspm32.fil')

    # just a async test to see if it works
    if test_async is not None:
        await asyncio.sleep(1)
        print(test_async)
    else:
        print("1")


    # await filterbank._init()

    return filterbank

async def main():
    # creates fb2 first and then fb1, because of sleep delay in create_filterbank method
    fb1, fb2 = await asyncio.gather(create_filterbank("2"), create_filterbank())
    wf1, wf2 = await asyncio.gather(create_waterfall(fb1), create_waterfall(fb2))

    fig, update, frames, repeat = wf1.animated_plotter()

    ani = animation.FuncAnimation(fig, update, frames=frames,repeat=repeat)
    plt.show(block=True)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
