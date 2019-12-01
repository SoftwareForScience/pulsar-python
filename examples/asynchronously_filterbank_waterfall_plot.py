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

async def create_filterbank():
    filterbank = Filterbank(filename='./pspm32.fil')
    # await filterbank._init()

    return filterbank

async def main():
    # fb = await create_filterbank()
    fb = Filterbank(filename='./pspm32.fil')

    # wf = waterfall.Waterfall(filter_bank=fb, fig=pyl.figure(), mode='stream')
    wf = await create_waterfall(fb)

    # fig, update, frames, repeat = await wf.animated_plotter()

    fig, update, frames, repeat = wf.animated_plotter()

    ani = animation.FuncAnimation(fig, update, frames=frames,repeat=repeat)
    plt.show(block=True)


if __name__ == "__main__":
    print(f"started at {time.strftime('%X')}")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    print(f"finished at {time.strftime('%X')}")
