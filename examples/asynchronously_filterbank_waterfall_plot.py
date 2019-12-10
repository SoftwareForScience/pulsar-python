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


async def create_waterfall(fb):
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

async def create_filterbank(test_async=None):

    '''
    creates an asyncronous Filterbank file. Also it checks if the the function is actually asynchronous.
    @param test_async: checks to so if the method is asynchronous.

    @type test_async: String
    @return: A filterbank object
    @rtype: Filterbank
    '''

    filterbank = AsyncFilterbank(filename='./pspm32.fil')

    # just a async test to see if it works
    if test_async is not None:
        # if the following line is commented out then it will first create fb1 and then fb2. See main() for the comment.
        await asyncio.sleep(1)
        print(test_async)
    else:
        print("1")

    return filterbank

async def main():
    '''
    creates streaming waterfall plot by getting a filterbank file as an input.

    @return: shows streaming plot. You can use "ipython" (search it on Google or go to:
    https://ipython.org/install.html) to get an animated plot
    '''

    # creates fb2 first and then fb1, because of sleep delay in create_filterbank method
    fb1, fb2 = await asyncio.gather(create_filterbank("2"), create_filterbank())
    wf1, wf2 = await asyncio.gather(create_waterfall(fb1), create_waterfall(fb2))

    fig, update, frames, repeat = wf1.animated_plotter()
    ani = animation.FuncAnimation(fig, update, frames=frames,repeat=repeat)

    fig2, update2, frames2, repeat2 = wf2.animated_plotter()
    ani2 = animation.FuncAnimation(fig2, update2, frames=frames2, repeat=repeat2)
    plt.show(block=True)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
