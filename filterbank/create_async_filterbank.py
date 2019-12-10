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

class CreateAsyncFilterbank:

    async def create_filterbank(self, filename="./pspm32.fil", test_async=None):

        '''
        creates an asyncronous Filterbank file. Also it checks if the the function is actually asynchronous.
        @param test_async: checks to so if the method is asynchronous.

        @type test_async: String
        @return: A filterbank object
        @rtype: Filterbank
        '''

        filterbank = AsyncFilterbank(filename=filename)

        # just a async test to see if it works
        if test_async is not None:
            # if the following line is commented out then it will first create fb1 and then fb2. See main() in
            # asynchronously_filterbank_waterfall_plot.py for the comment.
            await asyncio.sleep(1)
            print(test_async)
        else:
            print("1")

        return filterbank
