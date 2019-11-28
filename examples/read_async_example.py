"""
    Example of plotting a Power Spectral Density plot, using filterbank data
"""
# pylint: disable-all
import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from plot import opsd
from filterbank.header import read_header
from filterbank.filterbank import Filterbank
import asyncio
import time

async def create_filterbank():
    filterbank = Filterbank(filename='./pspm32.fil', read_all=True)
    await filterbank._init()

    return filterbank

async def main():
# Instatiate the filterbank reader and point to the filterbank file
    print(f"started at {time.strftime('%X')}")

    fb = await create_filterbank()
    for i in range(0, 127):
        row = await fb.next_row()
        print(row)


    # # read the data in the filterbank file
    # f, samples = await fb.select_data()
    # print(f)

    # print(samples)

    print(f"finished at {time.strftime('%X')}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
