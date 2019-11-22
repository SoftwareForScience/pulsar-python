"""
    Example of plotting a Power Spectral Density plot, using filterbank data
"""
import asyncio
from filterbank.filterbank import Filterbank

async def streaming():
    # Instatiate the filterbank reader and point to the filterbank file
    fb = Filterbank(filename='./pspm32.fil', read_all=True)

    # read the data in the filterbank file
    f, samples = fb.select_data()

    # Get the powerlevels and the frequencies
    for i in range(len(samples)):
        # for j in range(len(samples[i])):
        #     print(samples[i][j])
        print(samples[i])
        print("")
        await asyncio.sleep(1)

loop = asyncio.get_event_loop()
loop.run_until_complete(streaming())