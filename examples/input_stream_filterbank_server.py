import asyncio
from filterbank.filterbank import Filterbank
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import numpy as np
import pickle

async def print_filterbank_data(reader, writer):
    '''
    Reads and write data to the client. Also gets filterbank and gives data it contains back to the client.
    @param reader: reads input from sys.stdin
    @type reader: Reader
    @param writer: writes data to server
    @type writer: Writer
    @return: prints Filterbank data
    '''

    data = await reader.read(100)
    print(data)
    message = data.decode().rstrip("\n\r")
    addr = writer.get_extra_info('peername')

    # Instatiate the filterbank reader and point to the filterbank file
    fb = Filterbank(filename="./"+message, read_all=True)

    # read the data in the filterbank file
    f, samples = fb.select_data()

    # Get the powerlevels and the frequencies
    for i in range(len(samples)):
        # for j in range(len(samples[i])):
        # print(samples[i][j])
        writer.write((str(samples[i]).encode()))
        await writer.drain()
        # await asyncio.sleep(1)

    print(f"Got file {message!r} from {addr!r}")

    # writer.write(data)

    await print_filterbank_data(reader, writer)

    # print("Close the connection")
    # writer.close()

async def main():

    '''
    Asynchronous Input stream server for filterbank data. Input is a filterbank filename without "./".
    Output is printed filterbank data. When this server is running, go to commandline and type: telnet 127.0.0.1 8080
    if it is connected then write filename like: pspm32.fil
    @return: Server on localhost
    '''

    server = await asyncio.start_server(
        print_filterbank_data, '127.0.0.1', 8080)

    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')


    async with server:
        await server.serve_forever()

asyncio.run(main())