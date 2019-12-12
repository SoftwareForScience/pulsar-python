import asyncio
from filterbank.filterbank import Filterbank
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import os

async def print_filterbank_data(reader, writer, not_connected=True):
    '''
    Reads and write data to the client. Also gets filterbank and gives data it contains back to the client.
    @param reader: reads input from sys.stdin
    @type reader: Reader
    @param writer: writes data to server
    @type writer: Writer
    @return: prints Filterbank data
    '''

    commands = "Type 'pspm32.fil' or 'pspm16.fil' or 'pspm8.fil' " \
               "for filterbank data\nType '!quit' to close the connection\n"

    if not_connected:
        not_connected = False
        print("device connected")
        writer.write((("Welcome to the server!\n\n"+commands).encode()))
        await writer.drain()

    data = await reader.read(100)

    if data.decode().rstrip("\n\r") == "!quit":
        print("Close the connection")
        writer.close()

    else:
        print("Received: " + data.decode())
        message = data.decode().rstrip("\n\r")

        if message == "pspm32.fil" or message == "pspm16.fil" or message == "pspm8.fil":
            addr = writer.get_extra_info('peername')

            # Instatiate the filterbank reader and point to the filterbank file
            fb = Filterbank(filename="./" + message, read_all=True)

            # read the data in the filterbank file
            f, samples = fb.select_data()

            # Get the powerlevels and the frequencies
            for i in range(len(samples)):
                # for j in range(len(samples[i])):
                # print(samples[i][j])
                writer.write((str(samples[i]).encode()))
                writer.write((str("\n\n").encode()))
                await writer.drain()
                # await asyncio.sleep(1)

            print(f"Got file {message!r} from {addr!r}")

        else:
            writer.write(("'" + message + "'" + " is not accepted. \n" + commands).encode())

        await print_filterbank_data(reader, writer, not_connected)

async def main():

    '''
    Asynchronous Input stream server for filterbank data. Input is a filterbank filename without "./".
    Output is printed filterbank data. When this server is running, go to commandline and type:
    telnet <local_ip> 12345 if it is connected then write filename like: pspm32.fil
    @return: Server on localhost
    '''

    local_ip = os.popen("ipconfig getifaddr en0").read().strip('\n')
    server = await asyncio.start_server(
        print_filterbank_data, local_ip, 12345)

    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')


    async with server:
        await server.serve_forever()

asyncio.run(main())