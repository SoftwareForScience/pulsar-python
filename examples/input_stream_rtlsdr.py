import asyncio
from rtlsdr import RtlSdr

async def streaming():
    sdr = RtlSdr()

    async for samples in sdr.stream():
        print(samples[1:5])
        # ...

    # to stop streaming:
    await sdr.stop()

    # done
    sdr.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(streaming())