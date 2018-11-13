# # """
# #     Example of plotting a Power Spectral Density plot, using filterbank data
# # """
# # # pylint: disable-all
# # import os
# # import sys
# # import inspect
# # import numpy as np
# # import matplotlib.pyplot as plt

# # CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# # PARENT_DIR = os.path.dirname(CURRENT_DIR)
# # sys.path.insert(0, PARENT_DIR)

# # from plot import psd
# # from fourier import fft_vectorized
# # from filterbank.header import read_header
# # from filterbank.filterbank import Filterbank

# # # Instatiate the filterbank reader and point to the filterbank file
# # fb = Filterbank(filename='pspm32.fil')

# # # read the data in the filterbank file
# # _, samples = fb.select_data()

# # # Convert 2D Array to 1D Array with complex numbers
# # # samples = samples[0] + (samples[1] * 1j)

# # npfft = np.fft.fft2(samples)
# # myfft = samples
# # myfft[0] = fft_vectorized(samples[0])
# # myfft[1] = fft_vectorized(samples[1])
# # myfft2 = fft_vectorized(samples)
# # print('my2', myfft2.shape)
# # print(npfft.shape)
# # print('n00',npfft[0][0])
# # print('n10',npfft[1][0])
# # print(myfft.shape)
# # print('m00',myfft[0][0])
# # print('m10',myfft[1][0])
# # print(samples.shape)

# # print(fft_vectorized(samples[0][0:8]))


# # # Read the header of the filterbank file
# # # header = read_header('pspm32.fil')

# # # # Calculate the center frequency with the data in the header
# # # center_freq = header[b'fch1'] + float(header[b'nchans']) * header[b'foff'] / 2.0


# # # ## plotting shit
# # # # transformed_samples = fft_vectorized(samples)

# # # fs = 80

# # # t = np.arange(0, 1.6, 1/fs)

# # # sp = fft_vectorized(samples)

# # # print(samples[0])
# # # print(np.sin(2*np.pi * 40 * t))

# # # trange = np.linspace(0, fs, len(t)) + center_freq

# # # plt.plot(trange, np.abs(sp))
# # # plt.show()

# # # # Get the powerlevels and the frequencies
# # # PXX, freqs = psd(samples, nfft=1024, sample_rate=80,
# # #                  scale_by_freq=True, sides='twosided')

# # # # Calculate the powerlevel dB's
# # # power_levels = 10 * np.log10(PXX/(80))

# # # # Add the center frequency to the frequencies so they match the actual frequencies
# # # freqs = freqs + center_freq

# # # # Plot the PSD
# # # plt.grid(True)
# # # plt.xlabel('Frequency (MHz)')
# # # plt.ylabel('Intensity (dB)')
# # # plt.plot(freqs, power_levels)
# # # plt.show()


# """
#     Example of plotting a Power Spectral Density plot, using filterbank data
# """
# # pylint: disable-all
# import os
# import sys
# import inspect
# import numpy as np
# import matplotlib.pyplot as plt

# CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# PARENT_DIR = os.path.dirname(CURRENT_DIR)
# sys.path.insert(0, PARENT_DIR)

# from plot import opsd
# from plot import psd
# from fourier.fourier import fft2, fft_vectorized
# # from pylab import *
# from filterbank.header import read_header
# from filterbank.filterbank import Filterbank

# # Instatiate the filterbank reader and point to the filterbank file
# fb = Filterbank(filename='examples/pspm32.fil')

# # read the data in the filterbank file
# _, samples = fb.select_data()
# # print(x.shape)

# # samples = samples.reshape((samples.shape[0]*samples.shape[1],))

# print(samples.reshape((samples.shape[0]*samples.shape[1],)).shape)

# # evens = samples[::2]
# # odds = samples[1::2]

# # my = fft_vectorized(samples)

# # nump = np.fft.fft(samples)

# # print("my", my.shape)
# # print("np", nump.shape)

# # print(my[0:5])
# # print(nump[0:5])
# # Convert 2D Array to 1D Array with complex numbers
# # samples = samples[0] + (samples[1] * 1j)

# # # Read the header of the filterbank file
# header = read_header('examples/pspm32.fil')


# # # Calculate the center frequency with the data in the header
# center_freq = header[b'fch1'] + float(header[b'nchans']) * header[b'foff'] / 2.0

# # # Get the powerlevels and the frequencies

# def next_power_of_2(val):
#     """
#         Return the next integer that is a power of two

#         Params
#         ------
#         val : int
#     """

#     return int(2**(np.log2(val) + 1))
# # print(len(samples[0]))

# # y, x = mpsd(samples[0], nfft=8192, sample_rate=80, sides='twosided')

# x, y = psd(samples, 80, center_freq, nfft=5000)

# # psd(samples[0], NFFT=next_power_of_2(5000), Fs=80, Fc=center_freq)

# # x, y = npsd(samples, nfft=1024, sampling_frequency=80, center_frequency=center_freq)
# # # Calculate the powerlevel dB's
# # # power_levels = 10 * np.log10(PXX/(80))

# # # Add the center frequency to the frequencies so they match the actual frequencies
# # # x = freqs + center_freq

# # # Plot the PSD
# plt.grid(True)
# plt.xlabel('Frequency (MHz)')
# plt.ylabel('Intensity (dB)')
# plt.plot(x, y)
# plt.show()
import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
import matplotlib.pyplot as plt
from rtlsdr import *
from plot import psd as npsd
from plot import opsd
from pylab import *

sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.4e6
sdr.center_freq = 95e6
sdr.gain = 4

samples = sdr.read_samples(256*1024)
sdr.close()

# npsd
# x, y = psd(samples, sdr.sample_rate/1e6, sdr.center_freq/1e6, nfft=1024)

#matplotlib
# psd(samples, NFFT=256*1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)

# opsd
y, x, _ = opsd(samples, nfft=256, sample_rate=sdr.sample_rate/1e6, sides='twosided')

plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (dB)')
plt.plot(x, y)
plt.show()
show()