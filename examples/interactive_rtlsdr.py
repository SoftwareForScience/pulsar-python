from matplotlib import pyplot as plt
import matplotlib.animation as animation
from rtlsdr import RtlSdr
import numpy as np

sdr = RtlSdr()
# configure device
sdr.sample_rate = 2.4e6  # Hz
sdr.center_freq = 94.7e6  # Hz
sdr.freq_correction = 60   # PPM
sdr.gain = 'auto'

fig = plt.figure()
graph_out = fig.add_subplot(1, 1, 1)


def animate(i):
    graph_out.clear()
    #samples = sdr.read_samples(256*1024)
    samples = sdr.read_samples(128*1024)
    # use matplotlib to estimate and plot the PSD
    graph_out.psd(samples, NFFT=1024, Fs=sdr.sample_rate /
                  1e6, Fc=sdr.center_freq/1e6)
    #graph_out.xlabel('Frequency (MHz)')
    #graph_out.ylabel('Relative power (dB)')


try:
    ani = animation.FuncAnimation(fig, animate, interval=10)
    plt.show(block=True)
except KeyboardInterrupt:
    pass
finally:
    sdr.close() 