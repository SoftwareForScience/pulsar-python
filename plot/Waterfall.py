from __future__ import division
import matplotlib.animation as animation
from plot import opsd
import pylab as pyl
import numpy as np
import sys
from rtlsdr import RtlSdr

class Waterfall(object):
    NFFT                    = 1024
    num_samples_per_scan    = NFFT*16
    num_buffered_sweeps     = 100
    num_scans_per_sweep     = 1
    freq_inc_coarse         = 1e6
    freq_inc_fine           = 0.1e6
    gain_inc                = 5
    shift_key_down          = False
    keyboard_buffer         = []
    def __init__(self, sdr=None, fig=None, NFFT=None, num_samples_per_scan=None, num_buffered_sweeps=None, num_scans_per_sweep=None, freq_inc_coarse=None, freq_inc_fine=None, gain_inc=None):
        """
            Setup waterfall class
        """
        self.fig = fig if fig else pyl.figure()
        self.sdr = sdr if sdr else RtlSdr()

        #set constanst
        self.NFFT                   = NFFT if NFFT else self.NFFT
        self.num_samples_per_scan   = num_samples_per_scan if num_samples_per_scan else self.num_samples_per_scan
        self.num_buffered_sweeps    = num_buffered_sweeps if num_buffered_sweeps else self.num_buffered_sweeps
        self.num_scans_per_sweep    = num_scans_per_sweep if num_scans_per_sweep else self.num_scans_per_sweep
        self.freq_inc_coarse        = freq_inc_coarse if freq_inc_coarse else self.freq_inc_coarse
        self.freq_inc_fine          = freq_inc_fine if freq_inc_fine else self.freq_inc_fine
        self.gain_inc               = gain_inc if gain_inc else self.gain_inc

        self.init_plot()

    def init_plot(self):
        self.image_buffer = -100*np.ones((self.num_buffered_sweeps,\
                                 self.num_scans_per_sweep*self.NFFT))

        self.ax = self.fig.add_subplot(1,1,1)
        self.image = self.ax.imshow(self.image_buffer, aspect='auto',\
                                    interpolation='nearest', vmin=-50, vmax=10)
        self.ax.set_xlabel('Current frequency (MHz)')
        self.ax.get_yaxis().set_visible(False)

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

    def update_plot_labels(self):
        fc = self.sdr.fc
        rs = self.sdr.rs
        freq_range = (fc - rs/2)/1e6, (fc + rs*(self.num_scans_per_sweep - 0.5))/1e6

        self.image.set_extent(freq_range + (0, 1))
        self.fig.canvas.draw_idle()

    def update(self, *args):
        # save center freq. since we're gonna be changing it
        start_fc = self.sdr.fc

        # prepare space in buffer
        # TODO: use indexing to avoid recreating buffer each time
        self.image_buffer = np.roll(self.image_buffer, 1, axis=0)

        for scan_num, start_ind in enumerate(range(0, self.num_scans_per_sweep*self.NFFT, self.NFFT)):
            self.sdr.fc += self.sdr.rs*scan_num

            # estimate PSD for one scan
            samples = self.sdr.read_samples(self.num_samples_per_scan)
            psd_scan, f, _ = opsd(samples, sample_rate=2, nfft=self.NFFT, sides='twosided')
            # psd_scan, f = psd(samples, NFFT=self.NFFT)
            self.image_buffer[0, start_ind: start_ind+self.NFFT] = psd_scan #np.log10(psd_scan)
            # print(self.image_buffer)
        # plot entire sweep
        self.image.set_array(self.image_buffer)

        # restore original center freq.
        self.sdr.fc = start_fc

        return self.image,
    
    def on_scroll(self, event):
        if event.button == 'up':
            self.sdr.fc += self.freq_inc_fine if self.shift_key_down else self.freq_inc_coarse
            self.update_plot_labels()
        elif event.button == 'down':
            self.sdr.fc -= self.freq_inc_fine if self.shift_key_down else self.freq_inc_coarse
            self.update_plot_labels()

    def on_key_press(self, event):
        if event.key == '+':
            self.sdr.gain += self.gain_inc
        elif event.key == '-':
            self.sdr.gain -= self.gain_inc
        elif event.key == ' ':
            self.sdr.gain = 'auto'
        elif event.key == 'shift':
            self.shift_key_down = True
        elif event.key == 'right':
            self.sdr.fc += self.freq_inc_fine if self.shift_key_down else self.freq_inc_coarse
            self.update_plot_labels()
        elif event.key == 'left':
            self.sdr.fc -= self.freq_inc_fine if self.shift_key_down else self.freq_inc_coarse
            self.update_plot_labels()
        elif event.key == 'enter':
            # see if valid frequency was entered, then change center frequency

            try:
                # join individual key presses into a string
                input = ''.join(self.keyboard_buffer)

                # if we're doing multiple adjacent scans, we need to figure out
                # the appropriate center freq for the leftmost scan
                center_freq = float(input)*1e6 + (self.sdr.rs/2)*(1 - self.num_scans_per_sweep)
                self.sdr.fc = center_freq

                self.update_plot_labels()
            except ValueError:
                pass

            self.keyboard_buffer = []

        else:
            self.keyboard_buffer.append(event.key)
        
    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_key_down = False
    
    def start(self):
        self.update_plot_labels()
        ani = animation.FuncAnimation(self.fig, self.update, interval=50,
                blit=True)

        pyl.show()

        return
