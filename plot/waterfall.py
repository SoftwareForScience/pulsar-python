"""
    Module for plotting a waterfall plot
"""
import numpy as np
from plot import opsd
from fourier.fourier import fft_vectorized

class Waterfall():
    samples = None
    center_freq = None
    sample_freq = None
    freqs = None
    nfft = 1024
    samples_per_scan = nfft*16
    buffered_sweeps = 100
    scans_per_sweep = 1
    freq_inc_coarse = 1e6
    freq_inc_fine = 0.1e6
    gain_inc = 5
    shift_key_down = False
    keyboard_buffer = []
    plot = None
    image = None
    image_buffer = None
    iteration = 0

    def __init__(self, samples=None, center_freq=None, sample_freq=None, freqs=None,
                 fig=None, samples_per_scan=None,
                 buffered_sweeps=None, scans_per_sweep=None, freq_inc_coarse=None,
                 freq_inc_fine=None, gain_inc=None):
        """
            Setup waterfall object
        """

        if fig is None:
            raise ValueError("Need figure.")
        else:
            self.fig = fig

        if samples is None:
            raise ValueError("Expected sample data, but received none")
        else:
            self.samples = samples
        self.sample_freq = sample_freq #if sample_freq else None
        self.center_freq = center_freq #if center_freq else None
        self.samples_per_scan = samples_per_scan if samples_per_scan else self.samples_per_scan
        self.buffered_sweeps = buffered_sweeps if buffered_sweeps else self.buffered_sweeps
        self.scans_per_sweep = scans_per_sweep if scans_per_sweep else self.scans_per_sweep
        self.freq_inc_coarse = freq_inc_coarse if freq_inc_coarse else self.freq_inc_coarse
        self.freq_inc_fine = freq_inc_fine if freq_inc_fine else self.freq_inc_fine
        self.gain_inc = gain_inc if gain_inc else self.gain_inc
        self.freqs = np.asarray(freqs) #if freqs not None else None

        self.init_plot()

    def init_plot(self):
        # self.image_buffer = -100*np.ones((self.buffered_sweeps,\
        #                          self.scans_per_sweep*self.nfft))

        self.image_buffer = -100*np.ones(self.samples.shape)
        self.plot = self.fig.add_subplot(1, 1, 1)
        self.image = self.plot.imshow(self.image_buffer, aspect='auto',\
                                    interpolation='nearest', vmin=-50, vmax=10)
        self.plot.set_xlabel('Frequency (MHz)')
        self.plot.get_yaxis().set_visible(False)

        # self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        # self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        # self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

    def update_plot_labels(self):
        center_freq = self.center_freq
        sample_freq = self.sample_freq
        if self.freqs is not None:
            freq_range = self.freqs.min(), self.freqs.max()
        else:
            freq_range = ((center_freq - sample_freq/2)/1e6,\
                          (center_freq + sample_freq*(self.scans_per_sweep - 0.5))/1e6)
        print(self.image)
        self.image.set_extent(freq_range + (0, 1))
        self.fig.canvas.draw_idle()

    def get_row(self):
        self.iteration += 1
        row, _, _ = opsd(self.samples[self.iteration -1], nfft=128, sides='twosided')
        return row

    def get_image(self):
        self.update_plot_labels()
        i=0
        rows = np.ndarray(self.samples.shape, dtype=float)
        for row in self.samples:
            rows[i], _, _ = opsd(self.samples[i], nfft=128, sides='twosided')
            i+=1

        self.image.set_array(rows)
        return self.image

    def update(self, *args):
        # prepare space in buffer
        # TODO: use indexing to avoid recreating buffer each time
        self.image_buffer = np.roll(self.image_buffer, 1, axis=0)

        # for row in self.samples:
        # self.image_buffer[0] = 10*np.log10(self.get_row()/self.center_freq)
        self.image_buffer[0] = self.get_row()
        self.image.set_array(self.image_buffer)

        return self.image,

    def animated_plotter(self):
        self.update_plot_labels()

        return(self.fig, self.update, 50, True)