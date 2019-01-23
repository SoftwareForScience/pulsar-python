"""
    Module for plotting a waterfall plot
"""
import numpy as np
from .plot import opsd

class Waterfall():
    """
        Class for generating waterfall plots.
    """
    # pylint: disable=too-many-instance-attributes
    # All the attributes are needed.

    header = None
    samples = None
    sample_freq = None
    freqs = None
    nfft = 1024
    samples_per_scan = nfft*16
    buffered_sweeps = 100
    scans_per_sweep = 1
    plot = None
    image = None
    image_buffer = None

    # pylint: disable=R0913
    # All these attributes are needed.
    def __init__(self, fb=None, samples=None, center_freq=None, sample_freq=None,
                 fig=None, samples_per_scan=None,
                 buffered_sweeps=None, scans_per_sweep=None, freq_inc_coarse=None,
                 freq_inc_fine=None, gain_inc=None, max_n_rows=1024, mode='stream', t_obs=None):
        """
            Setup waterfall object
        """
        if fb is None:
            raise ValueError("A filterbank object input is needed to generate the plot.")
        else:
            self.fb = fb

        if fig is None:
            raise ValueError("Need figure.")
        else:
            self.fig = fig

        # if samples is None:
        #     raise ValueError("Expected sample data, but received none")
        # else:
        #     self.samples = samples

        self.header = fb.get_header()
        print(self.header)
        self.t_obs = t_obs if t_obs else 1
        print(self.t_obs)
        self.max_n_rows = max_n_rows

        self.sample_freq = sample_freq #if sample_freq else None
        self.center_freq = center_freq #if center_freq else None
        self.samples_per_scan = samples_per_scan if samples_per_scan else self.samples_per_scan
        self.buffered_sweeps = buffered_sweeps if buffered_sweeps else self.buffered_sweeps
        self.scans_per_sweep = scans_per_sweep if scans_per_sweep else self.scans_per_sweep
        # self.freq_inc_coarse = freq_inc_coarse if freq_inc_coarse else self.freq_inc_coarse
        # self.freq_inc_fine = freq_inc_fine if freq_inc_fine else self.freq_inc_fine
        # self.gain_inc = gain_inc if gain_inc else self.gain_inc

        print(self.header[b'tsamp'])
        print(self.t_obs)
        print(self.t_obs / 8e-05)
        if mode == "discrete":        
            # fb.read_filterbank()
            freqs, self.samples = fb.select_data(time_start=0, time_stop=int(self.t_obs//self.header[b'tsamp']))
            print(freqs.shape)
            print(self.samples.shape)
        else:
            freqs = fb.get_freqs()
            self.samples = self.fb.next_n_rows(self.max_n_rows)


        self.freqs = np.asarray(freqs) #if freqs not None else None

        self.init_plot()

    def init_plot(self):
        """
            Initialize the plot
        """
        # self.image_buffer = -100*np.ones((self.buffered_sweeps,\
        #                          self.scans_per_sweep*self.nfft))

        self.image_buffer = -100*np.ones(self.samples.shape)
        print(self.samples.shape)
        self.plot = self.fig.add_subplot(1, 1, 1)
        self.image = self.plot.imshow(self.image_buffer, aspect='auto',\
                                    interpolation='nearest', vmin=-50, vmax=10)
        self.plot.set_ylabel('Frequency (MHz)')
        # self.plot.get_yaxis().set_visible(True)

        # self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        # self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        # self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

    def update_plot_labels(self):
        """
            Set the plotlables.
        """
        center_freq = self.get_center_freq()
        sample_freq = self.sample_freq
        if self.freqs is not None:
            freq_range = self.freqs.min(), self.freqs.max()
        else:
            freq_range = ((center_freq - sample_freq/2)/1e6,\
                          (center_freq + sample_freq*(self.scans_per_sweep - 0.5))/1e6)

        self.image.set_extent((0, self.t_obs) + freq_range)
        self.fig.canvas.draw_idle()

    def get_next(self):
        return self.fb.next_row()

    # def get_row(self):
    #     """
    #         Returns the next row of data.
    #     """
    #     self.iteration += 1
    #     row, _, _ = opsd(self.samples[self.iteration -1], nfft=128, sides='twosided')
    #     return row

    def get_image(self):
        """
            Returns the image data of the full dataset, if using a discrete dataset.
        """
        self.update_plot_labels()
        img = np.rot90(np.flip(self.samples, 0))
        self.image.set_array(img)
        return self.image

    def update(self):
        """
            Updates the image with the next row of data, when using
            a continuous datastream.
        """
        # prepare space in buffer
        self.image_buffer = np.roll(self.image_buffer, 1, axis=0)

        # for row in self.samples:
        # self.image_buffer[0] = 10*np.log10(self.get_row()/self.center_freq)
        self.image_buffer[0] = self.get_next()
        self.image.set_array(self.image_buffer)

        return self.image

    def animated_plotter(self):
        """
            Returns the figure and update function for the animation function.
        """
        self.update_plot_labels()

        return(self.fig, self.update, 50, True)

    def get_center_freq(self):
        """
            returns the centerfrequency stored in the file header.
        """
        return self.header[b'center_freq']