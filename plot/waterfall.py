"""
    Module for plotting a waterfall plot
"""
import numpy as np

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
    def __init__(self, filer_bank=None, center_freq=None, sample_freq=None,
                 fig=None, samples_per_scan=None,
                 buffered_sweeps=None, scans_per_sweep=None,
                 max_n_rows=1024, mode='stream', t_obs=None):
        """
            Setup waterfall object
        """
        if filer_bank is None:
            raise ValueError("A filterbank object input is needed to generate the plot.")
        else:
            self.filer_bank = filer_bank

        if fig is None:
            raise ValueError("Need figure.")
        else:
            self.fig = fig

        self.header = filer_bank.get_header()
        print(self.header)
        self.t_obs = t_obs if t_obs else 1
        print(self.t_obs)
        self.max_n_rows = max_n_rows

        self.sample_freq = sample_freq
        self.center_freq = center_freq
        self.samples_per_scan = samples_per_scan if samples_per_scan else self.samples_per_scan
        self.buffered_sweeps = buffered_sweeps if buffered_sweeps else self.buffered_sweeps
        self.scans_per_sweep = scans_per_sweep if scans_per_sweep else self.scans_per_sweep

        print(self.header[b'tsamp'])
        print(self.t_obs)
        print(self.t_obs / 8e-05)
        if mode == "discrete":
            time_start = 0
            time_stop = int(self.t_obs//self.header[b'tsamp'])
            freqs, self.samples = filer_bank.select_data(time_start=time_start, time_stop=time_stop)
            print(freqs.shape)
            print(self.samples.shape)
        else:
            freqs = filer_bank.get_freqs()
            self.samples = self.filer_bank.next_n_rows(self.max_n_rows)


        self.freqs = np.asarray(freqs)

        self.init_plot()

    def init_plot(self):
        """
            Initialize the plot
        """
        self.image_buffer = -100*np.ones(self.samples.shape)
        print(self.samples.shape)
        self.plot = self.fig.add_subplot(1, 1, 1)
        self.image = self.plot.imshow(self.image_buffer, aspect='auto',\
                                    interpolation='nearest', vmin=-50, vmax=10)
        self.plot.set_ylabel('Frequency (MHz)')

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
        """
            Returns the next row of data in the filterbank object.
        """
        return self.filer_bank.next_row()

    def get_image(self):
        """
            Returns the image data of the full dataset, if using a discrete dataset.
        """
        self.update_plot_labels()
        img = np.rot90(np.flip(self.samples, 0))
        self.image.set_array(img)
        return self.image

    def update(self, i):
        """
            Updates the image with the next row of data, when using
            a continuous datastream.
        """
        print('update:', i)
        # prepare space in buffer
        self.image_buffer = np.roll(self.image_buffer, 1, axis=0)

        self.image_buffer[0] = self.get_next()
        self.image.set_array(self.image_buffer)

        return self.image

    def animated_plotter(self):
        """
            Returns the figure and update function for the animation function.
        """
        self.update_plot_labels()

        return(self.fig, self.update, 4096, True)

    def get_center_freq(self):
        """
            returns the centerfrequency stored in the file header.
        """
        return self.header[b'center_freq']
