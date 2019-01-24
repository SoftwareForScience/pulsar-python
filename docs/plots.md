# 4. Plotting diagrams

## 4.1 PSD
The `plot.psd` function can be used to generate the data for a Power Spectral Density plot. 

### 4.1.1 Parameters

| Parameter | Description |
| --- | --- |
| samples | 1-D array or sequence. Array or sequence containing the data to be plotted. |
| nfft | Integer, optional. The number of bins to be used. Defaults to 256. |
| sample_rate | Integer, optional. The sample rate of the input data in `samples`. Defaults to 2.|
| window | Callable, optional. The window function to be used. Defaults to `plot.window_hanning`. |
| sides | {'default', 'onesided', 'twosided'}. Specifies which sides of the spectrum to return. Default gives the default behavior, which returns one-sided for real data and both for complex data. 'onesided' forces the return of a one-sided spectrum, while 'twosided' forces two-sided. |

### 4.1.2 Returns
| Variable | Description |
| --- | --- |
| Pxx | 1-D array. The values for the power spectrum before scaling (real valued). |
| freqs | 1-D array. The frequencies corresponding to the elements in `Pxx`. |

### 4.1.3 Example Usage
```python
from filterbank.filterbank import Filterbank
import matplotlib.pyplot as plt
import numpy as np
from plot import psd
from filterbank.header import read_header

# Instatiate the filterbank reader and point to the filterbank file
fb = Filterbank(filename='examples/pspm32.fil')

# read the data in the filterbank file
_, samples = fb.select_data()

# Convert 2D Array to 1D Array with complex numbers
samples = samples[0] + (samples[1] * 1j)

# Read the header of the filterbank file
header = read_header('examples/pspm32.fil')

# Calculate the center frequency with the data in the header
center_freq = header[b'fch1'] + float(header[b'nchans']) * header[b'foff'] / 2.0

print(center_freq)
# Get the powerlevels and the frequencies
PXX, freqs, _ = psd(samples, nfft=1024, sample_rate=80, sides='twosided')

# Calculate the powerlevel dB's
power_levels =  10 * np.log10(PXX/(80))

# Add the center frequency to the frequencies so they match the actual frequencies
freqs = freqs + center_freq

# Plot the PSD
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (dB)')
plt.plot(freqs, power_levels)
plt.show()
```

## 4.2 Waterfall
The `plot.waterfall.Waterfall` class can be used to generate waterfall plots. 

### 4.2.1 Construction

| Parameter | Description |
| --- | --- |
| filter_bank | A `filterbank` object. |
| center_freq | The center frequency of the signal in the filterbank object |
| sample_freq | The sample frequency of the signal in the filterbank object|
| fig | An imaging object, like `pyplot.figure()` |
| mode | String `{discrete, stream}`. The mode to operate on. Use discrete for discrete datasets, and stream for stream data. Defaults to `stream`. |

### 4.2.2 Methods
| Method | Description | Parameters |
| --- | --- |
| init_plot(self) | Initialize the plot |
| update_plot_labes(self) | Generate the plot labels |
| get_next(self) | Returns the next row of data in the filterbank object |
| get_image(self) | Returns the image data of the full dataset, if using a discrete dataset. |
| update(self, i) | Updates the image with the next row of data, when using a continuous datastream. |
| animated_plotter(self) | Returns the figure and update function for matplotlib animation |
| get_center_freq(self) | Returns the center frequency stored in the filterbank header |

### 4.2.3 Example Usage
#### 4.2.3.1 With discrete data
```python
import matplotlib.animation as animation
from filterbank.header import read_header
from filterbank.filterbank import Filterbank
from plot import waterfall
import pylab as pyl
from plot.plot import next_power_of_2

fb = Filterbank(filename='./pspm32.fil', read_all=True)

wf = waterfall.Waterfall(filter_bank=fb, fig=pyl.figure(), mode="discrete")

img = wf.get_image()

pyl.show(img)
```

#### 4.2.3.2 With stream data
```python
import matplotlib.animation as animation
from filterbank.header import read_header
from filterbank.filterbank import Filterbank
from plot import waterfall
import pylab as pyl
from plot.plot import next_power_of_2


fb = Filterbank(filename='./pspm32.fil')

wf = waterfall.Waterfall(fb=fb, fig=pyl.figure(), mode="stream")

fig, update, frames, repeat = wf.animated_plotter()

ani = animation.FuncAnimation(fig, update, frames=frames,repeat=repeat)
pyl.show()
```


[Back to table of contents](../README.md)