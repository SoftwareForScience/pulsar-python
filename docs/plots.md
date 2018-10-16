# Plotting diagrams
## PSD
The `plot.psd` function can be used to generate the data for a Power Spectral Density plot. 

### Parameters

| Parameter | Description |
| --- | --- |
| samples | 1-D array or sequence. Array or sequence containing the data to be plotted. |
| nfft | Integer, optional. The number of bins to be used. Defaults to 256. |
| sample_rate | Integer, optional. The sample rate of the input data in `samples`. Defaults to 2.|
| window | Callable, optional. The window function to be used. Defaults to `plot.window_hanning`. |
| sides | {'default', 'onesided', 'twosided'}. Specifies which sides of the spectrum to return. Default gives the default behavior, which returns one-sided for real data and both for complex data. 'onesided' forces the return of a one-sided spectrum, while 'twosided' forces two-sided. |

### Returns
| Variable | Description |
| --- | --- |
| Pxx | 1-D array. The values for the power spectrum before scaling (real valued). |
| freqs | 1-D array. The frequencies corresponding to the elements in `Pxx`. |

### Example Usage
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