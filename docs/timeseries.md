# 5. Timeseries module
Learn how to use the timeseries object. The timeseries object is used for all timeseries related operations both filterbank and non-filterbank (general array) types. 

## 5.1 Initialize the timeseries object

The ``Timeseries`` module can be used with both the filterbank file and a general ``numpy`` array. 

### 5.1.1 Initialize using a filterbank file

- Create an filterbank object using the standard filterbank module. 
- Read the filterbank file in memory (do not use streams)
- Initialize the timeseries object using the filterbank object.  

```python
import filterbank.filterbank as Filterbank
import timeseries.timeseries as Timeseries

# Read the filterbank file from a file. 
filterbank_obj = Filterbank('./psm32.fil')

# Read the filterbank as a whole instead of as a stream. 
filterbank_obj =  filterbank_obj.read_filterbank()

# Initialize the timeseries object. 
ts = Timeseries().from_filterbank(filterbank_obj)
```

### 5.1.2 Initialize using a numpy array
```python
import numpy as np
import timeseries.timeseries as Timeseries

input_array = np.array([1, 2, 3, 4, 5, 7, 9, 10, 11])

ts = Timeseries(input_array)

```

## 5.2 Retrieve the timeseries array from timeseries object

### 5.2.1 Retrieve timeseries object

```python
# Assumed that your timeseries object has been initialized. 

timeseries_array =  timeseries.get()

```

## 5.3 Downsample the timeseries
The first implemented feature for the timeseries object is the downsample/decimate function. This enables you to downsample your timeseries by `q` scale. This will make your input array smaller and basically 'cuts' the other parts off. 

### 5.3.1 Downsample/Decimate
Called downsample in the current release because no anti-alliasing is used, might be renamed to decimate once more advanced operations (such as antialiasing) are used. 

```python

    # Downsampled array shall be 3 times smaller than the current timeseries (as initialized) 
    scale = 3
    # Returns an array with the downsampled timeseries, can also be retreived lated user timseries.get()
    downsampled_array = timeseries.downsample(scale)
```
