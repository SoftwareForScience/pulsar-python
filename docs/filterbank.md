# 2. Filterbank Tutorial

### 2.1 Create a filterbank object
```
filterbank = Filterbank(<PATH TO FILTERBANK FILE>)
```

### 2.2 Read the header from filterbank data
```
filterbank.header
```

Header data contains the following:
* source_name 
* DM
* machine_id
* telescope_id
* data_type
* fch1
* foff
* nchans
* nbits
* tstart
* tsamp
* nifs

freq_range = fch1 + nchans  
time_range = tstart + (tsamp/24/60/60)

freq_range is a tuple with a frequency start and a frequency stop  
The same applies to time_range

### Read filterbank file to 3d numpy array
```
filterbank.read_filterbank 
```
time_range and freq_range can be passed as parameters to select a specific portion of the data, for example:  
```
filterbank.read_filterbank(freq_range, time_range)
```

### Select a range of data from the filterbank file
```
filterbank.select_data
```
time_range and freq_range can be passed as parameters to select a specific portion of the data, for example:  
```
filterbank.read_filterbank(freq_range, time_range)
```