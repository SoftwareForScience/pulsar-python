# 2. Filterbank Tutorial

### 2.1 Create a filterbank object
```
filterbank = Filterbank(<PATH TO FILTERBANK FILE>)
```
This is an example without parameters, see 2.3 for an example with the parameters.
### 2.2 Read the header from filterbank data
```
filterbank.header
```

Header data contains the following:
```
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
```
freq_range = fch1 + nchans  
time_range = tstart + (tsamp/24/60/60)

freq_range is a tuple with a frequency start and a frequency stop  
The same applies to time_range

### 2.3 Read filterbank file to 3d numpy array
The attributes time_range and freq_range can be passed as parameters to select a specific portion of the filterbank file, for example:  
```
filterbank = Filterbank(<PATH TO FILTERBANK FILE>, freq_range, time_range)
```

### 2.4 Select a range of data from the filterbank file
The select_data method can be used to select a subset from the data read by the read_filterbank method.
```
filterbank.select_data(freq_range, time_range)
```