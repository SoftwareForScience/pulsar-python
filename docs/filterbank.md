# 2. Filterbank Tutorial

## 2.1 Create a filterbank object
```
filterbank = Filterbank(<PATH TO FILTERBANK FILE>)
```
This is an example without parameters, see 2.3 for an example with the parameters.
## 2.2 Read the header from filterbank data
```
filterbank.header
```

Header data contains the following:
| Variable | Description |
| --- | --- |
| source_name | name of filterbank file |
| P | period of pulsar in ms |
| DM | dispersion measure of pulsar |
| machine_id | id of machine used to receive signal data |
| telescope_id | id of telescope used to receive signal data |
| data_type | type of file |
| fch1 | center frequency of first filterbank channel (MHz) |
| foff | filterbank channel bandwidth (MHz) |
| nchans | number of filterbank channels |
| nbits | number of bits per time sample |
| tstart | timestamp of first sample (MJD) |
| nifs | number of seperate IF channels |

freq_range = fch1 + (foff * nchans + fch1)  
time_range = tstart + (tsamp/24/60/60)

freq_range is a tuple with a frequency start and a frequency stop  
The same applies to time_range

## 2.3 Read filterbank file to 3d numpy array
The attributes time_range and freq_range can be passed as parameters to select a specific portion of the filterbank file, for example:  
```
filterbank = Filterbank(<PATH TO FILTERBANK FILE>, freq_range, time_range)
```

## 2.4 Select a range of data from the filterbank file
The select_data method can be used to select a subset from the data read by the read_filterbank method.
```
filterbank.select_data(freq_range, time_range)
```

[Back to table of contents](../README.md)