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
| data_type | type of file `filterbank, time series` |
| fch1 | center frequency of first filterbank channel (MHz) |
| foff | filterbank channel bandwidth (MHz) |
| nchans | number of filterbank channels |
| nbits | number of bits per time sample `8, 16 or 32` |
| tstart | timestamp of first sample (MJD) |
| nifs | number of seperate intermediate-frequency channels |

freq_range = fch1 + (foff * nchans + fch1)  
time_range = tstart + (tsamp/24/60/60)

freq_range is a tuple with a frequency start and a frequency stop  
The same applies to time_range

## 2.3 Read filterbank file
The attributes time_range and freq_range can be passed as parameters to select a specific portion of the filterbank file, and the stream parameter is a boolean for whether to read the data as a stream, for example:  
```
filterbank = Filterbank(<PATH TO FILTERBANK FILE>, freq_range, time_range, stream)
```

## 2.4 Select a range of data from the filterbank file
The select_data method can be used to retrieve data from the Filterbank object.
The user has the option to give a `time` and/or `frequency` range to select a subset from the entire dataset.
```
filterbank.select_data(freq_range, time_range)
```
The `select_data` method returns an array of all different channels/frequencies and a large matrix with all the received radio signals.

The matrix contains for each time sample an array which has the intensity per channel/frequency.

## 2.5 Read filterbank as stream
When reading the filterbank file as a stream, the user should set the `stream`-parameter to `True` when initializing the filterbank object.

Each time the user calls the `next_row` method, it will retrieve an array with intensitiy per frequency for a new time sample from the filterbank file.
When the last iteration of the filterbank is reached, the new_row method will return `True`

```
filterbank.next_row()
```

[Back to table of contents](../README.md)