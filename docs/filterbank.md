# Filterbank Tutorial

### Create a filterbank object
> ``` filterbank = Filterbank(<PATH TO FILTERBANK FILE>) ```

### Read the header from filterbank data
> ``` filterbank.header ```

### Read filterbank file to 3d numpy array
> ``` filterbank.read_filterbank ```

### Calculate the frequency range
> ``` filterbank.setup_freqs ```

### Calculate the time range
> ``` filterbank.setup_time ```

### Calculate the channel range
> ``` filterbank.setup_chans ```

### Select a range of data from the filterbank file
> ``` filterbank.select_data ```