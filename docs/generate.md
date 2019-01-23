# 8. Generating mock data

## 8.1 Creating a filterbank file

To generate a filterbank file you may use the following example code:
```python
import filterbank.header as header

header = {
    b'source_name': b'P: 80.0000 ms, DM: 200.000',
    b'machine_id': 10,
    b'telescope_id': 4,
    b'data_type': 1,
    b'fch1': 400,
    b'foff': -0.062,
    b'nchans': 128,
    b'tstart': 6000.0,
    b'tsamp': 8e-05,
    b'nifs': 1,
    b'nbits': 8
}

header.generate_file("file_path/file_name", header)
```

The header data is passed in a header dict. For a more detailed explanation on the file header, please consult [chapter 2.2](docs/filterbank.md#22-read-the-header-from-filterbank-data). 

## 8.2 Generate signal

The generate_signal method generates a mock signal based on the following parameters:

| Variable | Description |
| --- | --- |
| noise_level | the max amplitude of the generated noise |
| period | period of the signal |
| t_obs | observation time in s |
| n_pts | intervals between samples |

## 8.3 Generate header

The generate_header method generates a header string based on the header dict provided in the example in chapter 8.1. The dict provides keys encoded in bytes and the method converts each keyword to a string using the keyword_to_string method (see below).

## 8.4 Keyword to string

The keyword_to_string method converts a keyword from the header dict to a serialized string.

## 8.5 Write data

Once all the required data is generated to a filterbank file. The data is written to the file as bytes.

[Back to table of contents](../README.md)