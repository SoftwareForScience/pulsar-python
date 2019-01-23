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