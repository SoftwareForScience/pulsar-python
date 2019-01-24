# 8. Pipeline

## 8.1 Introduction

The Pipeline module is used to execute the different modules in a specific order.
There are currently three different options for running the pipeline.  

These options include:
* read multiple rows, `read_n_rows`
* read single rows, `read_rows`
* read all rows, `read_static`

The constructor of the pipeline module will recognize which method is fit for running which method, by looking at the given arguments to the constructor.

| Parameter | Description |
| --- | --- |
| filename | The path to the filterbank file. |
| as_stream | This parameter decides whether the filterbank should be read as stream. |
| DM | The dispersion measure (DM) is used for performing dedispersion. |
| scale | The scale is used for performing downsampling the time series. |
| n | The `n` is the rowsize of chunks for reading the filterbank as stream. |
| size | The size parameter is used for deciding the size of the filterbank. |

After deciding which method to run for running the filterbank in a pipeline, it will measure the time it takes to run each method using `measure_method`. After running all the different methods, the constructor will append the results (a dictionary) to a txt file.

## 8.2 Read rows

The `read_rows` method reads the Filterbank data row per row. Because it only reads the filterbank per row, it is unable to execute most methods. The alternative for this method is the `read_n_rows` method, which is able to run all methods.

```
pipeline.Pipeline(<filterbank_file>, as_stream=True)
```

## 8.3 Read n rows

The `read_n_rows` method first splits all the filterbank data into chunks of n samples. After splitting the filterbank data in chunks, it will run the different modules of the pipeline for each chunk. The remaining data, that which does not fit into the sample size, is currently ignored.

The `n` or sample size should be a power of 2 multiplied with the given scale for the downsampling.

```
pipeline.Pipeline(<filterbank_file>, n=<size> , as_stream=True)
```

## 8.4 Read static

The `read_static` method reads the entire filterbank at once, and applies each method to the entire dataset. If the filterbank file is too large for running it in-memory, the alternative is using `read_n_rows`.

```
pipeline.Pipeline(<filterbank_file>)
```

## 8.5 Measure methods

The `measure_methods` is ran for each of the above methods, and calculates the time it takes to run each of the different methods. For each method it will create a key using the name of the method, and save the time it took to run the method as a value.
At the end, it will returns a dictionary with all the keys and values.

## 8.6 Overview of pipeline

Apart from the different modules described in the previous paragraphs, additional modules are required for this library to make detecting pulsar signals possible.
However, these additional modules have not been developed yet, and are required to be developed in the future. In this paragraph the additional
modules are listed and described.

Modules that are missing in the pipeline are highlighted using a `*`.

```
1. Read Filterbank as stream
2. Reduce RFI using clipping
3. Dedisperse radio signal
4. Transform dedispersed signal to TimeSeries
5. Run fast Fourier transformation on TimeSeries
6. * Identify and save birdies in file
7. * Perform Harmonic Summing
8. * Search and identify single and periodic signals
9. * Phase-fold remaining signals
10.* Do Transient searches
```

[Back to table of contents](../README.md)
