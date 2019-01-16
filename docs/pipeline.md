# 7. Pipeline

## 7.1 Introduction

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

After deciding which method to run for running the filterbank in a pipeline, it will measure the time it takes to run each method. At the end it will append the results to a txt file.

## 7.2 Read rows

The `read_rows` method reads the Filterbank data row per row. Because it only reads the filterbank per row, it is unable to execute most methods. The alternative for this method is the `read_n_rows` method, which is able to run all methods.

```
pipeline.Pipeline(<filterbank_file>, as_stream=True)
```

## 7.3 Read n rows

The `read_n_rows` method first splits all the filterbank data into chunks of n samples. After splitting the filterbank data in chunks, it will run the different modules of the pipeline for each chunk.

```
pipeline.Pipeline(<filterbank_file>, n=<size> , as_stream=True)
```

## 7.4 Read static

The `read_static` method reads the entire filterbank at once. If the filterbank file is too large for this method, the alternative is using `read_n_rows`.

```
pipeline.Pipeline(<filterbank_file>)
```

## 7.5 Measure methods

The `measure_methods` is ran for each of the above methods, and calculates the time it takes to run each of the different methods.
