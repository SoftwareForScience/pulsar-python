# Asteria
[![Build Status](https://travis-ci.com/AUAS-Pulsar/Asteria.svg?branch=master)](https://travis-ci.com/AUAS-Pulsar/Asteria)
[![codecov](https://codecov.io/gh/AUAS-Pulsar/Asteria/branch/master/graph/badge.svg)](https://codecov.io/gh/AUAS-Pulsar/Asteria)


## Introduction

Creating a free and open source framework that contains the generic algorithms and file handling for astronomical data sets. This framework will be modular. Similar to OpenCV, wherein specific modules can be added and disabled depended on the needs of a project. This framework will be implemented in Python and C++.

## Installation

### Requirements

    * numpy
    * python 3.6

## Documentation
1. [Getting started](docs/gettingstarted.md)
    1. [Example filterbank files](docs/gettingstarted.md#11-example-filterbank-files)
    2. [Import](docs/gettingstarted.md#12-import)
2. [Filterbank tutorial](docs/filterbank.md)
    1. [Create filterbank object](docs/filterbank.md#21-create-a-filterbank-object)
    2. [Read header from filterbank data](docs/filterbank.md#22-read-the-header-from-filterbank-data)
    3. [Read filterbank file](docs/filterbank.md#23-read-filterbank-file)
    4. [Select a range of data from the filterbank file](docs/filterbank.md#24-select-a-range-of-data-from-the-filterbank-file)
    5. [Read filterbank file as stream](docs/filterbank.md#25-read-filterbank-file-as-stream)
3. [Fourier transformation tutorial](docs/fourier.md)
    1. [DFT](docs/fourier.md#31-dft)
        1. [Parameters](docs/fourier.md#311-parameters)
        2. [Example usage](docs/fourier.md#312-example-usage)
    2. [FFT](docs/fourier.md#32-fft)
        1. [Parameters](docs/fourier.md#321-parameters)
        2. [Example usage](docs/fourier.md#322-example-usage)
    3. [IFFT](docs/fourier.md#33-ifft)
        1. [Parameters](docs/fourier.md#331-parameters)
        2. [Example usage](docs/fourier.md#332-example-usage)
4. [Plot tutorial](docs/plots.md)
    1. [PSD](docs/plots.md#41-psd)
        1. [Parameters](docs/plots.md#411-parameters)
        2. [Returns](docs/plots.md#412-returns)
        1. [Example usage](docs/plots.md#413-example-usage)
5. [Timeseries](docs/timeseries.md)
    1. [Initialize timeseries object](docs/timeseries.md#51-initialize-the-timeseries-object)
    2. [Retrieve timeseries](docs/timeseries.md#521-retrieve-timeseries-object)
    3. [Downsample timeseries](docs/timeseries.md#53-downsample-the-timeseries)
6. [Clipping](docs/clipping.md)
    1. [Clipping](docs/clipping.md#61-clipping)
    2. [Filter samples](docs/clipping.md#62-filter-samples)
    3. [Filter channels](docs/clipping.md#63-filter-channels)
    4. [Filter individual channels](docs/clipping.md#64-filter-individual-channels)
7. [Dedispersion](docs/dedispersion.md)
    1. [Dedisperse](docs/dedispersion.md#72-Dedisperse)
    2. [Find dispersion measure](docs/dedispersion.md#73-find_dm)
    3. [Find line](docs/dedispersion.md#74-find_line)
    4. [Find estimation intensity](docs/dedispersion.md#75-find_estimation_intensity)
8. [Pipeline](docs/pipeline.md)
    1. [Introduction](docs/pipeline.md#71-introduction)
    2. [Read rows](docs/pipeline.md#72-read-rows)
    3. [Read n rows](docs/pipeline.md#73-read-n-rows)
    3. [Read static](docs/pipeline.md#74-read-static)
    4. [Measure methods](docs/pipeline.md#75-measure-methods)
9. [Generating mock data](docs/generate.md)
    1. [Creating a filterbank file](docs/generate.md#81-creating-a-filterbank-file)
    2. [Generate signal](docs/generate.md#82-generate-signal)
    3. [Generate header](docs/generate.md#83-generate-header)
    4. [Keyword to string](docs/generate.md#84-keyword-to-string)
    5. [Write data](docs/generate.md#85-write-data)
