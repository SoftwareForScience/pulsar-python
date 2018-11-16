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
        1. [Parameters](docs/fourier.md#321-paramters)
        2. [Example usage](docs/fourier.md#322-example-usage)
4. [Plot tutorial](docs/plots.md)
    1. [PSD](docs/plots.md#41-psd)
        1. [Parameters](docs/plots.md#411-parameters)
        2. [Returns](docs/plots.md#412-returns)
        1. [Example usage](docs/plots.md#413-example-usage)
