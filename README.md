# Asteria
[![Build Status](https://travis-ci.com/AUAS-Pulsar/Asteria.svg?branch=master)](https://travis-ci.com/AUAS-Pulsar/Asteria)
[![codecov](https://codecov.io/gh/AUAS-Pulsar/Asteria/branch/master/graph/badge.svg)](https://codecov.io/gh/AUAS-Pulsar/Asteria)


## Introduction

Creating a free and open source framework that contains the generic algorithms and file handling for astronomical data sets. This framework will be modular. Similar to OpenCV, wherein specific modules can be added and disabled depended on the needs of a project. This framework will be implemented in Python and C++.

## Installation

### Requirements

    * numpy
    * python 3.6


# Getting started

## Example filterbank files
### Use one of the following filterbank files as an example:
* <a href="https://git.dev.ti-more.net/uploads/-/system/personal_snippet/2/bc063035797e978034adfb6f2da75e70/pspm8.fil">8 bit</a>
* <a href="https://git.dev.ti-more.net/uploads/-/system/personal_snippet/2/bc063035797e978034adfb6f2da75e70/pspm16.fil">16 bit</a>
* <a href="https://git.dev.ti-more.net/uploads/-/system/personal_snippet/2/bc063035797e978034adfb6f2da75e70/pspm32.fil">32 bit</a>

## Import
> ```from filterbank.filterbank import *```

# Tutorial

### Create a filterbank object
> ``` filterbank = Filterbank(<PATH TO FILTERBANK FILE>) ```

### Read the header from filterbank data
> ``` filterbank.header ```

### Read filterbank file to 3d numpy array
> ``` filterbank.read_filterbank ```

### Calculate the frequency range
> ``` filterbank.setup_freqs ```

### Calculate the time range
> ``` filterbank.setup_freqs ```