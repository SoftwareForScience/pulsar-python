# 7. Dedispersion

## 7.1 Introduction

- Pulsars produce a narrow beam of electromagnetic
radiation which rotates like a lighthouse beam, so
a pulse is seen as it sweeps over a radiotelescope.
The signal is spread over a wide frequency range.

- If space was an empty vacuum, all the signals would
travel at the same speed, but due to free electrons
different frequencies travel at slightly different speeds
(dispersion), this can be corrected by performing dedispersion.

## 7.2 Dedisperse

This method performs dedispersion on the filterbank data.

### 7.2.1 Parameters

| Parameters | Description |
|---|---|
| Samples | Array or sequence containing the intensities to be dedispersed. |
| highest_x | Specifies the amount of intensities that are used for estimating the minimum pulsar intensity |
| max_delay | Specifies the maximum allowed amount of samples between the currently considered pulsar signal and the next pulsar signal |
| DM | Dispersion measure (cm<sup>-3</sup> pc) |

### 7.2.2 Returns

| Variable | Description |
|---|---|
| Samples | Dedispersed samples |

## 7.3 find_dm

This method finds the dispersion measure.

### 7.3.1 Parameters

| Parameters | Description |
|---|---|
| Samples | Array or sequence containing the intensities to be dedispersed. |
| pulsar_intensity | Estimation of the minimum intensity for an intensity to be considered a pulsar |
| max_delay | Specifies the maximum allowed amount of samples between the currently considered pulsar signal and the next pulsar signal |

### 7.3.2 Returns

| Variable | Description |
|---|---|
| dm | Dispersion measure |

## 7.4 find_line

This method will attempt to find a continuous signal starting from the sample index given in the parameters. This method will stop if there isn't an intensity found within the max_delay higher than the pulsar_intensity.

### 7.4.1 Parameters

| Parameters | Description |
|---|---|
| Samples | Array or sequence containing the intensities to be dedispered. |
| start_sample_index | The index from which this method will continue to attempt and find a continuous pulsar signal |
| max_delay | Specifies the maximum allowed amount of samples between the currently considered pulsar signal and the next pulsar signal |
| pulsar_intensity | Estimation of the minimum intensity for an intensity to be considered a pulsar |

### 7.4.2 Returns

| Variable | Description |
|---|---|
| start_sample_index | The first frequency of the continuous signal found |
| previous_index | The last frequency of the continuous signal found |

## 7.5 find_estimation_intensity

This method finds the average intensity for the highest x intensities.
The `average_intensity` is considered a requirement for intensities to be considered a pulsar.

### 7.5.1 Parameters

| Parameters | Description |
|---|---|
| Samples | Array or sequence containing the intensities to be dedispersed. |
| max_delay | Specifies the maximum allowed amount of samples between the currently considered pulsar signal and the next pulsar signal |

### 7.5.2 Returns

| Variable | Description |
|---|---|
| average_intensity | Estimation of the minimum intensity for an intensity to be considered a pulsar |

[Back to table of contents](../README.md)