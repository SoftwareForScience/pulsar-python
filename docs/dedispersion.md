# 7. Dedispersion

## 7.1 Introduction

- Pulsars produce a narrow beam of electromagnetic
radiation which rotates like a lighthouse beam, so
a pulse is seen as it sweeps over a radiotelescope.
The signal is spread over a wide frequency range.

- If space was an empty vacuum, all the signals would
travel at the same speed, but due to free electrons
different frequencies travel at slightly different speeds
(dispersion).

## 7.2 Dedisperse

This method performs dedispersion on the filterbank data.

### 7.2.1 Parameters

| Parameters | Description |
|---|---|
| Samples | Array or sequence containing the data to be plotted. |
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
| Samples | 1-D array or sequence. Array or sequence containing the data to be plotted. |

### 7.3.2 Returns

| Variable | Description |
|---|---|
| dm | Dispersion measure |

## 7.4 find_line

This method finds a line starting from the sample index given in the parameters. This will stop if there isn't a intensity within the max_delay higher than the average_intensity.

### 7.4.1 Parameters

| Parameters | Description |
|---|---|
| Samples | 1-D array or sequence. Array or sequence containing the data to be plotted. |
| start_sample_index | The start sample index |
| max_delay | The max delay of the recieved signal |
| average_intensity | The average intensity |

### 7.4.2 Returns

| Variable | Description |
|---|---|
| start_sample_index | The first frequency |
| previous_sample_index | The higher frequency |

## 7.5 find_estimation_intensity

This method finds the average intensity for top x intensities.
The `average_intensity` is considered a requirement for intensities to be considered a pulsar.

### 7.5.1 Parameters

| Parameters | Description |
|---|---|
| Samples | Array or sequence containing the data to be plotted. |

### 7.5.2 Returns

| Variable | Description |
|---|---|
| average_intensity | The avarage intesity |