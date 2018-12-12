# 6. Clipping

## 6.1 Clipping

The `clipping.clipping` function can be used to remove noise from the data.

It combines all the individual methods of the clipping module to do this. The individual methods are described below in order.

```
clipping(<FREQUENCY_CHANNELS>, <TIME_SAMPLES>)
```

## 6.2 Filter samples

The `clipping.filter_samples` function can be used to remove entire time samples, that have noise in them.

The noise for a time sample is calculated by summing all the individual frequencies for one sample,
and then comparing it with the average(mean) of all samples. If the intensity of a sample is lower than
`average intensity per sample * factor`, the sample will be added back to the array. Otherwise, it's removed.

It is recommended to give only the first 2000 samples to this method, because adding more will only hurt the performance.

```
filter_samples(<TIME_SAMPLES>)
```

## 6.3 Filter channels

The `clipping.filter_channels` function can be used to remove entire channels/frequencies with noise from the data.

The noise for a frequency channel is identified by calculating the mean and standard deviation for each column in the data. After that, samples are removed if the mean or standard deviation of the intensity is higher than the `average intensity per channel * factor`.
The channels with noise are removed from both the list with channels, as well as the entire dataset.

```
filter_channels(<FREQUENCY_CHANNELS>, <TIME_SAMPLES>)
```

## 6.4 Filter individual channels

The `clipping.filter_indv_channels` function can be used to replace all the remaining samples with noise.

The noise for each individual sample is identified by calculating the mean intensity per channel. After that, samples are
replaced with the median of a channel if their intensity is higher than the `average intensity per channel * factor`.

```
filter_indv_channels(<TIME_SAMPLES>)
```

[Back to table of contents](../README.md)