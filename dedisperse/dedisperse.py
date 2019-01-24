'''
Dedisperses data
'''
# pylint: disable-msg=C0103
import numpy as np

def dedisperse(samples, highest_x=None, max_delay=0, disperion_measure=None):
    '''
    This method performs dedispersion on the filterbank data
    The maximum_delay specifies between the currently considered pulsar signal and the next pulsar
    signal should be
    The highest_x specifies the amount of intensities that are used for estimating the minimum
    pulsar intensity
    '''

    # Check if parameters contain a Dispersion Measure, if not, estimate one
    if disperion_measure is None:
        # Estimates the minimum for an intensity to be considered a pulsar
        pulsar_intensity = find_estimation_intensity(samples, highest_x)
        disperion_measure = find_dm(samples, pulsar_intensity, max_delay)

    # Distribute the Dispersion Measure over the amount of samples
    delays_per_sample = np.round(np.linspace(disperion_measure, 0, samples.shape[1])).astype(int)

    # Loop over the frequencies
    for i, _ in enumerate(delays_per_sample):

        # Temporary array that is used to later delay the frequency
        temporary_samples = []

        # Select frequency/column 'i' of all samples
        temporary_samples = samples[:, i]

        # Write back the frequency/column 'i' to the samples, using numpy's roll function
        samples[:, i] = np.roll(temporary_samples, delays_per_sample[i])

    return samples

def find_dm(samples, pulsar_intensity, max_delay):
    '''
    This method attempts to find a dispersion measure
    '''

    # Loop through the samples to find a pulsar intensity to start calculating from
    for s, sample in enumerate(samples[:, 0]):

        # If the sample meets the minimum intensity, attempt to find a line continuing from
        # this intensity
        if sample > pulsar_intensity:
            start_sample_index = s

            # Attempt to find a line, line_coordinates contains first and last index of the pulsar
            line_coordinates = find_line(samples, start_sample_index, max_delay, pulsar_intensity)

            # If a line is found, calculate and return the dispersion measure
            if line_coordinates is not None:
                disperion_measure = line_coordinates[1] - line_coordinates[0]
                return disperion_measure

    return None


def find_line(samples, start_sample_index, max_delay, pulsar_intensity):
    '''
    This method tries to find a line starting from the sample index given in the parameters
    it stops if there is no intensity within the max_delay higher than the average_intensity
    '''

    previous_index = start_sample_index

    failed_to_find_line = True

    # Loop through the frequencies
    for f, _ in enumerate(samples[1]):

        # Loop through previous intensity until the max delay is reached
        for i, intensity in enumerate(samples[:, f][previous_index:previous_index + max_delay]):

            # Skip the first frequency, since we start measuring from the first intensity
            if f == 0:
                failed_to_find_line = False
                break

            # If the intensity is higher than the pulsar_intensity, continue finding a signal
            if intensity > pulsar_intensity:
                previous_index = previous_index + i
                failed_to_find_line = False
                break

        # If there is no line found, return None
        if failed_to_find_line:
            return None

    # If all frequencies are looped, a continuous signal is found,
    # so we return the first and last index of the line
    return start_sample_index, previous_index

def find_estimation_intensity(samples, highest_x):
    '''
    This method finds the average intensity for the highest x intensities
    The average_intensity is considered a requirement for intensities to be considered a pulsar
    '''

    # Sum of all intensities
    sum_intensities = 0

    # Looks for the top x highest intensities in the samples and adds them up together
    for sample in samples:
        sum_intensities += np.sum(sorted(sample, reverse=True)[:highest_x])

    # Calculates the average_intensity
    average_intensity = (sum_intensities) / (samples.shape[0] * highest_x)

    return average_intensity
