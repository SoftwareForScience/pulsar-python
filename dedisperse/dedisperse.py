'''
    Dedisperses data
'''
# pylint: disable-msg=C0103
import numpy as np

def dedisperse(samples, dm=None):
    '''
    This method performs dedispersion on the filterbank data
    '''

    if dm is None:
        print("Finding possible DM's")
        dm = find_initial_line(samples)

    # Distribute the DM over the amount of samples
    delays_per_sample = np.round(np.linspace(dm, 0, samples.shape[1])).astype(int)

    # Loop over the frequencies
    for i, _ in enumerate(delays_per_sample):

        # Temporary array that is used to later delay the frequency
        temporary_samples = []

        # Select frequency/column 'i' of all samples
        temporary_samples = samples[:, i]

        # Write back the frequency/column 'i' to the samples, using numpy's roll function
        samples[:, i] = np.roll(temporary_samples, delays_per_sample[i])

    return samples


def estimate_dm(samples):
    '''
    This method attempts to find a dispersion measure
    '''

    # Tuple of frequency index and sample index
    initial_signal_point = find_initial_signal(samples)
    last_sample = initial_signal_point

    for i, frequency in enumerate(samples[1]):
        for j, data_point in enumerate(samples[:, i]):
            #print(samples[:, i][previous_time_sample:].shape[0])
            if(j > last_sample[1]):
                if(data_point > 10):
                    last_sample = i, j
                    print("At Frequency ", i, " found on Time sample ", j, " - ", data_point)
                    break

    highest_difference = 0
    
    estimated_dm = last_sample[1] - initial_signal_point[1]
    print("Estimated DM is", estimated_dm)
    return estimated_dm


def find_initial_signal(samples):
    '''
    This method attempts to find a viable data point to start estimating a dispersion measure from
    '''

    # Tuple of frequency index and sample index
    lowest_sample = 0, samples.shape[0]
    
    for i, frequency in enumerate(samples[1]):
        for j, data_point in enumerate(samples[:, i]):
            if(j < lowest_sample[1]):
                if(data_point > 5):
                    print("Initial signal found on freq, sample", i, j, data_point)
                    return i, j
    
    print("NO INITIAL SIGNAL FOUND")
    return None
    

def find_initial_line(samples):
    '''
    This method attempts to find a line to dedisperse
    '''
    
    avg_intensity = find_avg_intensity(samples, 10)
    max_delay = 10
    
    for s, sample in enumerate(samples[:, 0]):
        if(sample > avg_intensity):
            start_sample_index = s
            print("Attempting to find line on freq,", 0, "sample", s)
            line_coordinates = find_line(samples, start_sample_index, max_delay, avg_intensity)
            
            # If a line is found, calculate and return the dispersion measure
            if(line_coordinates is not None):
                dm = line_coordinates[1] - line_coordinates[0]
                print(dm)
                return dm
            
    return None


def find_line(samples, start_sample_index, max_delay, avg_intensity):
    '''
    This method tries to find a line starting from the sample index given in the parameters
    it stops if there is no intensity within the max_delay higher than the avg_intensity
    '''

    previous_sample_index = start_sample_index
    failed_to_find_line = True

    # Loop through the frequencies
    for f, frequency in enumerate(samples[1]):

        # Loop through previous intensity until the max delay is reached
        for i, intensity in enumerate(samples[:, f][previous_sample_index:previous_sample_index + max_delay]):

            # Skip the first frequency, since that is where the initial sample is we are measuring from
            if(f == 0):
                failed_to_find_line = False
                break

            # If the intensity is higher than the avg_intensity, continue finding a signal
            if(intensity > avg_intensity):
                print("Continuing to find line on freq,", f, "sample", previous_sample_index + i, intensity)
                previous_sample_index = previous_sample_index + i
                failed_to_find_line = False
                break

        # If there is no line found, return None
        if failed_to_find_line: 
            return None

    # If all frequencies are looped through, a line is found, so we return the start and end index of the line
    return start_sample_index, previous_sample_index
            

def find_avg_intensity(samples, top = 10):
    '''
    This method finds the average intensity for top x intensities
    The avg_intensity is considered a requirement for intensities to be considered a pulsar 
    '''

    sum_intensities = 0

    # Looks for the top x highest intensities in the samples and adds them up together
    for sample in samples:
        sum_intensities += np.sum(sorted(sample, reverse=True)[:top])

    # Calculates the avg_intensity
    avg_intensity = (sum_intensities) / (samples.shape[0] * top)

    return avg_intensity