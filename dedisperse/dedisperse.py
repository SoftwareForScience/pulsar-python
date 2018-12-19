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
        dm = estimate_dm(samples)

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
                    #print("At Frequency ", i, " found on Time sample ", j, " - ", data_point)
                    break

    highest_difference = 0
    '''
    for s, samples in enumerate(samples):
        for i, data_point in enumerate(samples):
            if(i > highest_difference):                
                if(data_point > 10):
                    print(s, i, " - ", data_point)                                        
                    highest_difference = i
                break
    '''
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
                if(data_point > 1):
                    print("Initial signal found on freq, sample", i, j)
                    return i, j
                    '''
                    print(lowest_sample, " ", j)
                    print("At Frequency ", i, " found on Time sample ", j, " - ", data_point)
                    lowest_sample = i, j
                    break
            if(lowest_sample[1] == 0):
                print("", lowest_sample)
                return i, j
                    '''

    print("NO INITIAL SIGNAL FOUND")
    return None
    
