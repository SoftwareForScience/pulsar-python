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
                    print("At Frequency ", i, " found on Time sample ", j, " - ", data_point)
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
                if(data_point > 5):
                    print("Initial signal found on freq, sample", i, j, data_point)
                    return i, j
                    '''
                    print(lowest_sample, " ", j)
                    lowest_sample = i, j
                    break
                    '''

    print("NO INITIAL SIGNAL FOUND")
    return None
    

def find_initial_line(samples):
    '''
    This method attempts to find a line to dedisperse
    '''
    
    avg_intensity = find_avg_intensity(samples, 10)
    max_delay = 8
    
    for s, sample in enumerate(samples[:, 1]):
        if(sample > avg_intensity):
            previous_sample_index = s
            print("Attempting to find line on freq,", 1, "sample", s)
            find_line(samples, previous_sample_index, max_delay, avg_intensity)
            

    print("NO INITIAL SIGNAL FOUND")
    return None


def find_line(samples, previous_sample_index, max_delay, avg_intensity):
    for f, frequency in enumerate(samples[1]):
        for i, intensity in enumerate(samples[:, f][previous_sample_index:previous_sample_index+max_delay]):
            if(intensity > avg_intensity):
                print("Continuing to find line on freq,", f, "sample", i, intensity)
                previous_sample_index = i
                break
            else:
                continue
                
 

def find_avg_intensity(samples, top = 10):
    '''
    Finds average intensity for top x intensities
    '''

    sum_intensities = 0
    # Looks for the 3 highest intensities in the first 10 samples
    for sample in samples:
        #top_samples.append((sorted([(x,i) for (i,x) in enumerate(sample)], reverse=True)[:3] ))
        sum_intensities += np.sum(sorted(sample, reverse=True)[:top])

    avg_intensity = (sum_intensities) / (samples.shape[0] * top)

    return (avg_intensity)