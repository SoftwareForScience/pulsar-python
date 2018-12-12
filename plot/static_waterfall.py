'''
    Plots the data in a waterfall plot
'''
import numpy as np

def waterfall_plot(data, frequencies):
    '''
    This method expects the data to be in the filterbank format
    '''

    # Reverse the data
    data = data[..., ::-1]
    frequencies = frequencies[::-1]

    # Calculate the frequency axis
    plot_f_begin = frequencies[0]
    plot_f_end = frequencies[-1] + (frequencies[1]-frequencies[0])

    # Calculate the timestamp axis
    n_samples = data.shape[0]
    t_delt = 8e-05
    t_0 = 50000

    timestamps = np.arange(0, n_samples) * t_delt / 24. / 60. / 60. + t_0

    plot_t_begin = timestamps[0]
    plot_t_end = timestamps[-1] + (timestamps[1] - timestamps[0])

    # Put axes together into one variable
    extent = (0.0, (plot_t_end-plot_t_begin)*24.*60.*60, plot_f_begin, plot_f_end)

    return data, extent
