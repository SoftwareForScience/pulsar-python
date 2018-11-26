"""
    Functions used to plot Power Spectral Density plots
"""
import numpy as np
from fourier import fourier

# pylint: disable-msg=R0913
# pylint: disable-msg=R0914


def window_hanning(window):
    '''
    Return window times the hanning window of len(window).

    See Also
    --------
    :func:`window_none`
        :func:`window_none` is another window algorithm.
    '''
    return np.hanning(len(window))*window

def next_power_of_2(val):
    """
        Return the next integer that is a power of two

        Params
        ------
        val : int
    """
    return int(2**(int(np.log2(val)) + 1))

def is_power_of_2(val):
    """
        Checks wheters the value is a power of two
    """
    return (val != 0) and ((val & (val -1)) == 0)


def psd(samples, sampling_frequency, center_frequency, nfft=None):
    """
        Plot the power spectral density.
    """
    # Length of signal
    nsamples = len(samples)

    # Number of FFT points
    if nfft is None:
        nfft = next_power_of_2(nsamples)

    if nfft is not is_power_of_2(nfft):
        nfft = next_power_of_2(nfft)

    if np.ndim(samples) > 1:
        samples = samples.reshape((samples.shape[0]*samples.shape[1],))

    # Frequency plotting vector
    freqs = sampling_frequency/2*(np.arange(-1, 1-2/nfft, 2/nfft, dtype=type(center_frequency)))
    freqs = np.asarray(freqs) + center_frequency
    ## Analysis

    # analyze spectrum
    indeces = np.arange(nfft/2-1, nfft-1, dtype=int)
    if np.ndim(samples) == 1:
        power = fourier.fft_shift(fourier.fft_vectorized(samples, nfft))
        # power = fourier.fftshift(power)
    elif np.ndim(samples) == 2:
        samples = samples.reshape((samples.shape[0]*samples.shape[1],))
        power = np.asarray(fourier.fft_vectorized(samples))
    else:
        raise ValueError("Only 1 and 2 dimensional sample arrays are supported right now.")

    power = np.abs(power)/np.sqrt(nsamples*sampling_frequency)

    # Convert to dBm/Mz
    power = 10*np.log10(power/center_frequency)

    freqs = [freqs[i] for i in indeces]
    power = [power[i] for i in indeces]

    return freqs, power

# pylint: disable=R0912
# All branches are needed
def opsd(samples, nfft=None, sample_rate=2, window=window_hanning, noverlap=0,
         detrend_func=None, pad_to=None, scale_by_freq=True, sides='twosided'):
    """
        Plot the power spectral density.
    """
    if nfft is None:
        nfft = len(samples)

    if pad_to is None:
        pad_to = nfft

    samples = np.asarray(samples)

    if len(samples) < nfft:
        sample_size = len(samples)
        samples = np.resize(samples, (nfft,))
        samples[sample_size:] = 0

    if sides == 'twosided':
        num_freqs = pad_to
        scaling_factor = 1.
        if pad_to % 2:
            freqcenter = (pad_to - 1)//2 + 1
        else:
            freqcenter = pad_to//2
    elif sides == 'onesided':
        scaling_factor = 2.
        if pad_to % 2:
            num_freqs = (pad_to + 1)//2
        else:
            num_freqs = pad_to//2 + 1

    result = stride_windows(samples, nfft, noverlap, axis=0)
    result = detrend(result, detrend_func, axis=0)
    result, window_vals = apply_window(result, window, axis=0,
                                       return_window=True)

    result = fourier.fft_vectorized(samples, nfft=nfft)
    freqs = fourier.fft_freq(pad_to, 1/sample_rate)[:num_freqs]

    result = np.conj(result) * result

    slc_stop = None
    if not nfft % 2:
        slc_stop = -1
    slc = slice(1, slc_stop, None)

    result[slc] *= scaling_factor

    if scale_by_freq:
        result /= sample_rate
        result /= (np.abs(window_vals)**2).sum()
    else:
        result /= np.abs(window_vals).sum()**2

    time = np.arange(nfft/2, len(samples) - nfft/2 + 1, nfft - noverlap)/sample_rate

    if sides == 'twosided':
        freqs = np.concatenate((freqs[freqcenter:], freqs[:freqcenter]))
        result = np.concatenate((result[freqcenter:],
                                 result[:freqcenter]), 0)
    elif not pad_to % 2:
        freqs[-1] *= -1

    result = 10*np.log10(result/freqcenter)
    # freqs = (freqs + freqcenter)/1e5

    return result, freqs, time


def detrend(samples, key=None, axis=None):
    '''
    Return samples with its trend removed.

    Parameters
    ----------
    x : array or sequence
        Array or sequence containing the data.

    key : [ 'default' | 'constant' | 'mean' | 'linear' | 'none'] or function
        Specifies the detrend algorithm to use. 'default' is 'mean', which is
        the same as :func:`detrend_mean`. 'constant' is the same. 'linear' is
        the same as :func:`detrend_linear`. 'none' is the same as
        :func:`detrend_none`. The default is 'mean'. See the corresponding
        functions for more details regarding the algorithms. Can also be a
        function that carries out the detrend operation.

    axis : integer
        The axis along which to do the detrending.

    See Also
    --------
    :func:`detrend_mean`
        :func:`detrend_mean` implements the 'mean' algorithm.

    :func:`detrend_linear`
        :func:`detrend_linear` implements the 'linear' algorithm.

    :func:`detrend_none`
        :func:`detrend_none` implements the 'none' algorithm.
    '''
    if key is None or key in ['constant', 'mean', 'default']:
        return detrend(samples, key=detrend_mean, axis=axis)
    if key == 'linear':
        return detrend(samples, key=detrend_linear, axis=axis)
    if key == 'none':
        return detrend(samples, key=detrend_none, axis=axis)
    if isinstance(key, str):
        raise ValueError("Unknown value for key %s, must be one of: "
                         "'default', 'constant', 'mean', "
                         "'linear', or a function" % key)
    if not callable(key):
        raise ValueError("Unknown value for key %s, must be one of: "
                         "'default', 'constant', 'mean', "
                         "'linear', or a function" % key)

    samples = np.asarray(samples)

    if axis is not None and axis+1 > samples.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    if (axis is None and samples.ndim == 0) or (not axis and samples.ndim == 1):
        return key(samples)

    # try to use the 'axis' argument if the function supports it,
    # otherwise use apply_along_axis to do it
    try:
        return key(samples, axis=axis)
    except TypeError:
        return np.apply_along_axis(key, axis=axis, arr=samples)


def stride_windows(samples, sample_size, noverlap=None, axis=0):
    '''
    Get all windows of samples with length sample_size as a single array, using strides
    to avoid data duplication.

    Parameters
    ----------
    samples : 1D array or sequence
        Array or sequence containing the data.

    sample_size : integer
        The number of data points in each window.

    noverlap : integer
        The overlap between adjacent windows.
        Default is 0 (no overlap)

    axis : integer
        The axis along which the windows will run.

    '''

    if noverlap is None:
        noverlap = 0

    if noverlap >= sample_size:
        raise ValueError('noverlap must be less than sample_size')
    if sample_size < 1:
        raise ValueError('sample_size cannot be less than 1')

    samples = np.asarray(samples)

    if samples.ndim != 1:
        raise ValueError('Only 1-dimensional arrays can be used')
    if sample_size == 1 and noverlap == 0:
        if axis == 0:
            return samples[np.newaxis]
        return samples[np.newaxis].transpose()
    if sample_size > len(samples):
        raise ValueError('sample_size cannot be greater than the length of samples')

    noverlap = int(noverlap)
    sample_size = int(sample_size)

    step = sample_size - noverlap
    if axis == 0:
        shape = (sample_size, (samples.shape[-1]-noverlap)//step)
        strides = (samples.strides[0], step*samples.strides[0])
    else:
        shape = ((samples.shape[-1]-noverlap)//step, sample_size)
        strides = (step*samples.strides[0], samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)


def detrend_mean(sequence, axis=None):
    '''
    Return sequence minus the mean(sequence).

    Parameters
    ----------
    sequence : array or sequence
        Array or sequence containing the data
        Can have any dimensionality

    axis : integer
        The axis along which to take the mean.  See numpy.mean for a
        description of this argument.

    See Also
    --------
    :func:`demean`
        This function is the same as :func:`demean` except for the default
        *axis*.

    :func:`detrend_linear`

    :func:`detrend_none`
        :func:`detrend_linear` and :func:`detrend_none` are other detrend
        algorithms.

    :func:`detrend`
        :func:`detrend` is a wrapper around all the detrend algorithms.
    '''
    sequence = np.asarray(sequence)

    if axis is not None and axis+1 > sequence.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    return sequence - sequence.mean(axis, keepdims=True)


def detrend_none(samples, axis=None): # pylint: disable-msg=W0613
    '''
    Return samples: no detrending.

    Parameters
    ----------
    samples : any object
        An object containing the data

    axis : integer
        This parameter is ignored.
        It is included for compatibility with detrend_mean

    See Also
    --------
    :func:`denone`
        This function is the same as :func:`denone` except for the default
        *axis*, which has no effect.

    :func:`detrend_mean`

    :func:`detrend_linear`
        :func:`detrend_mean` and :func:`detrend_linear` are other detrend
        algorithms.

    :func:`detrend`
        :func:`detrend` is a wrapper around all the detrend algorithms.
    '''
    return samples

# pylint: disable-msg=C0103
def detrend_linear(y):
    '''
    Return x minus best fit line; 'linear' detrending.

    Parameters
    ----------
    y : 0-D or 1-D array or sequence
        Array or sequence containing the data

    axis : integer
        The axis along which to take the mean.  See numpy.mean for a
        description of this argument.

    See Also
    --------
    :func:`delinear`
        This function is the same as :func:`delinear` except for the default
        *axis*.

    :func:`detrend_mean`

    :func:`detrend_none`
        :func:`detrend_mean` and :func:`detrend_none` are other detrend
        algorithms.

    :func:`detrend`
        :func:`detrend` is a wrapper around all the detrend algorithms.
    '''
    # This is faster than an algorithm based on linalg.lstsq.
    y = np.asarray(y)

    if y.ndim > 1:
        raise ValueError('y cannot have ndim > 1')

    # short-circuit 0-D array.
    if not y.ndim:
        return np.array(0., dtype=y.dtype)

    x = np.arange(y.size, dtype=float)

    C = np.cov(x, y, bias=1)
    b = C[0, 1]/C[0, 0]

    a = y.mean() - b*x.mean()
    return y - (b*x + a)

# pylint: enable-msg=C0103

def apply_window(samples, window, axis=0, return_window=None):
    '''
    Apply the given window to the given 1D or 2D array along the given axis.

    Parameters
    ----------
    samples : 1D or 2D array or sequence
        Array or sequence containing the data.

    window : function or array.
        Either a function to generate a window or an array with length
        *samples*.shape[*axis*]

    axis : integer
        The axis over which to do the repetition.
        Must be 0 or 1.  The default is 0

    return_window : bool
        If true, also return the 1D values of the window that was applied
    '''
    samples = np.asarray(samples)

    if samples.ndim != 1 and samples.ndim != 2:
        raise ValueError('only 1D or 2D arrays can be used')
    if axis+1 > samples.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    x_shape = list(samples.shape)
    x_shape_targ = x_shape.pop(axis)

    if isinstance(window, np.ndarray):
        if len(window) != x_shape_targ:
            raise ValueError('The len(window) must be the same as the shape '
                             'of x for the chosen axis')
        else:
            window_vals = window
    else:
        window_vals = window(np.ones(x_shape_targ, dtype=samples.dtype))

    if samples.ndim == 1:
        if return_window:
            return window_vals * samples, window_vals
        return window_vals * samples

    x_shape_other = x_shape.pop()

    other_axis = (axis+1) % 2

    window_vals_rep = stride_repeat(window_vals, x_shape_other, axis=other_axis)

    if return_window:
        return window_vals_rep * samples, window_vals

    return window_vals_rep * samples


def stride_repeat(samples, sample_size, axis=0):
    '''
    Repeat the values in an array in a memory-efficient manner.  Array samples is
    stacked vertically sample_size times.

    .. warning::

        It is not safe to write to the output array.  Multiple
        elements may point to the same piece of memory, so
        modifying one value may change others.

    Parameters
    ----------
    samples : 1D array or sequence
        Array or sequence containing the data.

    sample_size : integer
        The number of time to repeat the array.

    axis : integer
        The axis along which the data will run.

    References
    ----------
    `stackoverflow: Repeat NumPy array without replicating data?
    <http://stackoverflow.com/a/5568169>`_
    '''
    if axis not in [0, 1]:
        raise ValueError('axis must be 0 or 1')
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError('only 1-dimensional arrays can be used')

    if sample_size == 1:
        if axis == 0:
            return np.atleast_2d(samples)
        return np.atleast_2d(samples).T
    if sample_size < 1:
        raise ValueError('sample_size cannot be less than 1')

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. sample_size. See #3845.
    sample_size = int(sample_size)

    if axis == 0:
        shape = (sample_size, samples.size)
        strides = (0, samples.strides[0])
    else:
        shape = (samples.size, sample_size)
        strides = (samples.strides[0], 0)

    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)
