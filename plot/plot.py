import numpy as np
from fourier import fourier 

def window_hanning(x): 
    '''
    Return x times the hanning window of len(x).

    See Also
    --------
    :func:`window_none`
        :func:`window_none` is another window algorithm.
    '''
    return np.hanning(len(x))*x

def psd(x, NFFT=None, Fs=None, window=window_hanning, noverlap=None, detrend_func=None,
        pad_to=None, scale_by_freq=None, sides=None):

        if Fs is None:
            Fs = 2
        
        if NFFT is None:
            NFFT = 256
        
        if pad_to is None:
            pad_to = NFFT

        if noverlap is None:
            noverlap = 0

        x = np.asarray(x)

        if len(x) < NFFT:
            n = len(x)
            x = np.resize(x, (NFFT,))
            x[n:] = 0

        if sides == 'twosided':
            numFreqs = pad_to
            if pad_to % 2:
                freqcenter = (pad_to - 1)//2 + 1
            else:
                freqcenter = pad_to//2
            scaling_factor = 1.
        elif sides == 'onesided':
            if pad_to % 2:
                numFreqs = (pad_to + 1)//2
            else:
                numFreqs = pad_to//2 + 1
            scaling_factor = 2.

        result = stride_windows(x, NFFT, noverlap, axis=0)
        result = detrend(result, detrend_func, axis=0)
        result, windowVals = apply_window(result, window, axis=0,
                                          return_window=True)

        result = fourier.FFT_vectorized(x)
        freqs = fourier.fftfreq(pad_to, 1/Fs)[:numFreqs]

        result = np.conj(result) * result

        if not NFFT % 2:
            slc = slice(1, -1, None)
        
        else:
            slc = slice(1, None, None)

        result[slc] *= scaling_factor

        if scale_by_freq:
            result /= Fs
            result /= (np.abs(windowVals)**2).sum()
        else:
            result /= np.abs(windowVals).sum()**2

        t = np.arange(NFFT/2, len(x) - NFFT/2 + 1, NFFT - noverlap)/Fs

        if sides == 'twosided':
            freqs = np.concatenate((freqs[freqcenter:], freqs[:freqcenter]))
            result = np.concatenate((result[freqcenter:],
                                    result[:freqcenter]), 0)

        elif not pad_to % 2:
            freqs[-1] *= -1

        return result, freqs, t



def detrend(x, key=None, axis=None):
    '''
    Return x with its trend removed.

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
        return detrend(x, key=detrend_mean, axis=axis)
    elif key == 'linear':
        return detrend(x, key=detrend_linear, axis=axis)
    elif key == 'none':
        return detrend(x, key=detrend_none, axis=axis)
    elif isinstance(key, str):
        raise ValueError("Unknown value for key %s, must be one of: "
                         "'default', 'constant', 'mean', "
                         "'linear', or a function" % key)

    if not callable(key):
        raise ValueError("Unknown value for key %s, must be one of: "
                         "'default', 'constant', 'mean', "
                         "'linear', or a function" % key)

    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    if (axis is None and x.ndim == 0) or (not axis and x.ndim == 1):
        return key(x)

    # try to use the 'axis' argument if the function supports it,
    # otherwise use apply_along_axis to do it
    try:
        return key(x, axis=axis)
    except TypeError:
        return np.apply_along_axis(key, axis=axis, arr=x)


def stride_windows(x, n, noverlap=None, axis=0):
    '''
    Get all windows of x with length n as a single array, using strides
    to avoid data duplication.

    Parameters
    ----------
    x : 1D array or sequence
        Array or sequence containing the data.

    n : integer
        The number of data points in each window.

    noverlap : integer
        The overlap between adjacent windows.
        Default is 0 (no overlap)

    axis : integer
        The axis along which the windows will run.
   
    '''

    if noverlap is None:
        noverlap = 0
    
    if noverlap >= n:
        raise ValueError('noverlap must be less than n')
    if n < 1:
        raise ValueError('n cannot be less than 1')

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError('Only 1-dimensional arrays can be used')
    if n == 1 and noverlap == 0:
        if axis == 0:
            return x[np.newaxis]
        else:
            return x[np.newaxis].transpose()
    if n > len(x):
        raise ValueError('n cannot be greater than the length of x')

    noverlap = int(noverlap)
    n = int(n)

    step = n - noverlap
    if axis == 0:
        shape = (n, (x.shape[-1]-noverlap)//step)
        strides = (x.strides[0], step*x.strides[0])
    else:
        shape = ((x.shape[-1]-noverlap)//step, n)
        strides = (step*x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def detrend_mean(x, axis=None):
    '''
    Return x minus the mean(x).

    Parameters
    ----------
    x : array or sequence
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
    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    return x - x.mean(axis, keepdims=True)


def detrend_none(x, axis=None):
    '''
    Return x: no detrending.

    Parameters
    ----------
    x : any object
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
    return x


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

def apply_window(x, window, axis=0, return_window=None):
    '''
    Apply the given window to the given 1D or 2D array along the given axis.

    Parameters
    ----------
    x : 1D or 2D array or sequence
        Array or sequence containing the data.

    window : function or array.
        Either a function to generate a window or an array with length
        *x*.shape[*axis*]

    axis : integer
        The axis over which to do the repetition.
        Must be 0 or 1.  The default is 0

    return_window : bool
        If true, also return the 1D values of the window that was applied
    '''
    x = np.asarray(x)

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError('only 1D or 2D arrays can be used')
    if axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    xshape = list(x.shape)
    xshapetarg = xshape.pop(axis)

    # if cbook.iterable(window):
    #     print("CBOOK MANE")
    #     if len(window) != xshapetarg:
    #         raise ValueError('The len(window) must be the same as the shape '
    #                          'of x for the chosen axis')
    #     windowVals = window
    # else:
    windowVals = window(np.ones(xshapetarg, dtype=x.dtype))

    if x.ndim == 1:
        if return_window:
            return windowVals * x, windowVals
        else:
            return windowVals * x

    xshapeother = xshape.pop()

    otheraxis = (axis+1) % 2

    windowValsRep = stride_repeat(windowVals, xshapeother, axis=otheraxis)

    if return_window:
        return windowValsRep * x, windowVals
    else:
        return windowValsRep * x

def stride_repeat(x, n, axis=0):
    '''
    Repeat the values in an array in a memory-efficient manner.  Array x is
    stacked vertically n times.

    .. warning::

        It is not safe to write to the output array.  Multiple
        elements may point to the same piece of memory, so
        modifying one value may change others.

    Parameters
    ----------
    x : 1D array or sequence
        Array or sequence containing the data.

    n : integer
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
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('only 1-dimensional arrays can be used')

    if n == 1:
        if axis == 0:
            return np.atleast_2d(x)
        else:
            return np.atleast_2d(x).T
    if n < 1:
        raise ValueError('n cannot be less than 1')

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. n. See #3845.
    n = int(n)

    if axis == 0:
        shape = (n, x.size)
        strides = (0, x.strides[0])
    else:
        shape = (x.size, n)
        strides = (x.strides[0], 0)

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
