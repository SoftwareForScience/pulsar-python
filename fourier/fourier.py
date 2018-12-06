"""
    Fourier algorithm related utils
"""

import numpy as np

def fft_matrix(input_data, nfft=None, axis=-1):
    """
        Performs FFT for each row in matrix
    """
    input_data = np.asarray(input_data)

    result = [fft_vectorized(row, nfft, axis) for row in input_data]

    return result


def dft_slow(input_data):
    """
        Compute the discrete Fourier Transform of the one-dimensional array input_data

        Parameters
        ----------
        input_data : ndarray
            Array of length `n` containing the values to be transformed.

        Returns
        -------
        ndarray
            Array of length `n` containing the transformed values.
    """
    input_data = np.asarray(input_data)
    data_length = input_data.shape[0]
    data_sorted = np.arange(data_length)
    data_array = data_sorted.reshape((data_length, 1))
    exp_data = np.exp(-2j * np.pi * data_array * data_sorted / data_length)
    return np.dot(exp_data, input_data)


# pylint: disable=W0511
# TODO: support multidimensional fft
def fft_vectorized(input_data, nfft=None, axis=-1):
    """
        A vectorized, non-recursive version of the Cooley-Tukey FFT

        Parameters
        ----------
        input_data : ndarray
            Array of length `n` containing the values to be transformed.

        nfft : int (Optional)
            Integer describing the number of datapoints in the output. If
            nfft is larger than the input data size, this input data will be padded with zeroes.
            If it is smaller, the input data will be cropped to length `nfft`.

        axis : int (Optional, default = -1)
            Integer describing on what axis of the matrix the fft should be executed.
            This parameter is not yet fully used, will be in the future.
        Returns
        -------
        ndarray
            Array of length `n` containing the transformed values.
    """
    input_data = np.asarray(input_data)

    if nfft is None:
        nfft = input_data.shape[axis]

    if input_data.shape[axis] != nfft:
        input_shape = list(input_data.shape)
        index = [slice(None)]*len(input_shape)
        if input_shape[axis] > nfft:
            index[axis] = slice(0, nfft)
            input_data = input_data[tuple(index)]
        else:
            index[axis] = slice(0, input_shape[axis])
            input_shape[axis] = nfft
            zeroes = np.zeros(input_shape, input_data.dtype.char)
            zeroes[tuple(index)] = input_data
            input_data = zeroes

    data_length = input_data.shape[0]

    if np.log2(data_length) % 1 > 0:
        raise ValueError("Size of input data must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    min_data = min(data_length, 32)

    # Perform an O[N^2] DFT on all length-min_data sub-problems at once
    data_sorted = np.arange(min_data)
    data_matrix = data_sorted[:, None]
    exp_data = np.exp(-2j * np.pi * data_sorted * data_matrix / min_data)
    dot_product = np.dot(exp_data, input_data.reshape((min_data, -1)))

    # build-up each level of the recursive calculation all at once
    while dot_product.shape[0] < data_length:
        data_even = dot_product[:, :dot_product.shape[1] // 2]
        data_odd = dot_product[:, dot_product.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(dot_product.shape[0])
                        / dot_product.shape[0])[:, None]
        dot_product = np.vstack([data_even + factor * data_odd,
                                 data_even - factor * data_odd])

    return dot_product.ravel()


def fft_freq(window_len, spacing=1.0):
    """
        Return the Discrete Fourier Transform sample frequencies.

        The returned float array `f` contains the frequency bin centers in cycles
        per unit of the sample spacing (with zero at the start).  For instance, if
        the sample spacing is in seconds, then the frequency unit is cycles/second.

        Given a window length `n` and a sample spacing `d`::

          f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
          f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

        Parameters
        ----------
        window_len : int
            Window length.
        spacing : scalar, optional
            Sample spacing (inverse of the sampling rate). Defaults to 1.

        Returns
        -------
        f : ndarray
            Array of length `n` containing the sample frequencies.

        Examples
        --------
        >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
        >>> fourier = np.fft.fft(signal)
        >>> n = signal.size
        >>> timestep = 0.1
        >>> freq = np.fft.fftfreq(n, d=timestep)
        >>> freq
        array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])
    """
    val = 1.0 / (window_len * spacing)
    results = np.empty(window_len, int)
    window_half = (window_len-1)//2 + 1
    window_p1 = np.arange(0, window_half, dtype=int)
    results[:window_half] = window_p1
    window_p2 = np.arange(-(window_len//2), 0, dtype=int)
    results[window_half:] = window_p2
    return results * val


def fft_shift(samples, axes=None):
    """
        Shift the zero-frequency component to the center of the spectrum.
    """
    samples = np.asarray(samples)

    if axes is None:
        axes = tuple(range(samples.ndim))
        shift = [dim // 2 for dim in samples.shape]
    elif isinstance(axes, int):
        shift = samples.shape[axes] // 2
    else:
        shift = [samples.shape[ax] // 2 for ax in axes]

    return np.roll(samples, shift, axes)

def ifft(input_data):
    """
        Compute the inverse discrete Fourier Transform of the one-dimensional array input_data

        Parameters
        ----------
        input_data : ndarray
            Array of length `n` containing the values to be transformed.

        Returns
        -------
        ndarray
            Array of length `n` containing the transformed values.
    """
    input_data = np.asarray(input_data, dtype=type(input_data[0]))
    N = input_data.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("Size of input data must be a power of 2")

    N_min = min(N, 32)

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(2j * np.pi * n * k / N_min)
    X = 1/N * np.dot(M, X.reshape((N_min, -1)))

    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()