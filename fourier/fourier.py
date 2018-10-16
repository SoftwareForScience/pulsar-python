"""
    Fourier algorithm related utils
"""

import numpy as np


def dft_slow(input_data):
    """
       Compute the discrete Fourier Transform of the one-dimensional array x
    """
    input_data = np.asarray(input_data)
    data_length = input_data.shape[0]
    data_sorted = np.arange(data_length)
    data_array = data_sorted.reshape((data_length, 1))
    exp_data = np.exp(-2j * np.pi * data_array * data_sorted / data_length)
    return np.dot(exp_data, input_data)


def fft_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()


def fft_freq(n, d=1.0):
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
    n : int
        Window length.
    d : scalar, optional
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

    val = 1.0 / (n * d)
    results = np.empty(n, int)
    N = (n-1)//2 + 1
    p1 = np.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = np.arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val
