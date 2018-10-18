"""
    plot.py unit tests
"""
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
import plot.plot as plot

from .context import plot # pylint: disable-msg=E0611

class TestPlot(unittest.TestCase):
    """
        Testclass for testing the plotting functions
    """

    # def __init__(self):
    #     np.random.seed(0)
    #     n = 1000
    #
    #     self.sig_rand = np.random.standard_normal(n) + 100.
    #     self.sig_ones = np.ones(n)
    #
    # def test_window_hanning_rand(self):
    #     targ = np.hanning(len(self.sig_rand)) * self.sig_rand
    #     res = plot.window_hanning(self.sig_rand)
    #
    #     self.assertAlmostEquals(targ.all(), res.all())
    #
    # def test_window_hanning_ones(self):
    #     targ = np.hanning(len(self.sig_ones))
    #     res = plot.window_hanning(self.sig_ones)
    #
    #     self.assertAlmostEquals(targ.all(), res.all(), atol=1e-06)

class TestStride(object):
    def get_base(self, x):
        y = x
        while y.base is not None:
            y = y.base
        return y

    def calc_window_target(self, x, NFFT, noverlap=0, axis=0):
        '''This is an adaptation of the original window extraction
        algorithm.  This is here to test to make sure the new implementation
        has the same result'''
        step = NFFT - noverlap
        ind = np.arange(0, len(x) - NFFT + 1, step)
        n = len(ind)
        result = np.zeros((NFFT, n))

        # do the ffts of the slices
        for i in range(n):
            result[:, i] = x[ind[i]:ind[i] + NFFT]
        if axis == 1:
            result = result.T
        return result

    @pytest.mark.parametrize('shape', [(), (10, 1)], ids=['0D', '2D'])
    def test_stride_windows_invalid_input_shape(self, shape):
        x = np.arange(np.prod(shape)).reshape(shape)
        with pytest.raises(ValueError):
            plot.stride_windows(x, 5)

    @pytest.mark.parametrize('n, noverlap',
                             [(0, None), (11, None), (2, 2), (2, 3)],
                             ids=['n less than 1', 'n greater than input',
                                  'noverlap greater than n',
                                  'noverlap equal to n'])
    def test_stride_windows_invalid_params(self, n, noverlap):
        x = np.arange(10)
        with pytest.raises(ValueError):
            plot.stride_windows(x, n, noverlap)

    @pytest.mark.parametrize('shape', [(), (10, 1)], ids=['0D', '2D'])
    def test_stride_repeat_invalid_input_shape(self, shape):
        x = np.arange(np.prod(shape)).reshape(shape)
        with pytest.raises(ValueError):
            plot.stride_repeat(x, 5)

    @pytest.mark.parametrize('axis', [-1, 2],
                             ids=['axis less than 0',
                                  'axis greater than input shape'])
    def test_stride_repeat_invalid_axis(self, axis):
        x = np.array(0)
        with pytest.raises(ValueError):
            plot.stride_repeat(x, 5, axis=axis)

    def test_stride_repeat_n_lt_1_ValueError(self):
        x = np.arange(10)
        with pytest.raises(ValueError):
            plot.stride_repeat(x, 0)

    @pytest.mark.parametrize('axis', [0, 1], ids=['axis0', 'axis1'])
    @pytest.mark.parametrize('n', [1, 5], ids=['n1', 'n5'])
    def test_stride_repeat(self, n, axis):
        x = np.arange(10)
        y = plot.stride_repeat(x, n, axis=axis)

        expected_shape = [10, 10]
        expected_shape[axis] = n
        yr = np.repeat(np.expand_dims(x, axis), n, axis=axis)

        assert yr.shape == y.shape
        assert_array_equal(yr, y)
        assert tuple(expected_shape) == y.shape
        assert self.get_base(y) is x

    @pytest.mark.parametrize('axis', [0, 1], ids=['axis0', 'axis1'])
    @pytest.mark.parametrize('n, noverlap',
                             [(1, 0), (5, 0), (15, 2), (13, -3)],
                             ids=['n1-noverlap0', 'n5-noverlap0',
                                  'n15-noverlap2', 'n13-noverlapn3'])
    def test_stride_windows(self, n, noverlap, axis):
        x = np.arange(100)
        y = plot.stride_windows(x, n, noverlap=noverlap, axis=axis)

        expected_shape = [0, 0]
        expected_shape[axis] = n
        expected_shape[1 - axis] = 100 // (n - noverlap)
        yt = self.calc_window_target(x, n, noverlap=noverlap, axis=axis)

        assert yt.shape == y.shape
        assert_array_equal(yt, y)
        assert tuple(expected_shape) == y.shape
        assert self.get_base(y) is x

    @pytest.mark.parametrize('axis', [0, 1], ids=['axis0', 'axis1'])
    def test_stride_windows_n32_noverlap0_unflatten(self, axis):
        n = 32
        x = np.arange(n)[np.newaxis]
        x1 = np.tile(x, (21, 1))
        x2 = x1.flatten()
        y = plot.stride_windows(x2, n, axis=axis)

        if axis == 0:
            x1 = x1.T
        assert y.shape == x1.shape
        assert_array_equal(y, x1)

    def test_stride_ensure_integer_type(self):
        N = 100
        x = np.empty(N + 20, dtype='>f4')
        x.fill(np.NaN)
        y = x[10:-10]
        y.fill(0.3)
        # previous to #3845 lead to corrupt access
        y_strided = plot.stride_windows(y, 33, 0.6)
        assert_array_equal(y_strided, 0.3)
        # previous to #3845 lead to corrupt access
        y_strided = plot.stride_windows(y, 33.3, 0)
        assert_array_equal(y_strided, 0.3)
        # even previous to #3845 could not find any problematic
        # configuration however, let's be sure it's not accidentally
        # introduced
        y_strided = plot.stride_repeat(y, 33.815)
        assert_array_equal(y_strided, 0.3)


if __name__ == '__main__':
    unittest.main()
