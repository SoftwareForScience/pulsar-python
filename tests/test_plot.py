"""
    plot.py unit tests
"""
import unittest
import numpy as np
import pytest # pylint: disable-msg=E0401
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
import plot.plot

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

class TestStride():
    """
        Testclass for testing the stride functions.
    """
    def get_base(self, x_axis):
        """Gets base from x_axis"""
        y_axis = x_axis
        while y_axis.base is not None:
            y_axis = y_axis.base
        return y_axis

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

class TestDetrend(object):
    def setup(self):
        np.random.seed(0)
        n = 1000
        x = np.linspace(0., 100, n)

        self.sig_zeros = np.zeros(n)

        self.sig_off = self.sig_zeros + 100.
        self.sig_slope = np.linspace(-10., 90., n)

        self.sig_slope_mean = x - x.mean()

        sig_rand = np.random.standard_normal(n)
        sig_sin = np.sin(x*2*np.pi/(n/100))

        sig_rand -= sig_rand.mean()
        sig_sin -= sig_sin.mean()

        self.sig_base = sig_rand + sig_sin

        self.atol = 1e-08

    def test_detrend_none_0D_zeros(self):
        input = 0.
        targ = input
        res = plot.detrend_none(input)
        assert input == targ

    def test_detrend_none_0D_zeros_axis1(self):
        input = 0.
        targ = input
        res = plot.detrend_none(input, axis=1)
        assert input == targ

    def test_detrend_str_none_0D_zeros(self):
        input = 0.
        targ = input
        res = plot.detrend(input, key='none')
        assert input == targ

    def test_detrend_detrend_none_0D_zeros(self):
        input = 0.
        targ = input
        res = plot.detrend(input, key=plot.detrend_none)
        assert input == targ

    def test_detrend_none_0D_off(self):
        input = 5.5
        targ = input
        res = plot.detrend_none(input)
        assert input == targ

    def test_detrend_none_1D_off(self):
        input = self.sig_off
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1D_slope(self):
        input = self.sig_slope
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1D_base(self):
        input = self.sig_base
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1D_base_slope_off_list(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = input.tolist()
        res = plot.detrend_none(input.tolist())
        assert res == targ

    def test_detrend_none_2D(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        input = np.vstack(arri)
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_2D_T(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        input = np.vstack(arri)
        targ = input
        res = plot.detrend_none(input.T)
        assert_array_equal(res.T, targ)

    def test_detrend_mean_0D_zeros(self):
        input = 0.
        targ = 0.
        res = plot.detrend_mean(input)
        assert_almost_equal(res, targ)

    def test_detrend_str_mean_0D_zeros(self):
        input = 0.
        targ = 0.
        res = plot.detrend(input, key='mean')
        assert_almost_equal(res, targ)

    def test_detrend_detrend_mean_0D_zeros(self):
        input = 0.
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_mean)
        assert_almost_equal(res, targ)

    def test_detrend_mean_0D_off(self):
        input = 5.5
        targ = 0.
        res = plot.detrend_mean(input)
        assert_almost_equal(res, targ)

    def test_detrend_str_mean_0D_off(self):
        input = 5.5
        targ = 0.
        res = plot.detrend(input, key='mean')
        assert_almost_equal(res, targ)

    def test_detrend_detrend_mean_0D_off(self):
        input = 5.5
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_mean)
        assert_almost_equal(res, targ)

    def test_detrend_mean_1D_zeros(self):
        input = self.sig_zeros
        targ = self.sig_zeros
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1D_base(self):
        input = self.sig_base
        targ = self.sig_base
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1D_base_off(self):
        input = self.sig_base + self.sig_off
        targ = self.sig_base
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1D_base_slope(self):
        input = self.sig_base + self.sig_slope
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1D_base_slope_off(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1D_base_slope_off_axis0(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input, axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1D_base_slope_off_list(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input.tolist())
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1D_base_slope_off_list_axis0(self):
        input = self.sig_base + self.sig_slope + self.sig_off
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input.tolist(), axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_2D_default(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_2D_none(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input, axis=None)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_none_T(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri).T
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input, axis=None)
        assert_allclose(res.T, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axis0(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = plot.detrend_mean(input, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axis1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input, axis=1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2D_axism1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input, axis=-1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_2D_default(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_2D_none(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend(input, axis=None)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_str_mean_2D_axis0(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key='mean', axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_str_constant_2D_none_T(self):
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri).T
        targ = np.vstack(arrt)
        res = plot.detrend(input, key='constant', axis=None)
        assert_allclose(res.T, targ,
                        atol=1e-08)

    def test_detrend_str_default_2D_axis1(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend(input, key='default', axis=1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_detrend_mean_2D_axis0(self):
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key=plot.detrend_mean, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_bad_key_str_ValueError(self):
        input = self.sig_slope[np.newaxis]
        with pytest.raises(ValueError):
            plot.detrend(input, key='spam')

    def test_detrend_bad_key_var_ValueError(self):
        input = self.sig_slope[np.newaxis]
        with pytest.raises(ValueError):
            plot.detrend(input, key=5)

    def test_detrend_mean_0D_d0_ValueError(self):
        input = 5.5
        with pytest.raises(ValueError):
            plot.detrend_mean(input, axis=0)

    def test_detrend_0D_d0_ValueError(self):
        input = 5.5
        with pytest.raises(ValueError):
            plot.detrend(input, axis=0)

    def test_detrend_mean_1D_d1_ValueError(self):
        input = self.sig_slope
        with pytest.raises(ValueError):
            plot.detrend_mean(input, axis=1)

    def test_detrend_1D_d1_ValueError(self):
        input = self.sig_slope
        with pytest.raises(ValueError):
            plot.detrend(input, axis=1)

    def test_detrend_linear_0D_zeros(self):
        input = 0.
        targ = 0.
        res = plot.detrend_linear(input)
        assert_almost_equal(res, targ)

    def test_detrend_linear_0D_off(self):
        input = 5.5
        targ = 0.
        res = plot.detrend_linear(input)
        assert_almost_equal(res, targ)

    def test_detrend_str_linear_0D_off(self):
        input = 5.5
        targ = 0.
        res = plot.detrend(input, key='linear')
        assert_almost_equal(res, targ)

    def test_detrend_detrend_linear_0D_off(self):
        input = 5.5
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_linear)
        assert_almost_equal(res, targ)

    def test_detrend_linear_1d_off(self):
        input = self.sig_off
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope(self):
        input = self.sig_slope
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope_off(self):
        input = self.sig_slope + self.sig_off
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_str_linear_1d_slope_off(self):
        input = self.sig_slope + self.sig_off
        targ = self.sig_zeros
        res = plot.detrend(input, key='linear')
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off(self):
        input = self.sig_slope + self.sig_off
        targ = self.sig_zeros
        res = plot.detrend(input, key=plot.detrend_linear)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope_off_list(self):
        input = self.sig_slope + self.sig_off
        targ = self.sig_zeros
        res = plot.detrend_linear(input.tolist())
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_2D_ValueError(self):
        input = self.sig_slope[np.newaxis]
        with pytest.raises(ValueError):
            plot.detrend_linear(input)

    def test_detrend_str_linear_2d_slope_off_axis0(self):
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key='linear', axis=0)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off_axis1(self):
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri).T
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key=plot.detrend_linear, axis=0)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_str_linear_2d_slope_off_axis0_notranspose(self):
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend(input, key='linear', axis=1)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off_axis1_notranspose(self):
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri)
        targ = np.vstack(arrt)
        res = plot.detrend(input, key=plot.detrend_linear, axis=1)
        assert_allclose(res, targ, atol=self.atol)

if __name__ == '__main__':
    unittest.main()
