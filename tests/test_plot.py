"""
    plot.py unit tests
"""
import unittest
import pytest # pylint: disable-msg=E0401
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
from context import plot # pylint: disable-msg=E0611

class TestWindowHanning(unittest.TestCase):
    """
        Testclass for testing the plotting functions
    """
    np.random.seed(0)
    n = 1000

    sig_rand = np.random.standard_normal(n) + 100.
    sig_ones = np.ones(n)

    Fs = 100
    
    def test_window_hanning_rand(self):
        targ = np.hanning(len(self.sig_rand)) * self.sig_rand
        res = plot.window_hanning(self.sig_rand)
        self.assertAlmostEquals(targ.all(), res.all())
    
    def test_window_hanning_ones(self):
        targ = np.hanning(len(self.sig_ones))
        res = plot.window_hanning(self.sig_ones)
        self.assertAlmostEquals(targ.all(), res.all(), atol=1e-06)

class TestStride():
    """
        Testclass for testing the stride functions.
    """
    @staticmethod
    def get_base(x_axis):
        """Gets base from x_axis"""
        y_axis = x_axis
        while y_axis.base is not None:
            y_axis = y_axis.base
        return y_axis

    @staticmethod
    def calc_window_target(x_axis, n_fft, noverlap=0, axis=0):
        """This is an adaptation of the original window extraction
        algorithm.  This is here to test to make sure the new implementation
        has the same result"""
        step = n_fft - noverlap
        ind = np.arange(0, len(x_axis) - n_fft + 1, step)
        n_timesteps = len(ind)
        result = np.zeros((n_fft, n_timesteps))

        # do the ffts of the slices
        for i in range(n_timesteps):
            result[:, i] = x_axis[ind[i]:ind[i] + n_fft]
        if axis == 1:
            result = result.T
        return result

    @staticmethod
    @pytest.mark.parametrize('shape', [(), (10, 1)], ids=['0D', '2D'])
    def test_stride_windows_invalid_input_shape(shape):
        """Tests if a exception is raised when a stride_window has an invalid shape. """
        x_axis = np.arange(np.prod(shape)).reshape(shape)
        with pytest.raises(ValueError):
            plot.stride_windows(x_axis, 5)

    @staticmethod
    @pytest.mark.parametrize('n_timesteps, noverlap',
                             [(0, None), (11, None), (2, 2), (2, 3)],
                             ids=['n less than 1', 'n greater than input',
                                  'noverlap greater than n',
                                  'noverlap equal to n'])
    def test_stride_windows_invalid_params(n_timesteps, noverlap):
        """Tests if a exception is raised when invalid params are supplied. """
        x_axis = np.arange(10)
        with pytest.raises(ValueError):
            plot.stride_windows(x_axis, n_timesteps, noverlap)

    @staticmethod
    @pytest.mark.parametrize('shape', [(), (10, 1)], ids=['0D', '2D'])
    def test_stride_repeat_invalid_input_shape(shape):
        """Tests if a exception is raised when a invalid shape is supplied in 0D or 2D"""
        x_axis = np.arange(np.prod(shape)).reshape(shape)
        with pytest.raises(ValueError):
            plot.stride_repeat(x_axis, 5)

    @staticmethod
    @pytest.mark.parametrize('axis', [-1, 2],
                             ids=['axis less than 0',
                                  'axis greater than input shape'])
    def test_stride_repeat_invalid_axis(axis):
        """Checks is a exception is raised when the axis is not greater than the input shape. """
        x_axis = np.array(0)
        with pytest.raises(ValueError):
            plot.stride_repeat(x_axis, 5, axis=axis)

    @staticmethod
    def test_stride_repeat_n_timesteps_lt1_valuerror():
        """Test for stride ToDo Add more specific explanation"""
        x_axis = np.arange(10)
        with pytest.raises(ValueError):
            plot.stride_repeat(x_axis, 0)

    @pytest.mark.parametrize('axis', [0, 1], ids=['axis0', 'axis1'])
    @pytest.mark.parametrize('n_timesteps', [1, 5], ids=['n1', 'n5'])
    def test_stride_repeat(self, n_timesteps, axis):
        """Tests stride repeat function. """
        x_axis = np.arange(10)
        y_axis = plot.stride_repeat(x_axis, n_timesteps, axis=axis)

        expected_shape = [10, 10]
        expected_shape[axis] = n_timesteps
        # y_times_r is the former yr, renamed var for PyLint.
        y_times_r = np.repeat(np.expand_dims(x_axis, axis), n_timesteps, axis=axis)

        assert y_times_r.shape == y_axis.shape
        assert_array_equal(y_times_r, y_axis)
        assert tuple(expected_shape) == y_axis.shape
        assert self.get_base(y_axis) is x_axis

    @pytest.mark.parametrize('axis', [0, 1], ids=['axis0', 'axis1'])
    @pytest.mark.parametrize('n_timesteps, noverlap',
                             [(1, 0), (5, 0), (15, 2), (13, -3)],
                             ids=['n1-noverlap0', 'n5-noverlap0',
                                  'n15-noverlap2', 'n13-noverlapn3'])
    def test_stride_windows(self, n_timesteps, noverlap, axis):
        """Tests stride windows function"""
        x_axis = np.arange(100)
        y_axis = plot.stride_windows(x_axis, n_timesteps, noverlap=noverlap, axis=axis)

        expected_shape = [0, 0]
        expected_shape[axis] = n_timesteps
        expected_shape[1 - axis] = 100 // (n_timesteps - noverlap)
        y_target = self.calc_window_target(x_axis, n_timesteps, noverlap=noverlap, axis=axis)

        assert y_target.shape == y_axis.shape
        assert_array_equal(y_target, y_axis)
        assert tuple(expected_shape) == y_axis.shape
        assert self.get_base(y_axis) is x_axis

    @staticmethod
    @pytest.mark.parametrize('axis', [0, 1], ids=['axis0', 'axis1'])
    def test_stride_windows_n32_noverlap0_unflatten(axis):
        """Tests stride windows flattened with no overlap. """
        n_timesteps = 32
        x_axis = np.arange(n_timesteps)[np.newaxis]
        x1_axis = np.tile(x_axis, (21, 1))
        x2_axis = x1_axis.flatten()
        y_axis = plot.stride_windows(x2_axis, n_timesteps, axis=axis)

        if axis == 0:
            x1_axis = x1_axis.T
        assert y_axis.shape == x1_axis.shape
        assert_array_equal(y_axis, x1_axis)

    @staticmethod
    def test_stride_ensure_integer_type():
        """Tests if a stride equals tot he right integer type ToDo Write a better comment. """
        n_timesteps = 100
        x_axis = np.empty(n_timesteps + 20, dtype='>f4')
        x_axis.fill(np.NaN)
        y_axis = x_axis[10:-10]
        y_axis.fill(0.3)
        # previous to #3845 lead to corrupt access
        y_strided = plot.stride_windows(y_axis, 33, 0.6)
        assert_array_equal(y_strided, 0.3)
        # previous to #3845 lead to corrupt access
        y_strided = plot.stride_windows(y_axis, 33.3, 0)
        assert_array_equal(y_strided, 0.3)
        # even previous to #3845 could not find any problematic
        # configuration however, let's be sure it's not accidentally
        # introduced
        y_strided = plot.stride_repeat(y_axis, 33.815)
        assert_array_equal(y_strided, 0.3)

# pylint: disable=R0904
class TestDetrend():
    """Class for testing the detrend function. """
    def setup(self):
        """Initial setup for testing. X and Y axis are set accordingly.
        And a proper slope is calculated.  """
        np.random.seed(0)
        n_timesteps = 1000
        x_axis = np.linspace(0., 100, n_timesteps)

        self.sig_zeros = np.zeros(n_timesteps) # pylint: disable=W0201

        self.sig_off = self.sig_zeros + 100. # pylint: disable=W0201
        self.sig_slope = np.linspace(-10., 90., n_timesteps) # pylint: disable=W0201

        self.sig_slope_mean = x_axis - x_axis.mean() # pylint: disable=W0201

        sig_rand = np.random.standard_normal(n_timesteps)
        sig_sin = np.sin(x_axis*2*np.pi/(n_timesteps/100))

        sig_rand -= sig_rand.mean()
        sig_sin -= sig_sin.mean()

        self.sig_base = sig_rand + sig_sin # pylint: disable=W0201

        self.atol = 1e-08 # pylint: disable=W0201

    @staticmethod
    def test_detrend_none_0dimension_zeros():
        """A 0D zeros shall equal targ. """
        input = 0. # pylint: disable=W0622
        targ = input
        assert input == targ
    @staticmethod
    def test_detrend_none_0dimension_zeros_axis1():
        """Test for axis 1 """
        input = 0.  # pylint: disable=W0622
        targ = input
        assert input == targ

    @staticmethod
    def test_detrend_str_none_0dimension_zeros():
        """Tests for 0D_zeros strings. """
        input = 0. # pylint: disable=W0622
        targ = input
        assert input == targ

    @staticmethod
    def test_detrend_detrend_none_0dimension_zeros():
        """Tests for 0D zeros """
        input = 0. # pylint: disable=W0622
        targ = input
        assert input == targ

    @staticmethod
    def test_detrend_none_0dimension_off():
        """Tests for 0d without dimensions.. """

        input = 5.5  # pylint: disable=W0622
        targ = input
        assert input == targ

    def test_detrend_none_1dimension_off(self):
        """Tests without 1 dimension"""
        input = self.sig_off # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1dimension_slope(self):
        """Unit tests for detrend 1 dim_slope. """
        input = self.sig_slope # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1dimension_base(self):
        """Unit tests for detrend 1 dim slope. """
        input = self.sig_base # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1dimension_base_slope_off_list(self):
        """Unit test for detrend 1 dimension base slope off list"""
        input = self.sig_base + self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = input.tolist()
        res = plot.detrend_none(input.tolist())
        assert res == targ

    def test_detrend_none_2dimension(self):
        """Unit test for test_dtrend_none_2d"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_2dimension_time(self):
        """Unit test for test_detend_none_2dimension_time"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input.T)
        assert_array_equal(res.T, targ)

    @staticmethod
    def test_detrend_mean_0dimension_zeros():
        """Unit test for 0 dimension. """
        input = 0. # pylint: disable=W0622
        targ = 0.
        res = plot.detrend_mean(input)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_str_mean_0dimension_zeros():
        """Unit test for test_detrend_str_mean_0dimension_zeros"""
        input = 0. # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key='mean')
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_detrend_mean_0dimension_zeros():
        """Unit test for test_detrend_detrend_mean_0dimension_zeros"""
        input = 0. # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_mean)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_mean_0dimension_off():
        """Unit test for test_detrend_mean_0dimension_off"""
        input = 5.5 # pylint: disable=W0622
        targ = 0.
        res = plot.detrend_mean(input)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_str_mean_0dimension_off():
        """Unit test for test_detrend_str_mean_0dimension_off"""
        input = 5.5 # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key='mean')
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_detrend_mean_0dimension_off():
        """Unit test for test_detrend_detrend_mean_0dimension_off"""
        input = 5.5 # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_mean)
        assert_almost_equal(res, targ)

    def test_detrend_mean_1dimension_zeros(self):
        """Unit test for test_detrend_mean_1dimension_zeros"""
        input = self.sig_zeros # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1dimension_base(self):
        """Unit test for test_detrend_mean_1dimension_base"""
        input = self.sig_base # pylint: disable=W0622
        targ = self.sig_base
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1dimension_base_off(self):
        """Unit test for test_detrend_mean_1dimension_base_off"""
        input = self.sig_base + self.sig_off # pylint: disable=W0622
        targ = self.sig_base
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1dimension_base_slope(self):
        """Unit test for test_detrend_mean_1dimension_base_slope"""
        input = self.sig_base + self.sig_slope # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1dimension_base_slope_off(self):
        """Unit test for test_detrend_mean_1dimension_base_slope_off"""
        input = self.sig_base + self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1dimension_base_slope_off_axis0(self):
        """Unit test test_detrend_mean_1dimension_base_slope_off_axis0"""
        input = self.sig_base + self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input, axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1dimension_base_slope_off_list(self):
        """Unit test for test_detrend_mean_1dimension_base_slope_off_list"""
        input = self.sig_base + self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input.tolist())
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1dimension_base_slope_off_list_axis0(self):
        """Unit test test_detrend_mean_1dimension_base_slope_off_list_axis0"""
        input = self.sig_base + self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input.tolist(), axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_2dimension_default(self):
        """Unit test test_detrend_mean_2dimension_default"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_2dimension_none(self):
        """Unit test test_detrend_mean_2dimension_none"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input, axis=None)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2dimension_none_t(self):
        """Unit test test_detrend_mean_2dimesnion_none_t"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri).T # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input, axis=None)
        assert_allclose(res.T, targ,
                        atol=1e-08)

    def test_detrend_mean_2dimension_axis0(self):
        """Unit test for test_detrend_mean_2dimension_axis0"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T # pylint: disable=W0622
        targ = np.vstack(arrt).T
        res = plot.detrend_mean(input, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2dimension_axis1(self):
        """Unit test for  test_detrend_mean_2dimension_axis1"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input, axis=1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_mean_2dimension_axism1(self):
        """Unit test for test_detrend_mean_2dimension_axis1"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input, axis=-1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_2dimension_default(self):
        """Unit test test_detrend_2dimension_default"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_2dimension_none(self):
        """Unit test for test_detrend_2dimension_none"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend(input, axis=None)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_str_mean_2dimension_axis0(self):
        """Unit test for test_detrend_str_mean_2dimension_axis0"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T # pylint: disable=W0622
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key='mean', axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_str_constant_2dimension_none_t(self):
        """Unit test for test_detrend_str_constant_2dimension_none_t"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri).T # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend(input, key='constant', axis=None)
        assert_allclose(res.T, targ,
                        atol=1e-08)

    def test_detrend_str_default_2dimension_axis1(self):
        """Unit test for test_detrend_str_default_2dimension_axis1"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend(input, key='default', axis=1)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_detrend_mean_2dimension_axis0(self):
        """Unit test for test_detrend_mean_2dimension_axis0"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        arrt = [self.sig_base,
                self.sig_base,
                self.sig_base + self.sig_slope_mean,
                self.sig_base + self.sig_slope_mean]
        input = np.vstack(arri).T # pylint: disable=W0622
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key=plot.detrend_mean, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_bad_key_str_value_error(self):
        """Checks if valuerror is thrown when the key is is a str. """
        input = self.sig_slope[np.newaxis] # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend(input, key='spam')

    def test_detrend_bad_key_var_value_error(self):
        """Unit test that checks if a error is thrown when a wrong/bad key value is supplied. """
        input = self.sig_slope[np.newaxis] # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend(input, key=5)

    @staticmethod
    def test_detrend_mean_0d_d0_value_error():
        """Unit test to check if a value error is thrown. """
        input = 5.5 # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend_mean(input, axis=0)

    @staticmethod
    def test_detrend_0d_d0_value_error():
        """Unit test if a value error is thrown. """
        input = 5.5 # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend(input, axis=0)

    def test_detrend_mean_1d_d1_value_error(self):
        """Unit test that checks if a value error is thrown when a incorrect value is supplied. """
        input = self.sig_slope # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend_mean(input, axis=1)

    def test_detrend_1d_d1_value_error(self):
        """Unit test that checks if a value rror is thrown when a incorrect value is supplied. """
        input = self.sig_slope # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend(input, axis=1)

    @staticmethod
    def test_detrend_linear_0d_zeros():
        """Unit test for linear 0d detrend usage. """
        input = 0. # pylint: disable=W0622
        targ = 0.
        res = plot.detrend_linear(input)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_linear_0d_off():
        """Detrend linear unit test.  """
        input = 5.5 # pylint: disable=W0622
        targ = 0.
        res = plot.detrend_linear(input)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_str_linear_0d_off():
        """Unit test for checking str_linear_0d. """
        input = 5.5 # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key='linear')
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_detrend_linear_0d_off():
        """Unit test for linear 0d. """
        input = 5.5 # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_linear)
        assert_almost_equal(res, targ)

    def test_detrend_linear_1d_off(self):
        """Unit test for linear 1d. """
        input = self.sig_off # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope(self):
        """Unit test for 1d slope. """
        input = self.sig_slope # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope_off(self):
        """Unit test for 1d slope off. """
        input = self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_str_linear_1d_slope_off(self):
        """Unit test for 1d slope off. """
        input = self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend(input, key='linear')
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off(self):
        """Unit test for linear 1d slope off. """
        input = self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend(input, key=plot.detrend_linear)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope_off_list(self):
        """Unit test for 1d slope off list. """
        input = self.sig_slope + self.sig_off # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_linear(input.tolist())
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_2d_value_error(self):
        """Unit test that checks if a value error is thrown for linear 2d functions.  """
        input = self.sig_slope[np.newaxis] # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend_linear(input)

    def test_detrend_str_linear_2d_slope_off_axis0(self):
        """Off axis linear 2d unit test. """
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri).T # pylint: disable=W0622
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key='linear', axis=0)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off_axis1(self):
        """Unit test for slope off axis linear 1d. """
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri).T # pylint: disable=W0622
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key=plot.detrend_linear, axis=0)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_str_linear_2d_slope_off_axis0_notranspose(self):
        """Unit test for str linear 2d slope off axis0 no transpose test. """
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend(input, key='linear', axis=1)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off_axis1_notranspose(self):
        """Unit tests for 1d slope off axis notranspose. """
        arri = [self.sig_off,
                self.sig_slope,
                self.sig_slope + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_zeros,
                self.sig_zeros]
        input = np.vstack(arri) # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend(input, key=plot.detrend_linear, axis=1)
        assert_allclose(res, targ, atol=self.atol)

if __name__ == '__main__':
    unittest.main()
