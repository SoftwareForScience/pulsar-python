# pylint: disable-msg=C0302
"""
    plot.py unit tests
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
import pytest  # pylint: disable-msg=E0401
from .context import plot  # pylint: disable-msg=E0611

class TestStride:
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

        self.sig_zeros = np.zeros(n_timesteps)  # pylint: disable=W0201

        self.sig_off = self.sig_zeros + 100.  # pylint: disable=W0201
        self.sig_slope = np.linspace(-10., 90., n_timesteps)  # pylint: disable=W0201

        self.sig_slope_mean = x_axis - x_axis.mean()  # pylint: disable=W0201

        sig_rand = np.random.standard_normal(n_timesteps)
        sig_sin = np.sin(x_axis * 2 * np.pi / (n_timesteps / 100))

        sig_rand -= sig_rand.mean()
        sig_sin -= sig_sin.mean()

        self.sig_base = sig_rand + sig_sin  # pylint: disable=W0201

        self.atol = 1e-08  # pylint: disable=W0201

    @staticmethod
    def test_detrend_none():
        """Test if a detrend function can have a key of None. """
        input = 0. # pylint: disable=W0622
        detrend_none = plot.detrend(input, None)

        assert detrend_none == input

    @staticmethod
    def test_detrend_none_0dimension_zeros():
        """A 0D zeros shall equal targ. """
        input = 0.  # pylint: disable=W0622
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
        input = 0.  # pylint: disable=W0622
        targ = input
        assert input == targ

    @staticmethod
    def test_detrend_detrend_none_0dimension_zeros():
        """Tests for 0D zeros """
        input = 0.  # pylint: disable=W0622
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
        input = self.sig_off  # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1dimension_slope(self):
        """Unit tests for detrend 1 dim_slope. """
        input = self.sig_slope  # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1dimension_base(self):
        """Unit tests for detrend 1 dim slope. """
        input = self.sig_base  # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_1dimension_base_slope_off_list(self):
        """Unit test for detrend 1 dimension base slope off list"""
        input = self.sig_base + self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = input.tolist()
        res = plot.detrend_none(input.tolist())
        assert res == targ

    def test_detrend_none_2dimension(self):
        """Unit test for test_dtrend_none_2d"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        input = np.vstack(arri)  # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input)
        assert_array_equal(res, targ)

    def test_detrend_none_2dimension_time(self):
        """Unit test for test_detend_none_2dimension_time"""
        arri = [self.sig_base,
                self.sig_base + self.sig_off,
                self.sig_base + self.sig_slope,
                self.sig_base + self.sig_off + self.sig_slope]
        input = np.vstack(arri)  # pylint: disable=W0622
        targ = input
        res = plot.detrend_none(input.T)
        assert_array_equal(res.T, targ)

    @staticmethod
    def test_detrend_mean_0dimension_zeros():
        """Unit test for 0 dimension. """
        input = 0.  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend_mean(input)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_str_mean_0dimension_zeros():
        """Unit test for test_detrend_str_mean_0dimension_zeros"""
        input = 0.  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key='mean')
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_detrend_mean_0dimension_zeros():
        """Unit test for test_detrend_detrend_mean_0dimension_zeros"""
        input = 0.  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_mean)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_mean_0dimension_off():
        """Unit test for test_detrend_mean_0dimension_off"""
        input = 5.5  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend_mean(input)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_str_mean_0dimension_off():
        """Unit test for test_detrend_str_mean_0dimension_off"""
        input = 5.5  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key='mean')
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_detrend_mean_0dimension_off():
        """Unit test for test_detrend_detrend_mean_0dimension_off"""
        input = 5.5  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_mean)
        assert_almost_equal(res, targ)

    def test_detrend_mean_1dimension_zeros(self):
        """Unit test for test_detrend_mean_1dimension_zeros"""
        input = self.sig_zeros  # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1dimension_base(self):
        """Unit test for test_detrend_mean_1dimension_base"""
        input = self.sig_base  # pylint: disable=W0622
        targ = self.sig_base
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1dimension_base_off(self):
        """Unit test for test_detrend_mean_1dimension_base_off"""
        input = self.sig_base + self.sig_off  # pylint: disable=W0622
        targ = self.sig_base
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1dimension_base_slope(self):
        """Unit test for test_detrend_mean_1dimension_base_slope"""
        input = self.sig_base + self.sig_slope  # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_mean_1dimension_base_slope_off(self):
        """Unit test for test_detrend_mean_1dimension_base_slope_off"""
        input = self.sig_base + self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1dimension_base_slope_off_axis0(self):
        """Unit test test_detrend_mean_1dimension_base_slope_off_axis0"""
        input = self.sig_base + self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input, axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1dimension_base_slope_off_list(self):
        """Unit test for test_detrend_mean_1dimension_base_slope_off_list"""
        input = self.sig_base + self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input.tolist())
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_1dimension_base_slope_off_list_axis0(self):
        """Unit test test_detrend_mean_1dimension_base_slope_off_list_axis0"""
        input = self.sig_base + self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = self.sig_base + self.sig_slope_mean
        res = plot.detrend_mean(input.tolist(), axis=0)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_2dimension_default(self):
        """Unit test test_detrend_mean_2dimension_default"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)  # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend_mean(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_mean_2dimension_none(self):
        """Unit test test_detrend_mean_2dimension_none"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)  # pylint: disable=W0622
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
        input = np.vstack(arri).T  # pylint: disable=W0622
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
        input = np.vstack(arri).T  # pylint: disable=W0622
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
        input = np.vstack(arri)  # pylint: disable=W0622
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
        input = np.vstack(arri)  # pylint: disable=W0622
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
        input = np.vstack(arri)  # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend(input)
        assert_allclose(res, targ, atol=1e-08)

    def test_detrend_2dimension_none(self):
        """Unit test for test_detrend_2dimension_none"""
        arri = [self.sig_off,
                self.sig_base + self.sig_off]
        arrt = [self.sig_zeros,
                self.sig_base]
        input = np.vstack(arri)  # pylint: disable=W0622
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
        input = np.vstack(arri).T  # pylint: disable=W0622
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
        input = np.vstack(arri).T  # pylint: disable=W0622
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
        input = np.vstack(arri)  # pylint: disable=W0622
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
        input = np.vstack(arri).T  # pylint: disable=W0622
        targ = np.vstack(arrt).T
        res = plot.detrend(input, key=plot.detrend_mean, axis=0)
        assert_allclose(res, targ,
                        atol=1e-08)

    def test_detrend_bad_key_str_value_error(self):
        """Checks if valuerror is thrown when the key is is a str. """
        input = self.sig_slope[np.newaxis]  # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend(input, key='spam')

    def test_detrend_bad_key_var_value_error(self):
        """Unit test that checks if a error is thrown when a wrong/bad key value is supplied. """
        input = self.sig_slope[np.newaxis]  # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend(input, key=5)

    @staticmethod
    def test_detrend_mean_0d_d0_value_error():
        """Unit test to check if a value error is thrown. """
        input = 5.5  # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend_mean(input, axis=0)

    @staticmethod
    def test_detrend_0d_d0_value_error():
        """Unit test if a value error is thrown. """
        input = 5.5  # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend(input, axis=0)

    def test_detrend_mean_1d_d1_value_error(self):
        """Unit test that checks if a value error is thrown when a incorrect value is supplied. """
        input = self.sig_slope  # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend_mean(input, axis=1)

    def test_detrend_1d_d1_value_error(self):
        """Unit test that checks if a value rror is thrown when a incorrect value is supplied. """
        input = self.sig_slope  # pylint: disable=W0622
        with pytest.raises(ValueError):
            plot.detrend(input, axis=1)

    @staticmethod
    def test_detrend_linear_0d_zeros():
        """Unit test for linear 0d detrend usage. """
        input = 0.  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend_linear(input)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_linear_0d_off():
        """Detrend linear unit test.  """
        input = 5.5  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend_linear(input)
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_str_linear_0d_off():
        """Unit test for checking str_linear_0d. """
        input = 5.5  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key='linear')
        assert_almost_equal(res, targ)

    @staticmethod
    def test_detrend_detrend_linear_0d_off():
        """Unit test for linear 0d. """
        input = 5.5  # pylint: disable=W0622
        targ = 0.
        res = plot.detrend(input, key=plot.detrend_linear)
        assert_almost_equal(res, targ)

    def test_detrend_linear_1d_off(self):
        """Unit test for linear 1d. """
        input = self.sig_off  # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope(self):
        """Unit test for 1d slope. """
        input = self.sig_slope  # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope_off(self):
        """Unit test for 1d slope off. """
        input = self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_linear(input)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_str_linear_1d_slope_off(self):
        """Unit test for 1d slope off. """
        input = self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend(input, key='linear')
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_detrend_linear_1d_slope_off(self):
        """Unit test for linear 1d slope off. """
        input = self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend(input, key=plot.detrend_linear)
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_1d_slope_off_list(self):
        """Unit test for 1d slope off list. """
        input = self.sig_slope + self.sig_off  # pylint: disable=W0622
        targ = self.sig_zeros
        res = plot.detrend_linear(input.tolist())
        assert_allclose(res, targ, atol=self.atol)

    def test_detrend_linear_2d_value_error(self):
        """Unit test that checks if a value error is thrown for linear 2d functions.  """
        input = self.sig_slope[np.newaxis]  # pylint: disable=W0622
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
        input = np.vstack(arri).T  # pylint: disable=W0622
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
        input = np.vstack(arri).T  # pylint: disable=W0622
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
        input = np.vstack(arri)  # pylint: disable=W0622
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
        input = np.vstack(arri)  # pylint: disable=W0622
        targ = np.vstack(arrt)
        res = plot.detrend(input, key=plot.detrend_linear, axis=1)
        assert_allclose(res, targ, atol=self.atol)


class TestWindow:
    """
        Testclass for testing all the required window plot functions.
     """

    def setup(self):
        """Generates required mock data for testing. """
        np.random.seed(0)
        n_times = 1000

        self.sig_rand = np.random.standard_normal(n_times) + 100. # pylint: disable=W0201
        self.sig_ones = np.ones(n_times) # pylint: disable=W0201

    @staticmethod
    def check_window_apply_repeat(x_axis, window, bins_for_fft, noverlap):
        '''This is an adaptation of the original window application
        algorithm.  This is here to test to make sure the new implementation
        has the same result'''
        step = bins_for_fft - noverlap
        ind = np.arange(0, len(x_axis) - bins_for_fft + 1, step)
        n_times = len(ind)
        result = np.zeros((bins_for_fft, n_times))

        if np.iterable(window):
            window_values = window
        else:
            window_values = window(np.ones((bins_for_fft,), x_axis.dtype))

        # do the ffts of the slices
        for i in range(n_times):
            result[:, i] = window_values * x_axis[ind[i]:ind[i] + bins_for_fft]
        return result

    def test_window_hanning_rand(self):
        """Test for random hanning window. """
        targ = np.hanning(len(self.sig_rand)) * self.sig_rand
        res = plot.window_hanning(self.sig_rand)

        assert_allclose(targ, res, atol=1e-06)

    def test_window_hanning_ones(self):
        """Test for one_hanning_windows. """
        targ = np.hanning(len(self.sig_ones))
        res = plot.window_hanning(self.sig_ones)

        assert_allclose(targ, res, atol=1e-06)

    def test_apply_window_1d_axis1_value_error(self):
        """
        Test for apply_window on a single axis to check
         if a value error is thrown is the wrong value is supplied.
        """
        x_axis = self.sig_rand
        window = plot.window_hanning
        with pytest.raises(ValueError):
            plot.apply_window(x_axis, window, axis=1, return_window=False)

    def test_apply_window_1d_els_wrong_size_value_error(self):
        """Test for apply_window if a error is thrown when the wrong value is supplied. """
        x_axis = self.sig_rand
        window = plot.window_hanning(np.ones(x_axis.shape[0] - 1))
        with pytest.raises(ValueError):
            plot.apply_window(x_axis, window)

    @staticmethod
    def test_apply_window_0d_value_error():
        """Test apply_window 0d if a error is thrown when a incorrect value is supplied. """
        x_axis = np.array(0)
        window = plot.window_hanning
        with pytest.raises(ValueError):
            plot.apply_window(x_axis, window, axis=1, return_window=False)

    def test_apply_window_3d_value_error(self):
        """Test apply window 3 dimensional error is thrown when a incorrect value is supplied. """
        x_axis = self.sig_rand[np.newaxis][np.newaxis]
        window = plot.window_hanning
        with pytest.raises(ValueError):
            plot.apply_window(x_axis, window, axis=1, return_window=False)

    def test_apply_window_hanning_1d(self):
        """Test for the apply_window function in one dimension"""
        x_axis = self.sig_rand
        window = plot.window_hanning
        second_window = plot.window_hanning(np.ones(x_axis.shape[0]))
        y_axis, window2 = plot.apply_window(x_axis, window, return_window=True)
        y_time_steps = window(x_axis)
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)
        assert_array_equal(second_window, window2)

    def test_apply_window_hanning_1d_axis0(self):
        """Test for apply_window with a 1 dimensional axis. """
        x_axis = self.sig_rand
        window = plot.window_hanning
        y_axis = plot.apply_window(x_axis, window, axis=0, return_window=False)
        y_time_steps = window(x_axis)
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)

    def test_apply_window_hanning_els_1d_axis0(self):
        """Test for apply_window_hanning_els 1 dimensional axis. """
        x_axis = self.sig_rand
        window = plot.window_hanning(np.ones(x_axis.shape[0]))
        second_window = plot.window_hanning
        y_axis = plot.apply_window(x_axis, window, axis=0, return_window=False)
        y_time_steps = second_window(x_axis)
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)

    @staticmethod
    def test_apply_window_hanning_2d_axis0():
        """Test for apply_window with a two dimensional axis. """
        x_axis = np.random.standard_normal([1000, 10]) + 100.
        window = plot.window_hanning
        y_axis = plot.apply_window(x_axis, window, axis=0, return_window=False)
        y_time_steps = np.zeros_like(x_axis)
        for i in range(x_axis.shape[1]):
            y_time_steps[:, i] = window(x_axis[:, i])
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)

    @staticmethod
    def test_apply_window_hanning_els1_2d_axis0():
        """Test for apply_window els1 in 2 dimensions"""
        x_axis = np.random.standard_normal([1000, 10]) + 100.
        window = plot.window_hanning(np.ones(x_axis.shape[0]))
        second_window = plot.window_hanning
        y_axis = plot.apply_window(x_axis, window, axis=0, return_window=False)
        y_time_steps = np.zeros_like(x_axis)
        for i in range(x_axis.shape[1]):
            y_time_steps[:, i] = second_window(x_axis[:, i])
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)

    @staticmethod
    def test_apply_window_hanning_els2_2d_axis0():
        """Test for function apply_window_hanning els in 2 dimensions"""
        x_axis = np.random.standard_normal([1000, 10]) + 100.
        window = plot.window_hanning
        second_window = plot.window_hanning(np.ones(x_axis.shape[0]))
        y_axis, third_window = plot.apply_window(x_axis, window, axis=0, return_window=True)
        y_time_steps = np.zeros_like(x_axis)
        for i in range(x_axis.shape[1]):
            y_time_steps[:, i] = second_window * x_axis[:, i]
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)
        assert_array_equal(second_window, third_window)

    @staticmethod
    def test_apply_window_hanning_els3_2d_axis0():
        """Test for apply_window_hanning_els3 in 2 dimensions. """
        x_axis = np.random.standard_normal([1000, 10]) + 100.
        window = plot.window_hanning
        second_window = plot.window_hanning(np.ones(x_axis.shape[0]))
        y_axis, third_window = plot.apply_window(x_axis, window, axis=0, return_window=True)
        y_time_steps = plot.apply_window(x_axis, second_window, axis=0, return_window=False)
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)
        assert_array_equal(second_window, third_window)

    @staticmethod
    def test_apply_window_hanning_2d_axis1():
        """Test for function apply_window_hanning in 2 dimensions. """
        x_axis = np.random.standard_normal([10, 1000]) + 100.
        window = plot.window_hanning
        y_axis = plot.apply_window(x_axis, window, axis=1, return_window=False)
        y_time_steps = np.zeros_like(x_axis)
        for i in range(x_axis.shape[0]):
            y_time_steps[i, :] = window(x_axis[i, :])
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)

    @staticmethod
    def test_apply_window_hanning_2d__els1_axis1():
        """Test apply_window_hanning in 2 dimensions els1_axis1"""
        x_axis = np.random.standard_normal([10, 1000]) + 100.
        window = plot.window_hanning(np.ones(x_axis.shape[1]))
        second_window = plot.window_hanning
        y_axis = plot.apply_window(x_axis, window, axis=1, return_window=False)
        y_time_steps = np.zeros_like(x_axis)
        for i in range(x_axis.shape[0]):
            y_time_steps[i, :] = second_window(x_axis[i, :])
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)

    @staticmethod
    def test_apply_window_hanning_2d_els2_axis1():
        """Test apply_window_hanning function in 2dimensions els2_axis1"""
        x_axis = np.random.standard_normal([10, 1000]) + 100.
        window = plot.window_hanning
        second_window = plot.window_hanning(np.ones(x_axis.shape[1]))
        y_axis, third_window = plot.apply_window(x_axis, window, axis=1, return_window=True)
        y_time_steps = np.zeros_like(x_axis)
        for i in range(x_axis.shape[0]):
            y_time_steps[i, :] = second_window * x_axis[i, :]
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)
        assert_array_equal(second_window, third_window)

    @staticmethod
    def test_apply_window_hanning_2d_els3_axis1():
        """Test apply_window_hanning function in 2 dimensions. """
        x_axis = np.random.standard_normal([10, 1000]) + 100.
        window = plot.window_hanning
        second_window = plot.window_hanning(np.ones(x_axis.shape[1]))
        y_axis = plot.apply_window(x_axis, window, axis=1, return_window=False)
        y_time_steps = plot.apply_window(x_axis, second_window, axis=1, return_window=False)
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape == y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)

    def test_apply_window_stride_windows_hanning_2d_n13_noverlapn3_axis0(self):
        """Test for apply_window_stride_windows_hanning in 2d with no overlap. """
        x_axis = self.sig_rand
        window = plot.window_hanning
        y_axis_i = plot.stride_windows(x_axis, sample_size=13, noverlap=2, axis=0)
        y_axis = plot.apply_window(y_axis_i, window, axis=0, return_window=False)
        y_time_steps = self.check_window_apply_repeat(x_axis, window, 13, 2)
        assert y_time_steps.shape == y_axis.shape
        assert x_axis.shape != y_axis.shape
        assert_allclose(y_time_steps, y_axis, atol=1e-06)

    @staticmethod
    def test_apply_window_hanning_2d_stack_axis1():
        """Test test_apply_widnow_hanning function in 2 dimensions"""
        y_axis_data = np.arange(32)
        y_axis_data_two = y_axis_data + 5
        y_axis_data_three = y_axis_data + 3.3
        y_axis_control_1 = plot.apply_window(y_axis_data_two, plot.window_hanning)
        y_axis_control_2 = plot.window_hanning(y_axis_data_three)
        y_axis_data = np.vstack([y_axis_data_two, y_axis_data_three])
        y_axis_control = np.vstack([y_axis_control_1, y_axis_control_2])
        y_axis_data = np.tile(y_axis_data, (20, 1))
        y_axis_control = np.tile(y_axis_control, (20, 1))
        result = plot.apply_window(y_axis_data, plot.window_hanning, axis=1,
                                   return_window=False)
        assert_allclose(y_axis_control, result, atol=1e-08)

    @staticmethod
    def test_apply_window_hanning_2d_stack_windows_axis1():
        """Test for apply_window_hanning in 2 dimensions with stacked windows and axis1"""
        y_axis_data = np.arange(32)
        y_axis_data_two = y_axis_data + 5
        y_axis_data_three = y_axis_data + 3.3
        y_axis_control_1 = plot.apply_window(y_axis_data_two, plot.window_hanning)
        y_axis_control_2 = plot.window_hanning(y_axis_data_three)
        y_axis_data = np.vstack([y_axis_data_two, y_axis_data_three])
        y_axis_control = np.vstack([y_axis_control_1, y_axis_control_2])
        y_axis_data = np.tile(y_axis_data, (20, 1))
        y_axis_control = np.tile(y_axis_control, (20, 1))
        result = plot.apply_window(y_axis_data, plot.window_hanning, axis=1,
                                   return_window=False)
        assert_allclose(y_axis_control, result, atol=1e-08)

    @staticmethod
    def test_apply_window_hanning_2d_stack_windows_axis1_unflatten():
        """Test for the apply_window_hanning function in 2 dimensions
            and stacked windows with 1 axis. """
        n_time_steps = 32
        y_axis_data = np.arange(n_time_steps)
        y_axis_data_1 = y_axis_data + 5
        y_axis_data_2 = y_axis_data + 3.3
        y_axis_control_1 = plot.apply_window(y_axis_data_1, plot.window_hanning)
        y_axis_control_2 = plot.window_hanning(y_axis_data_2)
        y_axis_data = np.vstack([y_axis_data_1, y_axis_data_2])
        y_axis_control = np.vstack([y_axis_control_1, y_axis_control_2])
        y_axis_data = np.tile(y_axis_data, (20, 1))
        y_axis_control = np.tile(y_axis_control, (20, 1))
        y_axis_data = y_axis_data.flatten()
        y_axis_data_1 = plot.stride_windows(y_axis_data, 32, noverlap=0, axis=0)
        result = plot.apply_window(y_axis_data_1, plot.window_hanning, axis=0,
                                   return_window=False)
        assert_allclose(y_axis_control.T, result, atol=1e-08)
