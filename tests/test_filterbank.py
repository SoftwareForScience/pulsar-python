"""
    filterbank.py tests
"""

import unittest
from .context.filterbank import Filterbank

class TestFilterbank(unittest.TestCase):
    """
        Class for testing filterbank.py
    """
    def test_wrong_filename_raises_error(self):
        """
            Initialize filterbank with an incorrect filename
        """
        filename = './thispathdoesnotexist'
        with self.assertRaises(FileNotFoundError):
            Filterbank(filename)



    def test_select_frequency_range(self):
        """
            Initialize 8 bits filterbank
            and test if values are in frequency range
        """
        filename = './pspm8.fil'
        fil = Filterbank(filename)
        data = fil.select_data(freq_start=431, freq_stop=432)
        self.assertTrue(all(430.5 < i < 432.4  for i in data[0]))

    def test_select_frequency_range_reversed(self):
        """
            Initialize 16 bits filterbank
            and test if values are in frequency range
        """
        filename = './pspm16.fil'
        fil = Filterbank(filename)
        data = fil.select_data(freq_start=432, freq_stop=431)
        self.assertTrue(all(430.5 < i < 432.4  for i in data[0]))

    def test_filterbank_time_range(self):
        """
            Initialize 8 bits filterbank
            and test if values are in time range
        """
        filename = './pspm8.fil'
        time_range = (10, 30)
        time_delt = abs(time_range[0] - time_range[1])
        fil = Filterbank(filename)
        data = fil.select_data(time_start=time_range[0], time_stop=time_range[1])
        self.assertEqual(len(data[1]), time_delt)

    def test_filterbank_parameters(self):
        """
            Initialize 32 bits filterbank
            and test if all parameters work
        """
        filename = './pspm32.fil'
        freq_range = (433, 435)
        time_range = (10, 20)
        time_delt = abs(time_range[0] - time_range[1])
        fil = Filterbank(filename, freq_range=freq_range, time_range=time_range)
        data = fil.select_data()
        self.assertTrue(all(432.5 < i < 435.4  for i in data[0]))
        self.assertEqual(len(data[1]), time_delt)

if __name__ == '__main__':
    unittest.main()
